"""Smart multi-account routing with persistent conversation affinity."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from typing import AsyncGenerator, Callable, Optional

from ..client import APIClient
from ..constants import CONFIG_DIR
from ..exceptions import AuthExpiredError
from ..models import APIConfig
from .account import Account
from .bindings import ConversationBinding, ConversationBindingStore
from .store import AccountStore

logger = logging.getLogger(__name__)

Token = tuple[str, str]
STATS_FILE = CONFIG_DIR / "pool_stats.json"
ACQUIRE_TIMEOUT_SECONDS = 90.0


class NoAccountAvailableError(RuntimeError):
    """Raised when no safe account is available for a request."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AccountPool:
    """Route requests while keeping each conversation on one account/session."""

    def __init__(
        self,
        binding_store: Optional[ConversationBindingStore] = None,
    ) -> None:
        self._accounts: list[Account] = []
        self._registry_lock = asyncio.Lock()
        self._last_persist_at = 0.0
        self._persist_min_interval = 5.0
        self._bindings = binding_store or ConversationBindingStore()

    # ── Loading and persistence ───────────────────────────────

    def load_all(self) -> int:
        names = AccountStore.list_accounts()
        self._accounts = []
        for name in names:
            cfg = AccountStore.load(name)
            if cfg:
                self._accounts.append(Account(name=name, config=cfg))
                logger.info("Loaded account: %s", name)
        self._restore_stats()
        self._bindings.reload({account.name for account in self._accounts})
        return len(self._accounts)

    def _restore_stats(self) -> None:
        if not STATS_FILE.exists():
            return
        try:
            data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for acc in self._accounts:
            state = data.get(acc.name)
            if not isinstance(state, dict):
                continue
            try:
                exhausted_until = float(state.get("exhausted_until") or 0)
                if exhausted_until > time.time():
                    acc.exhausted_until = exhausted_until
                acc.consecutive_quota_hits = int(
                    state.get("consecutive_quota_hits") or 0
                )
                acc.total_requests = int(state.get("total_requests") or 0)
                acc.total_errors = int(state.get("total_errors") or 0)
                acc.last_used = float(state.get("last_used") or 0)
                acc.last_error = state.get("last_error")
            except (TypeError, ValueError):
                continue

    def _persist_stats(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                acc.name: {
                    "exhausted_until": acc.exhausted_until,
                    "consecutive_quota_hits": acc.consecutive_quota_hits,
                    "total_requests": acc.total_requests,
                    "total_errors": acc.total_errors,
                    "last_used": acc.last_used,
                    "last_error": acc.last_error,
                }
                for acc in self._accounts
            }
            tmp = STATS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(STATS_FILE)
            self._last_persist_at = time.monotonic()
        except OSError:
            pass

    def flush_stats(self) -> None:
        self._persist_stats()
        self._bindings.flush()

    def _maybe_persist_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_persist_at < self._persist_min_interval:
            return
        self._last_persist_at = now
        self._persist_stats()

    # ── Account registry ──────────────────────────────────────

    async def add_account(self, name: str, config: APIConfig) -> Account:
        AccountStore.save(name, config)
        account = Account(name=name, config=config)
        async with self._registry_lock:
            self._accounts.append(account)
        return account

    async def remove_account(self, name: str) -> bool:
        async with self._registry_lock:
            target = next((a for a in self._accounts if a.name == name), None)
            if target is not None:
                self._accounts = [a for a in self._accounts if a.name != name]
        self._bindings.delete_account(name)
        if target is not None:
            await target.close()
        return AccountStore.delete(name)

    def available_accounts(self) -> list[Account]:
        return [account for account in self._accounts if account.is_available]

    @property
    def accounts(self) -> list[Account]:
        return list(self._accounts)

    def is_empty(self) -> bool:
        return not self._accounts

    def _replace_accounts_for_test(self, accounts: list[Account]) -> None:
        self._accounts = list(accounts)

    # ── Selection ─────────────────────────────────────────────

    def _score(self, acc: Account) -> tuple:
        return (
            1 if acc.lock.locked() else 0,
            acc.consecutive_quota_hits,
            acc.total_errors,
            acc.last_used,
            random.random(),
        )

    def _pick(self, exclude: set[str]) -> Optional[Account]:
        candidates = [
            account
            for account in self._accounts
            if account.is_available and account.name not in exclude
        ]
        return min(candidates, key=self._score) if candidates else None

    def _next_account(self) -> Optional[Account]:
        return self._pick(set())

    def _account_by_name(self, name: str) -> Optional[Account]:
        return next((account for account in self._accounts if account.name == name), None)

    # ── Conversation lifecycle ────────────────────────────────

    def list_conversations(self) -> list[dict]:
        return self._bindings.list()

    async def delete_conversation(self, conversation_id: str) -> bool:
        binding = self._bindings.delete(conversation_id)
        if binding is None:
            return False
        account = self._account_by_name(binding.account_name)
        if account is not None:
            await account.drop_conversation(conversation_id)
        return True

    async def compact_conversation(self, conversation_id: str) -> bool:
        binding = self._bindings.get(conversation_id)
        if binding is None:
            return False
        account = await self._acquire_bound(binding)
        try:
            client = account.client_for(conversation_id, binding)
            compacted = await client.compact_session()
            if compacted:
                self._bindings.update_from_client(
                    conversation_id,
                    account.name,
                    client,
                )
            return compacted
        finally:
            if account.lock.locked():
                account.lock.release()

    # ── Streaming ─────────────────────────────────────────────

    async def send_message_stream(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        model_preset: Optional[dict] = None,
        file_ids: Optional[list[str]] = None,
        reset_session: bool = False,
        on_account_chosen: Optional[Callable[[Account], None]] = None,
        on_client_chosen: Optional[Callable[[Account, APIClient], None]] = None,
    ) -> AsyncGenerator[Token, None]:
        """Send through a healthy account without silently migrating a conversation.

        New conversations may fail over before any output is emitted. Once a
        conversation has a persisted binding, quota or availability failures are
        surfaced instead of moving it to another account and losing context.
        """
        rid = request_id or f"req_{int(time.time() * 1000)}"
        binding = self._bindings.get(conversation_id)
        excluded: set[str] = set()

        while True:
            was_bound = binding is not None
            if binding is not None:
                account = await self._acquire_bound(binding)
            else:
                account = await self._acquire_account(excluded)

            client = account.client_for(conversation_id, binding)
            try:
                if reset_session:
                    await client.reset_session()
                self._apply_request_options(client, model_preset, file_ids)
                self._notify_choice(
                    account,
                    client,
                    on_account_chosen,
                    on_client_chosen,
                )

                logger.info(
                    "[%s] using account=%s conversation=%s",
                    rid,
                    account.name,
                    conversation_id or "<stateless>",
                )
                streamed_any = False
                try:
                    async for token in client.send_message_stream(message):
                        streamed_any = True
                        yield token
                    account.mark_success()
                    if conversation_id:
                        self._bindings.update_from_client(
                            conversation_id,
                            account.name,
                            client,
                        )
                    self._maybe_persist_stats()
                    return
                except AuthExpiredError:
                    account.mark_error(Exception("auth expired"))
                    self._persist_stats()
                    if conversation_id and not was_bound:
                        await account.drop_conversation(conversation_id)
                    raise
                except Exception as error:
                    if streamed_any:
                        account.mark_error(error)
                        self._persist_stats()
                        raise
                    if account.is_quota_error(error):
                        cooldown = account.mark_exhausted()
                        account.last_error = str(error)[:200]
                        self._persist_stats()
                        if was_bound:
                            raise NoAccountAvailableError(
                                f"Conversation '{conversation_id}' is pinned to "
                                f"quota-exhausted account '{account.name}'.",
                                retry_after=cooldown,
                            ) from error
                        if conversation_id:
                            await account.drop_conversation(conversation_id)
                        excluded.add(account.name)
                        logger.warning(
                            "[%s] %s quota-exhausted (cooldown %ds): %s",
                            rid,
                            account.name,
                            int(cooldown),
                            error,
                        )
                        continue
                    account.mark_error(error)
                    self._persist_stats()
                    if conversation_id and not was_bound:
                        await account.drop_conversation(conversation_id)
                    raise
            finally:
                if account.lock.locked():
                    account.lock.release()

    @staticmethod
    def _apply_request_options(
        client: APIClient,
        model_preset: Optional[dict],
        file_ids: Optional[list[str]],
    ) -> None:
        if model_preset:
            if "model_type" in model_preset:
                client.model_type = model_preset["model_type"]
            if "thinking_enabled" in model_preset:
                client.thinking_enabled = model_preset["thinking_enabled"]
            if "search_enabled" in model_preset:
                client.search_enabled = model_preset["search_enabled"]
            if "auto_compact_threshold" in model_preset:
                client.auto_compact_threshold = model_preset["auto_compact_threshold"]
        if file_ids:
            client.set_pending_files(file_ids)

    @staticmethod
    def _notify_choice(
        account: Account,
        client: APIClient,
        on_account_chosen: Optional[Callable[[Account], None]],
        on_client_chosen: Optional[Callable[[Account, APIClient], None]],
    ) -> None:
        if on_account_chosen is not None:
            try:
                on_account_chosen(account)
            except Exception:
                logger.exception("on_account_chosen callback raised")
        if on_client_chosen is not None:
            try:
                on_client_chosen(account, client)
            except Exception:
                logger.exception("on_client_chosen callback raised")

    async def _acquire_bound(self, binding: ConversationBinding) -> Account:
        account = self._account_by_name(binding.account_name)
        if account is None:
            raise NoAccountAvailableError(
                f"Conversation '{binding.conversation_id}' is pinned to missing "
                f"account '{binding.account_name}'. Start a new conversation."
            )
        if not account.is_available:
            raise NoAccountAvailableError(
                f"Conversation '{binding.conversation_id}' is pinned to unavailable "
                f"account '{binding.account_name}'.",
                retry_after=account.cooldown_remaining,
            )
        try:
            await asyncio.wait_for(
                account.lock.acquire(),
                timeout=ACQUIRE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise NoAccountAvailableError(
                f"Account '{account.name}' is busy for conversation "
                f"'{binding.conversation_id}'."
            ) from error
        if not account.is_available:
            account.lock.release()
            raise NoAccountAvailableError(
                f"Account '{account.name}' became unavailable.",
                retry_after=account.cooldown_remaining,
            )
        return account

    async def _acquire_account(self, excluded: set[str]) -> Account:
        deadline = time.monotonic() + ACQUIRE_TIMEOUT_SECONDS
        while True:
            if not self._accounts:
                raise NoAccountAvailableError("No accounts configured in pool.")
            account = self._pick(excluded)
            if account is None:
                self._raise_no_available(excluded)
            assert account is not None
            if not account.lock.locked():
                await account.lock.acquire()
                if account.is_available:
                    return account
                account.lock.release()
                excluded.add(account.name)
                continue
            wait = min(0.25, max(0.05, deadline - time.monotonic()))
            if wait <= 0:
                self._raise_no_available(excluded)
            await asyncio.sleep(wait)
            if time.monotonic() >= deadline:
                self._raise_no_available(excluded)

    def _raise_no_available(self, excluded: set[str]) -> None:
        exhausted = [account for account in self._accounts if not account.is_available]
        retry_after = (
            min(account.cooldown_remaining for account in exhausted)
            if exhausted
            else None
        )
        raise NoAccountAvailableError(
            f"All {len(self._accounts)} accounts are quota-exhausted or busy "
            f"(excluded={len(excluded)}). Wait for cooldown or add more accounts.",
            retry_after=retry_after,
        )

    # ── Status and shutdown ───────────────────────────────────

    def status(self) -> dict:
        snapshot = list(self._accounts)
        return {
            "total_accounts": len(snapshot),
            "available_accounts": sum(1 for account in snapshot if account.is_available),
            "in_use": sum(1 for account in snapshot if account.lock.locked()),
            "active_conversations": len(self._bindings),
            "accounts": [account.to_status_dict() for account in snapshot],
        }

    async def close_all(self) -> None:
        self._persist_stats()
        self._bindings.flush()
        async with self._registry_lock:
            accounts = list(self._accounts)
            self._accounts.clear()
        for account in accounts:
            with contextlib.suppress(Exception):
                await account.close()
