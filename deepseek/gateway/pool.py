"""Smart multi-account routing with persistent conversation and file affinity."""
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
from ..metrics import metrics
from ..models import APIConfig
from .account import Account
from .bindings import ConversationBinding, ConversationBindingStore
from .files import FileAffinityError, FileAffinityStore
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
    """Route requests without silently breaking conversation or file lineage."""

    def __init__(
        self,
        binding_store: Optional[ConversationBindingStore] = None,
        file_store: Optional[FileAffinityStore] = None,
    ) -> None:
        self._accounts: list[Account] = []
        self._registry_lock = asyncio.Lock()
        self._last_persist_at = 0.0
        self._persist_min_interval = 5.0
        self._bindings = (
            binding_store if binding_store is not None else ConversationBindingStore()
        )
        self._files = file_store if file_store is not None else FileAffinityStore()

    def load_all(self) -> int:
        names = AccountStore.list_accounts()
        self._accounts = []
        for name in names:
            config = AccountStore.load(name)
            if config:
                self._accounts.append(Account(name=name, config=config))
                logger.info("Loaded account: %s", name)
        valid = {account.name for account in self._accounts}
        self._restore_stats()
        self._bindings.reload(valid)
        self._files.reload(valid)
        return len(self._accounts)

    def _restore_stats(self) -> None:
        if not STATS_FILE.exists():
            return
        try:
            data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for account in self._accounts:
            state = data.get(account.name)
            if not isinstance(state, dict):
                continue
            try:
                exhausted_until = float(state.get("exhausted_until") or 0)
                if exhausted_until > time.time():
                    account.exhausted_until = exhausted_until
                account.consecutive_quota_hits = int(
                    state.get("consecutive_quota_hits") or 0
                )
                account.total_requests = int(state.get("total_requests") or 0)
                account.total_errors = int(state.get("total_errors") or 0)
                account.last_used = float(state.get("last_used") or 0)
                account.last_error = state.get("last_error")
            except (TypeError, ValueError):
                continue

    def _persist_stats(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                account.name: {
                    "exhausted_until": account.exhausted_until,
                    "consecutive_quota_hits": account.consecutive_quota_hits,
                    "total_requests": account.total_requests,
                    "total_errors": account.total_errors,
                    "last_used": account.last_used,
                    "last_error": account.last_error,
                }
                for account in self._accounts
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
        if (
            self._last_persist_at > 0.0
            and now - self._last_persist_at < self._persist_min_interval
        ):
            return
        self._persist_stats()

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
        self._files.delete_account(name)
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

    def _score(self, account: Account) -> tuple:
        return (
            1 if account.lock.locked() else 0,
            account.consecutive_quota_hits,
            account.last_used,
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

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> tuple[str, str]:
        account = await self._acquire_account(set())
        try:
            file_id = await account.client.upload_file(content, filename, content_type)
            account.mark_success()
            self._files.bind(file_id, account.name)
            self._maybe_persist_stats()
            return file_id, account.name
        except Exception as error:
            account.mark_error(error)
            self._persist_stats()
            raise
        finally:
            if account.lock.locked():
                account.lock.release()

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
        """Send without moving an established conversation or uploaded file."""
        request = request_id or f"req_{int(time.time() * 1000)}"
        binding = self._bindings.get(conversation_id)
        if binding is not None:
            metrics.incr("conversation_affinity_hits")
        try:
            file_account = self._files.account_for(
                file_ids,
                require_all_known=len(self._accounts) > 1,
            )
        except FileAffinityError as error:
            raise NoAccountAvailableError(str(error)) from error
        if binding is not None and file_account and binding.account_name != file_account:
            raise NoAccountAvailableError(
                f"Conversation '{conversation_id}' is pinned to '{binding.account_name}' "
                f"but the attached file belongs to '{file_account}'."
            )

        excluded: set[str] = set()
        while True:
            was_bound = binding is not None
            route_is_pinned = was_bound or file_account is not None
            if binding is not None:
                account = await self._acquire_bound(binding)
            elif file_account:
                account = await self._acquire_named(file_account, conversation_id or "file request")
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
                    request,
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
                        if not was_bound:
                            metrics.incr("conversation_bindings_total")
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
                        if route_is_pinned:
                            raise NoAccountAvailableError(
                                f"Request is pinned to quota-exhausted account '{account.name}'.",
                                retry_after=cooldown,
                            ) from error
                        if conversation_id:
                            await account.drop_conversation(conversation_id)
                        excluded.add(account.name)
                        logger.warning(
                            "[%s] %s quota-exhausted (cooldown %ds): %s",
                            request,
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
        return await self._acquire_named(binding.account_name, binding.conversation_id)

    async def _acquire_named(self, account_name: str, label: str) -> Account:
        account = self._account_by_name(account_name)
        if account is None:
            raise NoAccountAvailableError(
                f"'{label}' is pinned to missing account '{account_name}'."
            )
        if not account.is_available:
            raise NoAccountAvailableError(
                f"'{label}' is pinned to unavailable account '{account_name}'.",
                retry_after=account.cooldown_remaining,
            )
        try:
            await asyncio.wait_for(account.lock.acquire(), timeout=ACQUIRE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as error:
            raise NoAccountAvailableError(
                f"Account '{account.name}' is busy for '{label}'."
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
