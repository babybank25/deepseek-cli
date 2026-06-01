"""Smart account pool with per-account locking, scoring, and failover.

Design goals
------------
* **Concurrency**: requests on different accounts run in parallel (the
  pool itself is NOT a single global lock — only per-account locks).
* **Smart selection**: pick the available account with the fewest recent
  errors, breaking ties by oldest ``last_used`` plus a small jitter so
  the load balances even when stats are equal.
* **Robust failover**: on quota errors, mark the account exhausted with
  exponential backoff and try the next available one. On auth errors,
  propagate immediately. On transient network errors, the underlying
  ``APIClient`` already retries — anything that escapes is treated as
  fatal for this attempt and the account is rotated.
* **Persisted state**: ``stats.json`` records cooldowns and counters so a
  server restart doesn't lose track of which accounts are exhausted.

The pool is safe to call from many concurrent requests.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from typing import AsyncGenerator, Callable, Optional

from ..constants import CONFIG_DIR
from ..exceptions import AuthExpiredError
from ..models import APIConfig
from .account import Account
from .store import AccountStore

logger = logging.getLogger(__name__)

Token = tuple[str, str]  # (type, text)

STATS_FILE = CONFIG_DIR / "pool_stats.json"
ACQUIRE_TIMEOUT_SECONDS = 90.0  # max wait for any account to become free


class NoAccountAvailableError(RuntimeError):
    """Raised when no account is available (all locked or all exhausted).

    Carries an optional ``retry_after`` (seconds) for HTTP 503 responses.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AccountPool:
    """Pool of DeepSeek accounts with per-account locking and smart routing."""

    def __init__(self) -> None:
        self._accounts: list[Account] = []
        self._registry_lock = asyncio.Lock()  # only protects _accounts mutations
        self._last_persist_at: float = 0.0
        # Persist successful-request stats at most once per this many seconds
        # to avoid thrashing the disk under high QPS. Critical events
        # (cooldowns, auth) bypass the throttle.
        self._persist_min_interval: float = 5.0

    # ── Loading & persistence ─────────────────────────────────

    def load_all(self) -> int:
        """Load all accounts from disk + restore prior cooldown state."""
        names = AccountStore.list_accounts()
        self._accounts = []
        for name in names:
            cfg = AccountStore.load(name)
            if cfg:
                self._accounts.append(Account(name=name, config=cfg))
                logger.info("Loaded account: %s", name)
        self._restore_stats()
        return len(self._accounts)

    def _restore_stats(self) -> None:
        if not STATS_FILE.exists():
            return
        try:
            data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for acc in self._accounts:
            s = data.get(acc.name)
            if not isinstance(s, dict):
                continue
            # Only restore cooldown if it's still in the future.
            try:
                eu = float(s.get("exhausted_until") or 0)
                if eu > time.time():
                    acc.exhausted_until = eu
                acc.consecutive_quota_hits = int(s.get("consecutive_quota_hits") or 0)
                acc.total_requests = int(s.get("total_requests") or 0)
                acc.total_errors = int(s.get("total_errors") or 0)
                acc.last_used = float(s.get("last_used") or 0)
                acc.last_error = s.get("last_error")
            except (TypeError, ValueError):
                continue

    def _persist_stats(self) -> None:
        """Write current stats to disk. Best-effort; failures are silent.

        Synchronous I/O on a small JSON file (~few KB). Called from async
        contexts after each request — typical write latency on a local
        filesystem is sub-millisecond, so we don't bother off-loading to a
        thread pool. If profiling shows this is a hot path on slow disks,
        wrap the body in ``await asyncio.to_thread(...)``.
        """
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
            pass  # stats are best-effort

    def flush_stats(self) -> None:
        """Public flush hook so admin endpoints can persist after edits."""
        self._persist_stats()

    def _maybe_persist_stats(self) -> None:
        """Throttled persist: skipped if we wrote within the last N seconds."""
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
        if target is not None:
            await target.close()
        return AccountStore.delete(name)

    def available_accounts(self) -> list[Account]:
        return [a for a in self._accounts if a.is_available]

    @property
    def accounts(self) -> list[Account]:
        """Read-only snapshot of all accounts. Use this from outside the pool."""
        return list(self._accounts)

    def is_empty(self) -> bool:
        return not self._accounts

    # ── Test seam ─────────────────────────────────────────────

    def _replace_accounts_for_test(self, accounts: list[Account]) -> None:
        """White-box hook so tests can install a custom account list without
        going through the disk-backed loader. Not part of the public API."""
        self._accounts = list(accounts)

    # ── Selection ─────────────────────────────────────────────

    def _score(self, acc: Account) -> tuple:
        """Lower score = more attractive. Tie-broken by jitter for fairness."""
        return (
            1 if acc.lock.locked() else 0,    # prefer idle
            acc.consecutive_quota_hits,        # prefer healthier
            acc.total_errors,                  # prefer fewer total errors
            acc.last_used,                     # prefer least recently used
            random.random(),                   # jitter
        )

    def _pick(self, exclude: set[str]) -> Optional[Account]:
        """Pick the most attractive available account not in ``exclude``."""
        candidates = [
            a for a in self._accounts
            if a.is_available and a.name not in exclude
        ]
        if not candidates:
            return None
        return min(candidates, key=self._score)

    def _next_account(self) -> Optional[Account]:
        """Backward-compat helper for tests; uses scoring without exclusion."""
        return self._pick(exclude=set())

    # ── Streaming ─────────────────────────────────────────────

    async def send_message_stream(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        model_preset: Optional[dict] = None,
        file_ids: Optional[list[str]] = None,
        on_account_chosen: Optional[Callable[[Account], None]] = None,
    ) -> AsyncGenerator[Token, None]:
        """Send a message via the smartest available account.

        Yields ``(type, text)`` tokens. On quota-class errors, retries on
        the next available account. On auth/fatal errors, propagates.

        ``model_preset`` and ``file_ids`` are applied to the chosen account
        UNDER its lock, so concurrent requests can't trample each other's
        in-flight settings. Preset is a dict like
        ``{"model_type": "expert", "thinking_enabled": True, "search_enabled": False}``.

        ``on_account_chosen`` (optional) is invoked synchronously with the
        ``Account`` object once the pool has acquired its lock and is about
        to forward the message. Lets callers correlate request → account
        for headers like ``X-Account-Used`` or post-stream inspection of
        ``client._just_compacted``.

        Raises:
            NoAccountAvailableError: when the pool is empty or all accounts
                are exhausted/in-use beyond the acquire timeout.
            AuthExpiredError: on hard auth failures (caller must re-auth).
        """
        rid = request_id or f"req_{int(time.time()*1000)}"
        excluded: set[str] = set()
        any_account_seen = False

        while True:
            acc = await self._acquire_account(excluded)
            any_account_seen = True
            try:
                # Apply config UNDER the account lock — safe from races
                if model_preset:
                    if "model_type" in model_preset:
                        acc.client.model_type = model_preset["model_type"]
                    if "thinking_enabled" in model_preset:
                        acc.client.thinking_enabled = model_preset["thinking_enabled"]
                    if "search_enabled" in model_preset:
                        acc.client.search_enabled = model_preset["search_enabled"]
                    if "auto_compact_threshold" in model_preset:
                        acc.client.auto_compact_threshold = (
                            model_preset["auto_compact_threshold"]
                        )
                if file_ids:
                    acc.client.set_pending_files(file_ids)

                if on_account_chosen is not None:
                    try:
                        on_account_chosen(acc)
                    except Exception:
                        # Caller's callback must never break the request.
                        logger.exception("on_account_chosen callback raised")

                logger.info("[%s] using account=%s", rid, acc.name)
                streamed_any = False
                try:
                    async for token in acc.client.send_message_stream(message):
                        streamed_any = True
                        yield token
                    acc.mark_success()
                    self._maybe_persist_stats()
                    return
                except AuthExpiredError:
                    acc.mark_error(Exception("auth expired"))
                    self._persist_stats()  # auth events: persist immediately
                    raise
                except Exception as e:
                    if streamed_any:
                        # Already wrote partial output to caller — don't retry
                        # on a different account, that would duplicate text.
                        acc.mark_error(e)
                        self._persist_stats()
                        raise
                    if acc.is_quota_error(e):
                        cooldown = acc.mark_exhausted()
                        acc.last_error = str(e)[:200]
                        excluded.add(acc.name)
                        self._persist_stats()  # cooldowns: persist immediately
                        logger.warning(
                            "[%s] %s quota-exhausted (cooldown %ds): %s",
                            rid, acc.name, int(cooldown), e,
                        )
                        continue  # try next account
                    # Non-quota, non-auth error → mark + propagate
                    acc.mark_error(e)
                    self._persist_stats()
                    raise
            finally:
                if acc.lock.locked():
                    acc.lock.release()

            # unreachable
            break

        if not any_account_seen:
            raise NoAccountAvailableError("Pool is empty.")

    async def _acquire_account(self, excluded: set[str]) -> Account:
        """Pick + lock an available account. Waits up to the acquire timeout.

        The pool is busy if every available account is currently locked by
        another request; we poll/wait without blocking the event loop.
        """
        deadline = time.monotonic() + ACQUIRE_TIMEOUT_SECONDS
        while True:
            if not self._accounts:
                raise NoAccountAvailableError("No accounts configured in pool.")

            acc = self._pick(excluded)
            if acc is None:
                # All available accounts are excluded → quota cascade
                self._raise_no_available(excluded)

            # Try to take the lock without blocking the whole pool. If it's
            # busy, fall back to ``acc.lock.acquire()`` with timeout so the
            # request waits politely for a free slot.
            if not acc.lock.locked():
                await acc.lock.acquire()
                # Re-check availability under lock (may have changed)
                if acc.is_available:
                    return acc
                # If account became exhausted while we waited, release + retry
                acc.lock.release()
                excluded.add(acc.name)
                continue

            # Account is busy — wait briefly, then re-pick (might find a
            # different idle account that just freed up).
            wait = min(0.25, max(0.05, deadline - time.monotonic()))
            if wait <= 0:
                self._raise_no_available(excluded)
            await asyncio.sleep(wait)
            if time.monotonic() >= deadline:
                self._raise_no_available(excluded)

    def _raise_no_available(self, excluded: set[str]) -> None:
        # Compute the soonest cooldown end across exhausted accounts.
        exhausted = [a for a in self._accounts if not a.is_available]
        retry_after: Optional[float] = None
        if exhausted:
            retry_after = min(a.cooldown_remaining for a in exhausted)
        raise NoAccountAvailableError(
            f"All {len(self._accounts)} accounts are quota-exhausted or busy "
            f"(excluded={len(excluded)}). Wait for cooldown or add more accounts.",
            retry_after=retry_after,
        )

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict:
        snapshot = list(self._accounts)
        avail = sum(1 for a in snapshot if a.is_available)
        in_use = sum(1 for a in snapshot if a.lock.locked())
        return {
            "total_accounts": len(snapshot),
            "available_accounts": avail,
            "in_use": in_use,
            "accounts": [a.to_status_dict() for a in snapshot],
        }

    async def close_all(self) -> None:
        # Flush any pending stats before tearing down so we never lose
        # cooldown counters across a graceful restart.
        self._persist_stats()
        async with self._registry_lock:
            accounts = list(self._accounts)
            self._accounts.clear()
        for acc in accounts:
            with contextlib.suppress(Exception):
                await acc.close()
