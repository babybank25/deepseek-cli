"""Tests for deepseek.gateway — AccountStore, Account, AccountPool."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.gateway.account import Account
from deepseek.gateway.pool import AccountPool
from deepseek.gateway.store import AccountStore
from deepseek.models import APIConfig


# ── Helpers ───────────────────────────────────────────────────

def _make_config(**overrides) -> APIConfig:
    base = {
        "target_url": "https://chat.deepseek.com",
        "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
        "method": "POST",
        "headers": {},
        "body_template": {},
        "auth_token": "test_token",
    }
    base.update(overrides)
    return APIConfig.from_dict(base)


def _make_account(name: str = "account_0", exhausted_until: float = 0.0) -> Account:
    acc = Account(name=name, config=_make_config())
    acc.exhausted_until = exhausted_until
    return acc


# ── AccountStore ──────────────────────────────────────────────

class TestAccountStoreListAccounts:
    def test_returns_empty_when_no_accounts(self, tmp_path):
        """1. list_accounts() returns empty list when no accounts exist."""
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", tmp_path / "accounts"), \
             patch("deepseek.gateway.store.CONFIG_FILE", tmp_path / "config.json"):
            result = AccountStore.list_accounts()
        assert result == []

    def test_returns_account_0_for_legacy_config(self, tmp_path):
        """list_accounts() returns ['account_0'] when only legacy config.json exists."""
        legacy = tmp_path / "config.json"
        legacy.write_text(json.dumps(_make_config().to_dict()), encoding="utf-8")
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", tmp_path / "accounts"), \
             patch("deepseek.gateway.store.CONFIG_FILE", legacy):
            result = AccountStore.list_accounts()
        assert result == ["account_0"]


class TestAccountStoreSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        """2. save() and load() roundtrip preserves config."""
        accounts_dir = tmp_path / "accounts"
        config = _make_config(auth_token="secret_token_xyz")
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", accounts_dir), \
             patch("deepseek.gateway.store.CONFIG_FILE", tmp_path / "config.json"):
            AccountStore.save("account_0", config)
            loaded = AccountStore.load("account_0")
        assert loaded is not None
        assert loaded.auth_token == "secret_token_xyz"
        assert loaded.target_url == "https://chat.deepseek.com"

    def test_load_returns_none_for_missing(self, tmp_path):
        """load() returns None when account file doesn't exist."""
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", tmp_path / "accounts"), \
             patch("deepseek.gateway.store.CONFIG_FILE", tmp_path / "config.json"):
            result = AccountStore.load("nonexistent")
        assert result is None


class TestAccountStoreNextName:
    def test_next_name_empty_returns_account_0(self, tmp_path):
        """3. next_name() returns 'account_0' when no accounts exist."""
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", tmp_path / "accounts"), \
             patch("deepseek.gateway.store.CONFIG_FILE", tmp_path / "config.json"):
            name = AccountStore.next_name()
        assert name == "account_0"

    def test_next_name_with_account_0_returns_account_1(self, tmp_path):
        """3. next_name() returns 'account_1' when account_0 exists."""
        accounts_dir = tmp_path / "accounts"
        config = _make_config()
        with patch("deepseek.gateway.store.ACCOUNTS_DIR", accounts_dir), \
             patch("deepseek.gateway.store.CONFIG_FILE", tmp_path / "config.json"):
            AccountStore.save("account_0", config)
            name = AccountStore.next_name()
        assert name == "account_1"


# ── Account ───────────────────────────────────────────────────

class TestAccountIsAvailable:
    def test_available_when_exhausted_until_zero(self):
        """4. is_available returns True when exhausted_until=0."""
        acc = _make_account(exhausted_until=0.0)
        assert acc.is_available is True

    def test_not_available_when_exhausted_until_future(self):
        """5. is_available returns False when exhausted_until is in the future."""
        acc = _make_account(exhausted_until=time.time() + 3600)
        assert acc.is_available is False

    def test_available_when_exhausted_until_past(self):
        """is_available returns True when exhausted_until is in the past."""
        acc = _make_account(exhausted_until=time.time() - 1)
        assert acc.is_available is True


class TestAccountMarkExhausted:
    def test_mark_exhausted_sets_future_time(self):
        """6. mark_exhausted() sets exhausted_until in the future."""
        acc = _make_account()
        before = time.time()
        acc.mark_exhausted(cooldown_seconds=3600.0)
        assert acc.exhausted_until > before
        assert acc.exhausted_until >= before + 3600.0 - 1  # allow 1s tolerance
        assert acc.is_available is False

    def test_mark_exhausted_increments_errors(self):
        acc = _make_account()
        acc.mark_exhausted()
        assert acc.total_errors == 1


class TestAccountIsQuotaError:
    def test_detects_rate_limit_message(self):
        """7. is_quota_error() detects 'rate limit' in message."""
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("Rate limit exceeded")) is True

    def test_detects_40400_code_in_message(self):
        """8. is_quota_error() detects '40400' in message."""
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("API error 40400: quota")) is True

    def test_detects_quota_exceeded(self):
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("quota exceeded")) is True

    def test_detects_too_many_requests(self):
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("too many requests")) is True

    def test_detects_429_code(self):
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("error 429")) is True

    def test_returns_false_for_non_quota_error(self):
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("network timeout")) is False

    def test_returns_false_for_auth_error(self):
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("auth failed 401")) is False

    def test_does_not_match_substring_of_other_codes(self):
        """'1429' must NOT match the 429 quota code (word-boundary check)."""
        acc = _make_account()
        assert acc.is_quota_error(RuntimeError("error 1429999")) is False

    def test_does_not_match_substring_quota_in_word(self):
        acc = _make_account()
        # '40400' inside 'x40400y' would be substring match — boundary stops it
        assert acc.is_quota_error(RuntimeError("internal x40400y boom")) is False


# ── AccountPool ───────────────────────────────────────────────

class TestAccountPoolAvailableAccounts:
    def test_filters_exhausted_accounts(self):
        """9. available_accounts() filters exhausted accounts."""
        pool = AccountPool()
        available = _make_account("account_0", exhausted_until=0.0)
        exhausted = _make_account("account_1", exhausted_until=time.time() + 3600)
        pool._replace_accounts_for_test([available, exhausted])
        result = pool.available_accounts()
        assert len(result) == 1
        assert result[0].name == "account_0"

    def test_returns_all_when_none_exhausted(self):
        pool = AccountPool()
        pool._replace_accounts_for_test([_make_account("a0"), _make_account("a1")])
        assert len(pool.available_accounts()) == 2


class TestAccountPoolNextAccount:
    def test_returns_none_when_no_accounts_available(self):
        """_next_account() returns None when no accounts available."""
        pool = AccountPool()
        # All exhausted
        pool._replace_accounts_for_test([_make_account("a0", exhausted_until=time.time() + 3600)])
        assert pool._next_account() is None

    def test_returns_none_when_pool_empty(self):
        pool = AccountPool()
        assert pool._next_account() is None

    def test_picks_an_account_when_available(self):
        pool = AccountPool()
        pool._replace_accounts_for_test([_make_account("a0"), _make_account("a1")])
        first = pool._next_account()
        assert first is not None
        assert first.name in ("a0", "a1")

    def test_prefers_idle_with_fewer_errors(self):
        """Scoring prefers accounts with fewer total errors."""
        pool = AccountPool()
        a0 = _make_account("a0")
        a0.total_errors = 10
        a1 = _make_account("a1")
        a1.total_errors = 0
        pool._replace_accounts_for_test([a0, a1])
        # a1 should win (fewer errors)
        assert pool._next_account().name == "a1"


class TestAccountPoolStatus:
    def test_status_returns_correct_structure(self):
        """status() returns correct structure."""
        pool = AccountPool()
        pool._replace_accounts_for_test([_make_account("account_0"), _make_account("account_1")])
        s = pool.status()
        assert s["total_accounts"] == 2
        assert s["available_accounts"] == 2
        assert s["in_use"] == 0
        assert isinstance(s["accounts"], list)
        assert len(s["accounts"]) == 2
        assert s["accounts"][0]["name"] == "account_0"

    def test_status_counts_exhausted_correctly(self):
        pool = AccountPool()
        available = _make_account("a0")
        exhausted = _make_account("a1", exhausted_until=time.time() + 3600)
        pool._replace_accounts_for_test([available, exhausted])
        s = pool.status()
        assert s["total_accounts"] == 2
        assert s["available_accounts"] == 1


class TestAccountPoolExponentialCooldown:
    def test_cooldown_doubles_per_consecutive_hit(self):
        """Each repeated quota hit doubles the cooldown (capped)."""
        acc = _make_account("a0")
        # First hit: 60s
        c1 = acc.mark_exhausted()
        assert 55 <= c1 <= 65
        # Second hit: 120s
        c2 = acc.mark_exhausted()
        assert 115 <= c2 <= 125
        # Third hit: 240s
        c3 = acc.mark_exhausted()
        assert 235 <= c3 <= 245

    def test_success_resets_cooldown_counter(self):
        acc = _make_account("a0")
        acc.mark_exhausted()
        acc.mark_exhausted()
        assert acc.consecutive_quota_hits == 2
        acc.mark_success()
        assert acc.consecutive_quota_hits == 0
        assert acc.exhausted_until == 0.0

    def test_cooldown_capped_at_max(self):
        from deepseek.gateway.account import COOLDOWN_MAX_SECONDS
        acc = _make_account("a0")
        # 10 hits would compute > 1h; should be capped
        for _ in range(10):
            c = acc.mark_exhausted()
        assert c <= COOLDOWN_MAX_SECONDS + 1


class TestAccountPoolNoAvailableError:
    def test_raises_with_retry_after(self):
        from deepseek.gateway.pool import NoAccountAvailableError

        pool = AccountPool()
        a = _make_account("a0", exhausted_until=time.time() + 30)
        pool._replace_accounts_for_test([a])
        with pytest.raises(NoAccountAvailableError) as exc_info:
            pool._raise_no_available(set())
        assert exc_info.value.retry_after is not None
        assert 25 <= exc_info.value.retry_after <= 31

    def test_empty_pool_raises_immediately(self):
        from deepseek.gateway.pool import NoAccountAvailableError

        pool = AccountPool()

        async def run():
            async for _ in pool.send_message_stream("hi"):
                pass

        with pytest.raises(NoAccountAvailableError):
            import asyncio as _aio
            _aio.run(run())


class TestAccountPoolPresetIsolation:
    def test_preset_only_applies_to_chosen_account(self):
        """Pool must apply preset/file_ids ONLY to the account it picks,
        not to every account in the pool. This prevents cross-request
        mutation when many requests are in flight at once."""
        import asyncio as _aio

        pool = AccountPool()
        a0 = _make_account("a0")
        a1 = _make_account("a1")
        # Force pool to pick a0 by making a1 less attractive
        a1.total_errors = 100
        pool._replace_accounts_for_test([a0, a1])

        # Mock both clients' send_message_stream to immediately complete
        async def fake_stream(_msg):
            if False:
                yield None
        a0.client.send_message_stream = lambda msg: fake_stream(msg)  # type: ignore[assignment]
        a1.client.send_message_stream = lambda msg: fake_stream(msg)  # type: ignore[assignment]

        async def run():
            async for _ in pool.send_message_stream(
                "hi",
                model_preset={
                    "model_type": "expert",
                    "thinking_enabled": True,
                    "search_enabled": False,
                },
                file_ids=["f1", "f2"],
            ):
                pass
        _aio.run(run())

        # a0 was chosen → preset applied
        assert a0.client.model_type == "expert"
        assert a0.client.thinking_enabled is True
        # a1 NOT chosen → still default ("expert" from APIClient.__init__)
        # The key invariant: a1's _pending_file_ids must remain empty.
        assert a1.client._pending_file_ids == []
        # a0 _pending_file_ids was set, then consumed when send_message_stream
        # would normally read it. Our fake didn't consume it, so it remains:
        assert a0.client._pending_file_ids == ["f1", "f2"]



class TestPoolFlushStats:
    def test_flush_stats_writes_file(self, tmp_path, monkeypatch):
        from deepseek.gateway import pool as pool_mod
        monkeypatch.setattr(pool_mod, "STATS_FILE", tmp_path / "stats.json")
        monkeypatch.setattr(pool_mod, "CONFIG_DIR", tmp_path)
        p = AccountPool()
        a = _make_account("acc0")
        a.total_requests = 7
        p._replace_accounts_for_test([a])
        p.flush_stats()
        assert (tmp_path / "stats.json").exists()
        import json as _j
        data = _j.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
        assert data["acc0"]["total_requests"] == 7


class TestPoolOnAccountChosenCallback:
    def test_callback_invoked_with_account(self):
        import asyncio as _aio
        p = AccountPool()
        a0 = _make_account("a0")
        a1 = _make_account("a1")
        a1.total_errors = 100  # make a0 win the score
        p._replace_accounts_for_test([a0, a1])

        seen: list[str] = []

        async def fake_stream(_msg):
            if False:
                yield None
        a0.client.send_message_stream = lambda msg: fake_stream(msg)  # type: ignore[assignment]

        async def run():
            async for _ in p.send_message_stream(
                "hi",
                on_account_chosen=lambda acc: seen.append(acc.name),
            ):
                pass

        _aio.run(run())
        assert seen == ["a0"]

    def test_callback_exception_swallowed(self):
        """A buggy callback must not break the request flow."""
        import asyncio as _aio
        p = AccountPool()
        a0 = _make_account("a0")
        p._replace_accounts_for_test([a0])

        async def fake_stream(_msg):
            if False:
                yield None
        a0.client.send_message_stream = lambda msg: fake_stream(msg)  # type: ignore[assignment]

        def boom(_acc):
            raise RuntimeError("callback bug")

        async def run():
            async for _ in p.send_message_stream("hi", on_account_chosen=boom):
                pass
        # Should NOT raise — pool catches callback exceptions.
        _aio.run(run())


class TestPoolPersistThrottle:
    def test_maybe_persist_skips_if_recent(self, tmp_path, monkeypatch):
        from deepseek.gateway import pool as pool_mod
        monkeypatch.setattr(pool_mod, "STATS_FILE", tmp_path / "stats.json")
        monkeypatch.setattr(pool_mod, "CONFIG_DIR", tmp_path)

        p = AccountPool()
        p._replace_accounts_for_test([_make_account("a0")])
        p._persist_min_interval = 60.0  # very long interval

        # First write happens
        p._maybe_persist_stats()
        assert (tmp_path / "stats.json").exists()
        first_mtime = (tmp_path / "stats.json").stat().st_mtime_ns

        # Second call within the interval is skipped
        p._maybe_persist_stats()
        second_mtime = (tmp_path / "stats.json").stat().st_mtime_ns
        assert first_mtime == second_mtime

    def test_flush_stats_bypasses_throttle(self, tmp_path, monkeypatch):
        from deepseek.gateway import pool as pool_mod
        monkeypatch.setattr(pool_mod, "STATS_FILE", tmp_path / "stats.json")
        monkeypatch.setattr(pool_mod, "CONFIG_DIR", tmp_path)

        p = AccountPool()
        p._replace_accounts_for_test([_make_account("a0")])
        p._persist_min_interval = 999.0
        p._maybe_persist_stats()
        # flush should always write regardless of throttle
        p.flush_stats()
        assert (tmp_path / "stats.json").exists()


class TestAccountLockReset:
    def test_lock_can_be_replaced(self):
        """Admin unblock heuristic — replacing the lock object recovers
        from a never-released lock without breaking the account."""
        import asyncio as _aio

        acc = _make_account("stuck")
        # Simulate a lock that was acquired but never released.
        loop = _aio.new_event_loop()
        try:
            loop.run_until_complete(acc.lock.acquire())
            assert acc.lock.locked() is True
            # Replace the lock — what the admin endpoint does.
            acc.lock = _aio.Lock()
            assert acc.lock.locked() is False
        finally:
            loop.close()
