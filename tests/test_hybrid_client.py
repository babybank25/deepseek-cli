"""Tests for the hybrid auth/PoW shell layered over the v2 client core."""
from __future__ import annotations

import pytest

from deepseek._client_legacy import APIClient as LegacyAPIClient
from deepseek.client import APIClient
from deepseek.exceptions import AuthExpiredError
from deepseek.models import APIConfig
from deepseek.protocol import ProtocolCapabilities


def _config() -> APIConfig:
    return APIConfig.from_dict(
        {
            "target_url": "https://chat.deepseek.com",
            "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
            "method": "POST",
            "headers": {},
            "body_template": {},
            "auth_token": "token",
        }
    )


class FakeAuth:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.invalidated = 0
        self.recoveries = 0

    def invalidate(self) -> None:
        self.invalidated += 1

    async def recover(self) -> bool:
        self.recoveries += 1
        return self.result

    async def ensure_valid(self, *, force: bool = False) -> bool:
        return self.result


@pytest.mark.asyncio
async def test_auth_error_before_output_recovers_once(monkeypatch):
    auth = FakeAuth(True)
    client = APIClient(_config(), auth_manager=auth)  # type: ignore[arg-type]
    calls = 0

    async def fake_legacy_stream(self, _message):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AuthExpiredError("expired")
        yield ("text", "recovered")

    monkeypatch.setattr(LegacyAPIClient, "send_message_stream", fake_legacy_stream)
    tokens = [token async for token in client.send_message_stream("hi")]
    assert tokens == [("text", "recovered")]
    assert auth.recoveries == 1
    assert auth.invalidated == 1


@pytest.mark.asyncio
async def test_auth_error_after_partial_output_is_not_retried(monkeypatch):
    auth = FakeAuth(True)
    client = APIClient(_config(), auth_manager=auth)  # type: ignore[arg-type]

    async def fake_legacy_stream(self, _message):
        yield ("text", "partial")
        raise AuthExpiredError("expired")

    monkeypatch.setattr(LegacyAPIClient, "send_message_stream", fake_legacy_stream)
    received = []
    with pytest.raises(AuthExpiredError):
        async for token in client.send_message_stream("hi"):
            received.append(token)
    assert received == [("text", "partial")]
    assert auth.recoveries == 0


@pytest.mark.asyncio
async def test_pow_prefers_local_wasm(monkeypatch):
    client = APIClient(_config())
    challenge = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "abc",
        "salt": "salt",
        "difficulty": 1,
        "expire_at": 123,
        "signature": "sig",
    }

    async def fake_challenge(target_path=None):
        return challenge

    client._fetch_pow_challenge = fake_challenge  # type: ignore[assignment]
    client._solve_pow_wasm = lambda *args: 42  # type: ignore[assignment]

    async def browser_should_not_run(*args, **kwargs):
        raise AssertionError("browser fallback should not run")

    monkeypatch.setattr("deepseek.client.solve_pow_browser", browser_should_not_run)
    token = await client._generate_pow_token_for_path("/api/v0/chat/completion")
    assert token


@pytest.mark.asyncio
async def test_pow_uses_browser_before_node(monkeypatch):
    client = APIClient(_config())
    challenge = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "abc",
        "salt": "salt",
        "difficulty": 1,
        "expire_at": 123,
        "signature": "sig",
    }

    async def fake_challenge(target_path=None):
        return challenge

    client._fetch_pow_challenge = fake_challenge  # type: ignore[assignment]
    client._solve_pow_wasm = lambda *args: None  # type: ignore[assignment]
    monkeypatch.setattr("deepseek.client.solve_pow_browser", _async_value(43))
    monkeypatch.setattr(
        "deepseek.client.solve_pow_node",
        lambda _challenge: (_ for _ in ()).throw(AssertionError("Node should not run")),
    )
    assert await client._generate_pow_token_for_path("/path")


@pytest.mark.asyncio
async def test_pow_fails_closed_when_all_solvers_fail(monkeypatch):
    client = APIClient(_config())
    challenge = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "abc",
        "salt": "salt",
        "difficulty": 1,
        "expire_at": 123,
        "signature": "sig",
    }

    async def fake_challenge(target_path=None):
        return challenge

    client._fetch_pow_challenge = fake_challenge  # type: ignore[assignment]
    client._solve_pow_wasm = lambda *args: None  # type: ignore[assignment]
    monkeypatch.setattr("deepseek.client.solve_pow_browser", _async_value(None))
    monkeypatch.setattr("deepseek.client.solve_pow_node", lambda _challenge: None)
    with pytest.raises(RuntimeError, match="No compatible DeepSeek PoW solver"):
        await client._generate_pow_token_for_path("/path")


@pytest.mark.asyncio
async def test_missing_pow_challenge_with_stale_auth_raises_auth_error(monkeypatch):
    client = APIClient(_config())

    async def no_challenge(target_path=None):
        return None

    client._fetch_pow_challenge = no_challenge  # type: ignore[assignment]
    stale = ProtocolCapabilities(
        reachable=True,
        auth_ok=False,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/api/v0/chat/completion",
        session_path="/api/v0/chat_session/create",
        pow_challenge_path="/api/v0/chat/create_pow_challenge",
        status_code=403,
        error="forbidden",
    )
    monkeypatch.setattr("deepseek.protocol.probe_protocol", _async_value(stale))

    with pytest.raises(AuthExpiredError):
        await client._generate_pow_token_for_path("/api/v0/chat/completion")


def _async_value(value):
    async def inner(*args, **kwargs):
        return value

    return inner
