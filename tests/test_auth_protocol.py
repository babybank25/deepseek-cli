"""Tests for side-effect-free protocol probing and auth recovery policy."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepseek.auth import AuthManager
from deepseek.models import APIConfig
from deepseek.protocol import ProtocolCapabilities, probe_protocol


def _config() -> APIConfig:
    return APIConfig.from_dict(
        {
            "target_url": "https://chat.deepseek.com",
            "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
            "method": "POST",
            "headers": {},
            "body_template": {},
            "auth_token": "token",
            "cookies": {"ds_session_id": "cookie"},
        }
    )


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "code": 0,
            "data": {
                "biz_code": 0,
                "biz_data": {
                    "challenge": {
                        "algorithm": "DeepSeekHashV1",
                        "challenge": "abc",
                        "expire_at": 123,
                    }
                },
            },
        }


class _FakeHTTPClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_protocol_probe_reports_auth_and_pow(monkeypatch):
    monkeypatch.setattr("deepseek.protocol.httpx.AsyncClient", lambda **kwargs: _FakeHTTPClient())
    result = await probe_protocol(_config())
    assert result.reachable is True
    assert result.auth_ok is True
    assert result.pow_ok is True
    assert result.pow_algorithm == "DeepSeekHashV1"


@pytest.mark.asyncio
async def test_auth_manager_reuses_valid_saved_auth_without_browser(monkeypatch):
    manager = AuthManager(_config())
    good = ProtocolCapabilities(
        reachable=True,
        auth_ok=True,
        pow_ok=True,
        pow_algorithm="DeepSeekHashV1",
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=200,
    )
    monkeypatch.setattr("deepseek.auth.probe_protocol", AsyncMock(return_value=good))
    browser = AsyncMock(return_value=False)
    monkeypatch.setattr(manager, "_recover_with_browser", browser)
    assert await manager.ensure_valid() is True
    browser.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_manager_serializes_browser_recovery(monkeypatch):
    manager = AuthManager(_config())
    bad = ProtocolCapabilities(
        reachable=True,
        auth_ok=False,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=401,
    )
    monkeypatch.setattr("deepseek.auth.probe_protocol", AsyncMock(return_value=bad))
    recovery = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_recover_with_browser", recovery)
    assert await manager.ensure_valid(force=True) is True
    recovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_recoveries_share_one_browser_refresh(monkeypatch):
    manager = AuthManager(_config())
    bad = ProtocolCapabilities(
        reachable=True,
        auth_ok=False,
        pow_ok=False,
        pow_algorithm=None,
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=401,
    )
    monkeypatch.setattr("deepseek.auth.probe_protocol", AsyncMock(return_value=bad))

    started = asyncio.Event()
    release = asyncio.Event()
    recovery_calls = 0

    async def recover_once():
        nonlocal recovery_calls
        recovery_calls += 1
        started.set()
        await release.wait()
        manager._validated = True
        return True

    monkeypatch.setattr(manager, "_recover_with_browser", recover_once)

    first = asyncio.create_task(manager.recover())
    await started.wait()
    second = asyncio.create_task(manager.recover())
    release.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert recovery_calls == 1


@pytest.mark.asyncio
async def test_force_revalidates_when_auth_was_already_valid(monkeypatch):
    manager = AuthManager(_config())
    manager._validated = True
    good = ProtocolCapabilities(
        reachable=True,
        auth_ok=True,
        pow_ok=True,
        pow_algorithm="DeepSeekHashV1",
        completion_path="/c",
        session_path="/s",
        pow_challenge_path="/p",
        status_code=200,
    )
    probe = AsyncMock(return_value=good)
    monkeypatch.setattr("deepseek.auth.probe_protocol", probe)

    assert await manager.ensure_valid(force=True) is True
    probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_recovery_stops_sniffer_after_start_failure(monkeypatch):
    manager = AuthManager(_config())
    stopped = 0

    class FakeSniffer:
        context = None

        def __init__(self, *args, **kwargs):
            pass

        async def start(self, _url):
            raise RuntimeError("browser failed")

        async def stop(self):
            nonlocal stopped
            stopped += 1

    monkeypatch.setattr("deepseek.auth.NetworkSniffer", FakeSniffer)

    assert await manager._recover_with_browser() is False
    assert stopped == 1
