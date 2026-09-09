"""Regression tests for persistent gateway conversation affinity."""
from __future__ import annotations

import time

import pytest

from deepseek.gateway.account import Account
from deepseek.gateway.bindings import ConversationBinding, ConversationBindingStore
from deepseek.gateway.pool import AccountPool, NoAccountAvailableError
from deepseek.models import APIConfig


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


def _account(name: str) -> Account:
    return Account(name=name, config=_config())


class TestConversationBindingStore:
    def test_roundtrip_persists_lineage(self, tmp_path):
        path = tmp_path / "bindings.json"
        store = ConversationBindingStore(path=path)
        binding = store.bind("conv-1", "a0")
        binding.session_id = "session-1"
        binding.parent_message_id = "msg-9"
        binding.model_type = "expert"
        binding.thinking_enabled = True
        binding.touch()
        store.flush()

        loaded = ConversationBindingStore(path=path).get("conv-1")
        assert loaded is not None
        assert loaded.account_name == "a0"
        assert loaded.session_id == "session-1"
        assert loaded.parent_message_id == "msg-9"
        assert loaded.thinking_enabled is True

    def test_expired_binding_is_dropped(self, tmp_path):
        path = tmp_path / "bindings.json"
        store = ConversationBindingStore(path=path, ttl_seconds=1)
        binding = store.bind("old", "a0")
        binding.updated_at = time.time() - 5
        store.flush()

        loaded = ConversationBindingStore(path=path, ttl_seconds=1)
        assert loaded.get("old") is None


class TestAccountConversationClients:
    def test_conversations_get_isolated_clients(self):
        account = _account("a0")
        first = account.client_for("conv-a")
        second = account.client_for("conv-b")
        assert first is not second
        assert first is not account.client
        assert second is not account.client

    def test_persisted_lineage_hydrates_client(self):
        account = _account("a0")
        binding = ConversationBinding(
            conversation_id="conv-a",
            account_name="a0",
            session_id="session-x",
            parent_message_id="message-y",
            model_type="default",
            thinking_enabled=False,
            search_enabled=True,
            updated_at=time.time(),
        )
        client = account.client_for("conv-a", binding)
        assert client.session_id == "session-x"
        assert client.last_message_id == "message-y"
        assert client._is_first_message is False
        assert client._session_flags == ("default", False, True)


@pytest.mark.asyncio
async def test_gateway_pins_conversation_to_first_successful_account(tmp_path, monkeypatch):
    store = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=store)
    a0 = _account("a0")
    a1 = _account("a1")
    a0.last_used = 0.0
    a1.last_used = 1.0
    pool._replace_accounts_for_test([a0, a1])

    async def fake_stream(self, _message):
        self.session_id = self.session_id or f"session-{id(self)}"
        self.last_message_id = "msg-1"
        yield ("text", "ok")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", fake_stream)

    chosen: list[str] = []
    async for _ in pool.send_message_stream(
        "first",
        conversation_id="conv-1",
        on_client_chosen=lambda account, _client: chosen.append(account.name),
    ):
        pass
    assert chosen == ["a0"]

    # Make the other account the scheduler's preferred choice. Affinity must still win.
    a0.last_used = 999.0
    a1.last_used = 0.0
    chosen.clear()
    async for _ in pool.send_message_stream(
        "second",
        conversation_id="conv-1",
        on_client_chosen=lambda account, _client: chosen.append(account.name),
    ):
        pass
    assert chosen == ["a0"]
    assert store.get("conv-1").account_name == "a0"


@pytest.mark.asyncio
async def test_bound_conversation_never_silently_migrates_on_quota(tmp_path, monkeypatch):
    store = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=store)
    a0 = _account("a0")
    a1 = _account("a1")
    a0.last_used = 0.0
    a1.last_used = 1.0
    pool._replace_accounts_for_test([a0, a1])

    async def ok_stream(self, _message):
        self.session_id = "session-a"
        self.last_message_id = "msg-a"
        yield ("text", "ok")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", ok_stream)
    async for _ in pool.send_message_stream("first", conversation_id="conv-1"):
        pass

    a0.exhausted_until = time.time() + 60
    with pytest.raises(NoAccountAvailableError) as exc:
        async for _ in pool.send_message_stream("second", conversation_id="conv-1"):
            pass
    assert "pinned" in str(exc.value).lower()
    assert store.get("conv-1").account_name == "a0"


@pytest.mark.asyncio
async def test_delete_conversation_drops_binding_and_client(tmp_path):
    store = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=store)
    account = _account("a0")
    pool._replace_accounts_for_test([account])
    store.bind("conv-1", "a0")
    account.client_for("conv-1", store.get("conv-1"))

    assert await pool.delete_conversation("conv-1") is True
    assert store.get("conv-1") is None
    assert "conv-1" not in account.conversation_clients
