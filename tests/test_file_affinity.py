"""Regression tests for file/account affinity in gateway mode."""
import pytest

from deepseek.gateway.account import Account
from deepseek.gateway.bindings import ConversationBindingStore
from deepseek.gateway.files import FileAffinityError, FileAffinityStore
from deepseek.gateway.pool import AccountPool, NoAccountAvailableError
from deepseek.models import APIConfig


def _config():
    return APIConfig.from_dict(
        {
            "target_url": "https://chat.deepseek.com",
            "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
            "method": "POST",
            "headers": {},
            "body_template": {},
        }
    )


def test_file_affinity_roundtrip_and_conflict(tmp_path):
    store = FileAffinityStore(path=tmp_path / "files.json")
    store.bind("file-a", "account-a")
    store.bind("file-b", "account-b")
    loaded = FileAffinityStore(path=tmp_path / "files.json")
    assert loaded.account_for(["file-a"]) == "account-a"
    with pytest.raises(FileAffinityError):
        loaded.account_for(["file-a", "file-b"])


def test_file_affinity_can_require_every_file_to_be_known(tmp_path):
    store = FileAffinityStore(path=tmp_path / "files.json")
    store.bind("known", "a0")

    assert store.account_for(["known", "unknown"]) == "a0"
    with pytest.raises(FileAffinityError, match="unknown file"):
        store.account_for(["known", "unknown"], require_all_known=True)


@pytest.mark.asyncio
async def test_file_affinity_forces_account_choice(tmp_path, monkeypatch):
    files = FileAffinityStore(path=tmp_path / "files.json")
    bindings = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=bindings, file_store=files)
    a0 = Account("a0", _config())
    a1 = Account("a1", _config())
    a0.last_used = 999.0
    a1.last_used = 0.0
    pool._replace_accounts_for_test([a0, a1])
    files.bind("file-a", "a0")

    async def stream(self, _message):
        yield ("text", "ok")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", stream)
    chosen = []
    async for _ in pool.send_message_stream(
        "hello",
        file_ids=["file-a"],
        on_account_chosen=lambda account: chosen.append(account.name),
    ):
        pass
    assert chosen == ["a0"]


@pytest.mark.asyncio
async def test_conversation_and_file_account_conflict_fails(tmp_path):
    files = FileAffinityStore(path=tmp_path / "files.json")
    bindings = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=bindings, file_store=files)
    pool._replace_accounts_for_test([Account("a0", _config()), Account("a1", _config())])
    bindings.bind("conv", "a0")
    files.bind("file-a", "a1")
    with pytest.raises(NoAccountAvailableError, match="attached file"):
        async for _ in pool.send_message_stream(
            "hello",
            conversation_id="conv",
            file_ids=["file-a"],
        ):
            pass


@pytest.mark.asyncio
async def test_unknown_file_id_fails_closed_with_multiple_accounts(tmp_path, monkeypatch):
    files = FileAffinityStore(path=tmp_path / "files.json")
    bindings = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=bindings, file_store=files)
    pool._replace_accounts_for_test([Account("a0", _config()), Account("a1", _config())])

    async def stream(self, _message):
        yield ("text", "should-not-route")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", stream)
    with pytest.raises(NoAccountAvailableError, match="unknown file"):
        async for _ in pool.send_message_stream("hello", file_ids=["legacy-file"]):
            pass


@pytest.mark.asyncio
async def test_mixed_known_and_unknown_file_fails_closed_with_multiple_accounts(tmp_path, monkeypatch):
    files = FileAffinityStore(path=tmp_path / "files.json")
    bindings = ConversationBindingStore(path=tmp_path / "bindings.json")
    pool = AccountPool(binding_store=bindings, file_store=files)
    pool._replace_accounts_for_test([Account("a0", _config()), Account("a1", _config())])
    files.bind("known", "a0")

    async def stream(self, _message):
        yield ("text", "should-not-route")

    monkeypatch.setattr("deepseek.client.APIClient.send_message_stream", stream)
    with pytest.raises(NoAccountAvailableError, match="unknown file"):
        async for _ in pool.send_message_stream(
            "hello",
            file_ids=["known", "unknown"],
        ):
            pass
