"""Tests for OpenAI conversation/session resolution."""
import pytest

from deepseek.conversation import (
    ConversationIndex,
    UnknownPreviousResponseError,
)


def _messages():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]


def test_explicit_conversation_wins(tmp_path):
    index = ConversationIndex(path=tmp_path / "sessions.json")
    resolved = index.resolve(messages=_messages(), conversation_id="conv-explicit")
    assert resolved.conversation_id == "conv-explicit"
    assert resolved.source == "explicit"


def test_previous_response_resumes_persisted_conversation(tmp_path):
    path = tmp_path / "sessions.json"
    index = ConversationIndex(path=path)
    request = [{"role": "user", "content": "hello"}]
    index.remember(
        conversation_id="conv-1",
        response_id="resp-1",
        request_messages=request,
        assistant_message={"role": "assistant", "content": "hi"},
        session_id="sid",
        parent_message_id="mid",
        model_type="expert",
        thinking_enabled=True,
        search_enabled=False,
    )
    loaded = ConversationIndex(path=path)
    resolved = loaded.resolve(
        messages=[{"role": "user", "content": "next"}],
        previous_response_id="resp-1",
    )
    assert resolved.conversation_id == "conv-1"
    assert loaded.state("conv-1").session_id == "sid"


def test_exact_completed_history_resumes_without_custom_id(tmp_path):
    index = ConversationIndex(path=tmp_path / "sessions.json")
    index.remember(
        conversation_id="conv-1",
        response_id="resp-1",
        request_messages=[{"role": "user", "content": "hello"}],
        assistant_message={"role": "assistant", "content": "hi"},
        session_id="sid",
        parent_message_id="mid",
        model_type="expert",
        thinking_enabled=False,
        search_enabled=False,
    )
    resolved = index.resolve(messages=_messages())
    assert resolved.conversation_id == "conv-1"
    assert resolved.source == "history"


def test_unknown_previous_response_fails_closed(tmp_path):
    index = ConversationIndex(path=tmp_path / "sessions.json")
    with pytest.raises(UnknownPreviousResponseError):
        index.resolve(
            messages=[{"role": "user", "content": "next"}],
            previous_response_id="resp-missing",
        )
