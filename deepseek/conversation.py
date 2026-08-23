"""OpenAI request/session resolution and persisted DeepSeek lineage."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .constants import CONFIG_DIR
from .metrics import metrics

SESSION_INDEX_FILE = CONFIG_DIR / "openai_sessions.json"
SESSION_TTL_SECONDS = 30 * 60
MAX_PERSISTED_MESSAGES = 100


class UnknownPreviousResponseError(ValueError):
    pass


@dataclass
class ConversationState:
    conversation_id: str
    session_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    model_type: str = "expert"
    thinking_enabled: bool = False
    search_enabled: bool = False
    history: list[dict] = field(default_factory=list)
    updated_at: float = 0.0


@dataclass(frozen=True)
class ConversationResolution:
    conversation_id: str
    source: str
    is_new: bool


def _stable_message(message: object) -> object:
    if not isinstance(message, dict):
        return message
    keep = {}
    for key in ("role", "content", "tool_call_id", "tool_calls", "name"):
        if key in message:
            keep[key] = message[key]
    return keep


def history_fingerprint(messages: list[dict]) -> str:
    payload = json.dumps(
        [_stable_message(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def completed_history_before_latest_user(messages: list[dict]) -> list[dict]:
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        return messages[:-1]
    return messages


class ConversationIndex:
    """Map OpenAI response/history identifiers to persisted DeepSeek lineage."""

    def __init__(
        self,
        path: Path = SESSION_INDEX_FILE,
        ttl_seconds: float = SESSION_TTL_SECONDS,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._states: dict[str, ConversationState] = {}
        self._responses: dict[str, str] = {}
        self._histories: dict[str, str] = {}
        self._load()

    def resolve(
        self,
        *,
        messages: list[dict],
        conversation_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> ConversationResolution:
        self._prune()
        explicit = (conversation_id or chat_session_id or "").strip()
        if explicit:
            return ConversationResolution(
                conversation_id=explicit,
                source="explicit",
                is_new=explicit not in self._states,
            )

        previous = (previous_response_id or "").strip()
        if previous:
            resolved = self._responses.get(previous)
            if not resolved:
                raise UnknownPreviousResponseError(
                    f"Unknown previous_response_id: {previous}"
                )
            metrics.incr("session_resume_previous_response")
            return ConversationResolution(resolved, "previous_response", False)

        history = completed_history_before_latest_user(messages)
        if history:
            resolved = self._histories.get(history_fingerprint(history))
            if resolved and resolved in self._states:
                metrics.incr("session_resume_history")
                return ConversationResolution(resolved, "history", False)

        return ConversationResolution(
            conversation_id=f"conv_{uuid.uuid4().hex}",
            source="new",
            is_new=True,
        )

    def state(self, conversation_id: str) -> Optional[ConversationState]:
        self._prune()
        return self._states.get(conversation_id)

    def prior_history(self, conversation_id: str) -> list[dict]:
        state = self.state(conversation_id)
        return [dict(message) for message in state.history] if state else []

    def remember(
        self,
        *,
        conversation_id: str,
        response_id: str,
        request_messages: list[dict],
        assistant_message: dict,
        session_id: Optional[str],
        parent_message_id: Optional[str],
        model_type: str,
        thinking_enabled: bool,
        search_enabled: bool,
    ) -> None:
        completed = [*request_messages, assistant_message][-MAX_PERSISTED_MESSAGES:]
        self._states[conversation_id] = ConversationState(
            conversation_id=conversation_id,
            session_id=session_id,
            parent_message_id=parent_message_id,
            model_type=model_type,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            history=[dict(message) for message in completed],
            updated_at=time.time(),
        )
        self._responses[response_id] = conversation_id
        self._histories[history_fingerprint(completed)] = conversation_id
        self._save()

    def delete(self, conversation_id: str) -> bool:
        existed = self._states.pop(conversation_id, None) is not None
        self._responses = {
            key: value
            for key, value in self._responses.items()
            if value != conversation_id
        }
        self._histories = {
            key: value
            for key, value in self._histories.items()
            if value != conversation_id
        }
        if existed:
            self._save()
        return existed

    def list_states(self) -> list[dict]:
        self._prune()
        now = time.time()
        result = []
        for state in self._states.values():
            value = asdict(state)
            value.pop("history", None)
            value["idle_seconds"] = int(max(0.0, now - state.updated_at))
            result.append(value)
        result.sort(key=lambda item: item["updated_at"], reverse=True)
        return result

    def _prune(self) -> None:
        now = time.time()
        expired = {
            conversation_id
            for conversation_id, state in self._states.items()
            if state.updated_at and now - state.updated_at > self.ttl_seconds
        }
        if not expired:
            return
        for conversation_id in expired:
            self._states.pop(conversation_id, None)
        self._responses = {
            key: value for key, value in self._responses.items() if value not in expired
        }
        self._histories = {
            key: value for key, value in self._histories.items() if value not in expired
        }
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        states = raw.get("states") if isinstance(raw, dict) else None
        if isinstance(states, dict):
            for conversation_id, value in states.items():
                if not isinstance(value, dict):
                    continue
                try:
                    history = value.get("history")
                    self._states[conversation_id] = ConversationState(
                        conversation_id=conversation_id,
                        session_id=value.get("session_id"),
                        parent_message_id=value.get("parent_message_id"),
                        model_type=str(value.get("model_type") or "expert"),
                        thinking_enabled=bool(value.get("thinking_enabled", False)),
                        search_enabled=bool(value.get("search_enabled", False)),
                        history=[
                            dict(message)
                            for message in history
                            if isinstance(message, dict)
                        ] if isinstance(history, list) else [],
                        updated_at=float(value.get("updated_at") or 0.0),
                    )
                except (TypeError, ValueError):
                    continue
        responses = raw.get("responses") if isinstance(raw, dict) else None
        histories = raw.get("histories") if isinstance(raw, dict) else None
        if isinstance(responses, dict):
            self._responses = {
                str(key): str(value)
                for key, value in responses.items()
                if isinstance(value, str)
            }
        if isinstance(histories, dict):
            self._histories = {
                str(key): str(value)
                for key, value in histories.items()
                if isinstance(value, str)
            }
        self._prune()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "states": {
                            conversation_id: {
                                key: value
                                for key, value in asdict(state).items()
                                if key != "conversation_id"
                            }
                            for conversation_id, state in self._states.items()
                        },
                        "responses": self._responses,
                        "histories": self._histories,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError:
            pass
