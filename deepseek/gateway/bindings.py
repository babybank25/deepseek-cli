"""Persistent gateway conversation-to-account/session bindings."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..constants import CONFIG_DIR

if TYPE_CHECKING:
    from ..client import APIClient

BINDINGS_FILE = CONFIG_DIR / "conversation_bindings.json"
DEFAULT_BINDING_TTL_SECONDS = 30 * 60


@dataclass
class ConversationBinding:
    conversation_id: str
    account_name: str
    session_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    model_type: str = "expert"
    thinking_enabled: bool = False
    search_enabled: bool = False
    updated_at: float = 0.0

    def touch(self) -> None:
        self.updated_at = time.time()


class ConversationBindingStore:
    """Small atomic JSON store for gateway affinity and DeepSeek lineage."""

    def __init__(
        self,
        path: Path = BINDINGS_FILE,
        ttl_seconds: float = DEFAULT_BINDING_TTL_SECONDS,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._bindings: dict[str, ConversationBinding] = {}
        self.reload()

    def reload(self, valid_accounts: Optional[set[str]] = None) -> None:
        self._bindings = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        for conversation_id, value in raw.items():
            if not isinstance(conversation_id, str) or not isinstance(value, dict):
                continue
            try:
                binding = ConversationBinding(
                    conversation_id=conversation_id,
                    account_name=str(value["account_name"]),
                    session_id=(
                        str(value["session_id"])
                        if value.get("session_id") is not None
                        else None
                    ),
                    parent_message_id=(
                        str(value["parent_message_id"])
                        if value.get("parent_message_id") is not None
                        else None
                    ),
                    model_type=str(value.get("model_type") or "expert"),
                    thinking_enabled=bool(value.get("thinking_enabled", False)),
                    search_enabled=bool(value.get("search_enabled", False)),
                    updated_at=float(value.get("updated_at") or 0.0),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if valid_accounts is not None and binding.account_name not in valid_accounts:
                continue
            if binding.updated_at and now - binding.updated_at > self.ttl_seconds:
                continue
            self._bindings[conversation_id] = binding
        self._save()

    def get(self, conversation_id: Optional[str]) -> Optional[ConversationBinding]:
        if not conversation_id:
            return None
        binding = self._bindings.get(conversation_id)
        if binding is None:
            return None
        if binding.updated_at and time.time() - binding.updated_at > self.ttl_seconds:
            self.delete(conversation_id)
            return None
        return binding

    def bind(self, conversation_id: str, account_name: str) -> ConversationBinding:
        binding = self._bindings.get(conversation_id)
        if binding is None or binding.account_name != account_name:
            binding = ConversationBinding(
                conversation_id=conversation_id,
                account_name=account_name,
            )
            self._bindings[conversation_id] = binding
        binding.touch()
        self._save()
        return binding

    def update_from_client(
        self,
        conversation_id: str,
        account_name: str,
        client: "APIClient",
    ) -> ConversationBinding:
        binding = self.bind(conversation_id, account_name)
        binding.session_id = client.session_id
        binding.parent_message_id = client.last_message_id
        binding.model_type = client.model_type
        binding.thinking_enabled = client.thinking_enabled
        binding.search_enabled = client.search_enabled
        binding.touch()
        self._save()
        return binding

    def delete(self, conversation_id: str) -> Optional[ConversationBinding]:
        binding = self._bindings.pop(conversation_id, None)
        if binding is not None:
            self._save()
        return binding

    def delete_account(self, account_name: str) -> list[str]:
        removed = [
            conversation_id
            for conversation_id, binding in self._bindings.items()
            if binding.account_name == account_name
        ]
        for conversation_id in removed:
            self._bindings.pop(conversation_id, None)
        if removed:
            self._save()
        return removed

    def list(self) -> list[dict]:
        now = time.time()
        result = []
        for binding in list(self._bindings.values()):
            if binding.updated_at and now - binding.updated_at > self.ttl_seconds:
                self._bindings.pop(binding.conversation_id, None)
                continue
            item = asdict(binding)
            item["idle_seconds"] = int(max(0.0, now - binding.updated_at))
            result.append(item)
        result.sort(key=lambda item: item["updated_at"], reverse=True)
        return result

    def __len__(self) -> int:
        return len(self._bindings)

    def flush(self) -> None:
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            payload = {
                conversation_id: {
                    key: value
                    for key, value in asdict(binding).items()
                    if key != "conversation_id"
                }
                for conversation_id, binding in self._bindings.items()
            }
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError:
            pass
