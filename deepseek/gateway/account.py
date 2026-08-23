"""Single-account state and helpers.

Each account owns a default client plus isolated per-conversation clients. The
coarse account lock intentionally preserves the existing one-request-per-account
behavior while preventing DeepSeek session lineage from leaking across logical
conversations.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from ..client import APIClient
from ..constants import DEFAULT_SYSTEM_PROMPT
from ..models import APIConfig

if TYPE_CHECKING:
    from .bindings import ConversationBinding

# Error codes that indicate quota/rate-limit exhaustion.
QUOTA_ERROR_CODES = {40400, 40401, 40402, 429}
QUOTA_ERROR_MESSAGES = {
    "quota exceeded",
    "rate limit",
    "too many requests",
    "daily limit",
    "usage limit",
}
_QUOTA_CODE_PATTERNS = [re.compile(rf"\b{c}\b") for c in QUOTA_ERROR_CODES]

# Exponential cooldown for repeat quota hits.
COOLDOWN_BASE_SECONDS = 60.0
COOLDOWN_MAX_SECONDS = 3600.0


@dataclass
class Account:
    """One DeepSeek account with isolated conversation clients and quota state."""

    name: str
    config: APIConfig
    client: APIClient = field(init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _conversation_clients: dict[str, APIClient] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    exhausted_until: float = 0.0
    consecutive_quota_hits: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_used: float = 0.0
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        self.client = self._new_client()

    def _new_client(self) -> APIClient:
        client = APIClient(self.config)
        client.system_prompt = DEFAULT_SYSTEM_PROMPT
        return client

    def client_for(
        self,
        conversation_id: Optional[str],
        binding: Optional["ConversationBinding"] = None,
    ) -> APIClient:
        """Return isolated state for a conversation, hydrating persisted lineage."""
        if not conversation_id:
            return self.client
        existing = self._conversation_clients.get(conversation_id)
        if existing is not None:
            return existing

        client = self._new_client()
        if binding is not None and binding.session_id:
            client.session_id = binding.session_id
            client.last_message_id = binding.parent_message_id
            client.model_type = binding.model_type
            client.thinking_enabled = binding.thinking_enabled
            client.search_enabled = binding.search_enabled
            client._is_first_message = False
            client._session_flags = (
                client.model_type,
                client.thinking_enabled,
                client.search_enabled,
            )
        self._conversation_clients[conversation_id] = client
        return client

    async def drop_conversation(self, conversation_id: str) -> bool:
        client = self._conversation_clients.pop(conversation_id, None)
        if client is None:
            return False
        with contextlib.suppress(Exception):
            await client.close()
        return True

    @property
    def conversation_clients(self) -> dict[str, APIClient]:
        return dict(self._conversation_clients)

    # ── Availability ──────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return time.time() >= self.exhausted_until

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.exhausted_until - time.time())

    # ── State transitions ─────────────────────────────────────

    def mark_exhausted(self, cooldown_seconds: Optional[float] = None) -> float:
        """Mark this account quota-exhausted with exponential backoff."""
        if cooldown_seconds is None:
            cooldown_seconds = min(
                COOLDOWN_BASE_SECONDS * (2 ** self.consecutive_quota_hits),
                COOLDOWN_MAX_SECONDS,
            )
        self.exhausted_until = time.time() + cooldown_seconds
        self.consecutive_quota_hits += 1
        self.total_errors += 1
        return cooldown_seconds

    def mark_success(self) -> None:
        self.total_requests += 1
        self.last_used = time.time()
        self.consecutive_quota_hits = 0
        self.exhausted_until = 0.0

    def mark_error(self, error: Exception) -> None:
        self.total_errors += 1
        self.last_error = str(error)[:200]

    # ── Classification ────────────────────────────────────────

    def is_quota_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        if any(phrase in msg for phrase in QUOTA_ERROR_MESSAGES):
            return True
        return any(pattern.search(msg) for pattern in _QUOTA_CODE_PATTERNS)

    # ── Reporting ─────────────────────────────────────────────

    def to_status_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.is_available,
            "cooldown_remaining_seconds": int(self.cooldown_remaining),
            "exhausted_until": self.exhausted_until if not self.is_available else None,
            "consecutive_quota_hits": self.consecutive_quota_hits,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "last_used": self.last_used or None,
            "last_error": self.last_error,
            "in_use": self.lock.locked(),
            "active_conversations": len(self._conversation_clients),
        }

    async def close(self) -> None:
        clients = [self.client, *self._conversation_clients.values()]
        self._conversation_clients.clear()
        seen: set[int] = set()
        for client in clients:
            identity = id(client)
            if identity in seen:
                continue
            seen.add(identity)
            with contextlib.suppress(Exception):
                await client.close()
