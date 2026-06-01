"""Single-account state and helpers.

Each ``Account`` owns its own ``APIClient`` and an ``asyncio.Lock`` ensuring
at most one in-flight request hits the underlying DeepSeek session — the
session is stateful (parent_message_id) so concurrent writes would corrupt
the conversation.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..client import APIClient
from ..constants import DEFAULT_SYSTEM_PROMPT
from ..models import APIConfig

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
COOLDOWN_BASE_SECONDS = 60.0      # first quota hit  →  1 min
COOLDOWN_MAX_SECONDS = 3600.0     # capped at         1 h


@dataclass
class Account:
    """One DeepSeek account with its own APIClient, lock, and quota state."""

    name: str
    config: APIConfig
    client: APIClient = field(init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    exhausted_until: float = 0.0
    consecutive_quota_hits: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_used: float = 0.0
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        self.client = APIClient(self.config)
        self.client.system_prompt = DEFAULT_SYSTEM_PROMPT

    # ── Availability ──────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return time.time() >= self.exhausted_until

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.exhausted_until - time.time())

    # ── State transitions ─────────────────────────────────────

    def mark_exhausted(self, cooldown_seconds: Optional[float] = None) -> float:
        """Mark this account quota-exhausted with exponential backoff.

        Returns the cooldown duration applied (seconds).
        """
        if cooldown_seconds is None:
            # Exponential: 60, 120, 240, 480, ... capped
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
        # Successful request → reset consecutive quota counter.
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
        for pat in _QUOTA_CODE_PATTERNS:
            if pat.search(msg):
                return True
        return False

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
        }

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass
