"""
Custom exceptions for the DeepSeek CLI.

Public exceptions (AuthExpiredError) are raised to callers.
Private exceptions (_PowExpiredError, _SessionNotFoundError) are internal
signals used within APIClient's retry loops and should not escape the client.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional


class AuthExpiredError(Exception):
    """Auth token or session has expired — user must run /reauth."""


class RateLimitError(RuntimeError):
    """Upstream rate limit with an optional server-provided retry delay."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after_seconds(
    value: Optional[str],
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """Parse Retry-After as delta-seconds or an HTTP date."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return seconds

    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - current).total_seconds())


class _PowExpiredError(Exception):
    """Internal: PoW token rejected by server — caller should re-solve and retry."""


class _SessionNotFoundError(Exception):
    """Internal: session_id is no longer valid — caller should recreate session."""
