"""Tests for deepseek.exceptions."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.exceptions import (
    AuthExpiredError,
    RateLimitError,
    _PowExpiredError,
    _SessionNotFoundError,
    parse_retry_after_seconds,
)


def test_auth_expired_error_is_exception():
    e = AuthExpiredError("Token expired")
    assert isinstance(e, Exception)
    assert str(e) == "Token expired"


def test_pow_expired_error_is_exception():
    e = _PowExpiredError("PoW rejected")
    assert isinstance(e, Exception)


def test_session_not_found_error_is_exception():
    e = _SessionNotFoundError("Session gone")
    assert isinstance(e, Exception)


def test_errors_are_distinct():
    assert not issubclass(AuthExpiredError, _PowExpiredError)
    assert not issubclass(_PowExpiredError, AuthExpiredError)
    assert not issubclass(_SessionNotFoundError, AuthExpiredError)


def test_rate_limit_error_carries_retry_after():
    error = RateLimitError("limited", retry_after=12.5)
    assert isinstance(error, RuntimeError)
    assert error.retry_after == 12.5


def test_parse_retry_after_seconds_delta_value():
    assert parse_retry_after_seconds("15") == 15.0


def test_parse_retry_after_seconds_http_date():
    now = datetime(2026, 9, 14, 3, 0, tzinfo=timezone.utc)
    assert parse_retry_after_seconds(
        "Mon, 14 Sep 2026 03:00:09 GMT",
        now=now,
    ) == 9.0


def test_parse_retry_after_seconds_rejects_invalid_value():
    assert parse_retry_after_seconds("not-a-date") is None
