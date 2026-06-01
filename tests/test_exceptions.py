"""Tests for deepseek.exceptions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.exceptions import AuthExpiredError, _PowExpiredError, _SessionNotFoundError


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
