"""Tests for deepseek.models — APIConfig and CapturedRequest."""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.models import APIConfig, CapturedRequest


# ── CapturedRequest ───────────────────────────────────────────

def test_captured_request_defaults():
    req = CapturedRequest(
        url="https://example.com",
        method="POST",
        request_headers={},
        request_body="",
        response_status=200,
        response_headers={},
        response_body="",
    )
    assert req.url == "https://example.com"
    assert req.method == "POST"
    assert req.timestamp > 0


# ── APIConfig.from_dict ───────────────────────────────────────

def _minimal_config_dict(**overrides) -> dict:
    base = {
        "target_url": "https://chat.deepseek.com",
        "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
        "method": "POST",
        "headers": {},
        "body_template": {},
    }
    base.update(overrides)
    return base


def test_from_dict_minimal():
    cfg = APIConfig.from_dict(_minimal_config_dict())
    assert cfg.target_url == "https://chat.deepseek.com"
    assert cfg.auth_token == ""
    assert cfg.pow_response == ""
    assert cfg.config_version == 1


def test_from_dict_ignores_unknown_keys():
    d = _minimal_config_dict(future_field="ignored", another_new_key=42)
    cfg = APIConfig.from_dict(d)
    assert not hasattr(cfg, "future_field")
    assert not hasattr(cfg, "another_new_key")


def test_from_dict_backward_compat_missing_fields():
    """Old configs without new fields should load with defaults."""
    d = _minimal_config_dict()
    # Simulate old config without new path fields
    cfg = APIConfig.from_dict(d)
    assert cfg.completion_path == "/api/v0/chat/completion"
    assert cfg.session_path == "/api/v0/chat_session/create"
    assert cfg.pow_challenge_path == "/api/v0/chat/create_pow_challenge"


def test_from_dict_preserves_custom_paths():
    d = _minimal_config_dict(
        completion_path="/api/v1/chat/completion",
        session_path="/api/v1/session/create",
        pow_challenge_path="/api/v1/pow/challenge",
    )
    cfg = APIConfig.from_dict(d)
    assert cfg.completion_path == "/api/v1/chat/completion"
    assert cfg.session_path == "/api/v1/session/create"


def test_to_dict_roundtrip():
    cfg = APIConfig.from_dict(_minimal_config_dict(auth_token="tok123"))
    d = cfg.to_dict()
    cfg2 = APIConfig.from_dict(d)
    assert cfg2.auth_token == "tok123"
    assert cfg2.completion_path == cfg.completion_path


# ── token_expires_in ─────────────────────────────────────────

def _make_jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.fakesig"


def test_token_expires_in_future():
    future_exp = int(time.time()) + 3600
    cfg = APIConfig.from_dict(_minimal_config_dict(auth_token=_make_jwt(future_exp)))
    remaining = cfg.token_expires_in()
    assert remaining is not None
    assert 3500 < remaining < 3700


def test_token_expires_in_past():
    past_exp = int(time.time()) - 100
    cfg = APIConfig.from_dict(_minimal_config_dict(auth_token=_make_jwt(past_exp)))
    remaining = cfg.token_expires_in()
    assert remaining is not None
    assert remaining < 0


def test_token_expires_in_no_token():
    cfg = APIConfig.from_dict(_minimal_config_dict())
    assert cfg.token_expires_in() is None


def test_token_expires_in_non_jwt():
    cfg = APIConfig.from_dict(_minimal_config_dict(auth_token="not-a-jwt"))
    assert cfg.token_expires_in() is None


# ── cookie_expires_soon ───────────────────────────────────────

def test_cookie_expires_soon_detects_expiring():
    soon_exp = int(time.time()) + 3600  # 1 hour from now
    jwt = _make_jwt(soon_exp)
    cfg = APIConfig.from_dict(_minimal_config_dict(cookies={"session": jwt}))
    expiring = cfg.cookie_expires_soon(threshold_hours=24)
    assert "session" in expiring


def test_cookie_expires_soon_ignores_fresh():
    far_exp = int(time.time()) + 7 * 24 * 3600  # 7 days
    jwt = _make_jwt(far_exp)
    cfg = APIConfig.from_dict(_minimal_config_dict(cookies={"session": jwt}))
    expiring = cfg.cookie_expires_soon(threshold_hours=24)
    assert "session" not in expiring


def test_cookie_expires_soon_ignores_non_jwt():
    cfg = APIConfig.from_dict(_minimal_config_dict(cookies={"plain": "value"}))
    expiring = cfg.cookie_expires_soon()
    assert expiring == []
