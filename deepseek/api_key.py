"""Secure local API-key provisioning for the HTTP compatibility server."""
from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import CONFIG_DIR

API_KEY_FILE = CONFIG_DIR / "api_key"
API_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass(frozen=True)
class APIKeyInfo:
    key: str
    source: str
    path: Optional[Path] = None


def _first_key(text: str) -> str:
    for line in text.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    return ""


def load_or_create_api_key(explicit: Optional[str] = None) -> APIKeyInfo:
    """Resolve CLI/env/file key or generate a persistent owner-local key."""
    if explicit and explicit.strip():
        return APIKeyInfo(explicit.strip(), "cli")

    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return APIKeyInfo(env_key, "env")

    if API_KEY_FILE.exists():
        try:
            key = _first_key(API_KEY_FILE.read_text(encoding="utf-8"))
        except OSError:
            key = ""
        if key:
            return APIKeyInfo(key, "file", API_KEY_FILE)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    key = f"sk-ds-{secrets.token_hex(24)}"
    tmp = API_KEY_FILE.with_suffix(".tmp")
    tmp.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(API_KEY_FILE)
    try:
        os.chmod(API_KEY_FILE, 0o600)
    except OSError:
        pass
    return APIKeyInfo(key, "generated", API_KEY_FILE)


def extract_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> str:
    """Accept OpenAI-style Bearer auth or the common x-api-key header."""
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if not authorization:
        return ""
    value = authorization.strip()
    prefix = "bearer "
    if value.lower().startswith(prefix):
        return value[len(prefix) :].strip()
    return value


def is_api_key_authorized(
    authorization: Optional[str],
    x_api_key: Optional[str],
    expected: str,
) -> bool:
    actual = extract_api_key(authorization, x_api_key)
    return bool(actual) and hmac.compare_digest(actual, expected)
