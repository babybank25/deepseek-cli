"""Data models shared across the package."""
import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .constants import DEFAULT_POW_WORKER_URL


@dataclass
class CapturedRequest:
    """A single intercepted XHR/fetch request from the browser sniffer."""

    url: str
    method: str
    request_headers: dict
    request_body: str
    response_status: int
    response_headers: dict
    response_body: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class APIConfig:
    """Saved configuration for replaying API calls.

    Unknown keys are ignored on load so configs remain forward-compatible.
    """

    target_url: str
    api_endpoint: str
    method: str
    headers: dict
    body_template: dict
    auth_type: str = "bearer"
    auth_token: str = ""
    cookies: dict = field(default_factory=dict)
    refresh_endpoint: str = ""
    use_stream: bool = True
    pow_response: str = ""
    completion_path: str = "/api/v0/chat/completion"
    session_path: str = "/api/v0/chat_session/create"
    pow_challenge_path: str = "/api/v0/chat/create_pow_challenge"
    pow_worker_url: str = DEFAULT_POW_WORKER_URL
    config_version: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig":
        """Load config while preserving backward compatibility with v1 files."""
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {key: value for key, value in data.items() if key in known}
        filtered.setdefault("pow_response", "")
        filtered.setdefault("refresh_endpoint", "")
        filtered.setdefault("cookies", {})
        filtered.setdefault("completion_path", "/api/v0/chat/completion")
        filtered.setdefault("session_path", "/api/v0/chat_session/create")
        filtered.setdefault("pow_challenge_path", "/api/v0/chat/create_pow_challenge")
        filtered.setdefault("pow_worker_url", DEFAULT_POW_WORKER_URL)
        filtered.setdefault("config_version", 2)
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)

    def token_expires_in(self) -> Optional[float]:
        return _jwt_seconds_until_exp(self.auth_token)

    def cookie_expires_soon(self, threshold_hours: float = 24.0) -> list[str]:
        threshold_seconds = threshold_hours * 3600
        expiring: list[str] = []
        for name, value in self.cookies.items():
            remaining = _jwt_seconds_until_exp(value)
            if remaining is not None and remaining < threshold_seconds:
                expiring.append(name)
        return expiring


def _jwt_seconds_until_exp(token: str) -> Optional[float]:
    """Read an unverified JWT expiry timestamp when the token is JWT-shaped."""
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_raw = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_raw))
        expires = payload.get("exp")
        if expires is None:
            return None
        return float(expires) - time.time()
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
