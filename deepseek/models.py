"""
Data models shared across the package.
"""
import base64
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


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

    Designed for forward-compatibility: unknown keys from future config
    versions are silently ignored via ``from_dict``.
    """
    target_url: str
    api_endpoint: str
    method: str
    headers: dict
    body_template: dict
    auth_type: str = "bearer"       # bearer | cookie | header
    auth_token: str = ""
    cookies: dict = field(default_factory=dict)
    refresh_endpoint: str = ""
    use_stream: bool = True
    pow_response: str = ""          # x-ds-pow-response token (captured from browser)
    # Discovered API paths — populated during --discover so we don't hardcode them
    completion_path: str = "/api/v0/chat/completion"
    session_path: str = "/api/v0/chat_session/create"
    pow_challenge_path: str = "/api/v0/chat/create_pow_challenge"
    # Schema version for future migrations
    config_version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig":
        """Load config, ignoring unknown keys so future fields don't crash old code."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        # Fill defaults for any missing optional fields (backward compat)
        filtered.setdefault("pow_response", "")
        filtered.setdefault("refresh_endpoint", "")
        filtered.setdefault("cookies", {})
        filtered.setdefault("completion_path", "/api/v0/chat/completion")
        filtered.setdefault("session_path", "/api/v0/chat_session/create")
        filtered.setdefault("pow_challenge_path", "/api/v0/chat/create_pow_challenge")
        filtered.setdefault("config_version", 1)
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    def token_expires_in(self) -> Optional[float]:
        """Return seconds until JWT auth_token expires, or None if unknown/not a JWT."""
        return _jwt_seconds_until_exp(self.auth_token)

    def cookie_expires_soon(self, threshold_hours: float = 24.0) -> list[str]:
        """Return names of cookies expiring within ``threshold_hours``.

        Only JWT-shaped cookies are inspected; opaque cookies are skipped.
        """
        threshold_s = threshold_hours * 3600
        expiring: list[str] = []
        for name, value in self.cookies.items():
            remaining = _jwt_seconds_until_exp(value)
            if remaining is not None and remaining < threshold_s:
                expiring.append(name)
        return expiring


def _jwt_seconds_until_exp(token: str) -> Optional[float]:
    """Decode a JWT-shaped string and return seconds until ``exp`` claim.

    Returns ``None`` for empty input, malformed JWTs, missing ``exp`` claim,
    or any decoding error. Does NOT verify the signature — DeepSeek tokens
    are opaque to us, we only need the expiry timestamp.
    """
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        # JWT base64-url-encodes the payload without padding
        payload_raw = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_raw))
        exp = payload.get("exp")
        if exp is None:
            return None
        return float(exp) - time.time()
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
