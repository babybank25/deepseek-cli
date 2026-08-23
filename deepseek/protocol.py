"""Side-effect-free DeepSeek Web protocol capability probes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from .models import APIConfig


@dataclass(frozen=True)
class ProtocolCapabilities:
    reachable: bool
    auth_ok: bool
    pow_ok: bool
    pow_algorithm: Optional[str]
    completion_path: str
    session_path: str
    pow_challenge_path: str
    status_code: Optional[int] = None
    error: Optional[str] = None


def _probe_headers(config: APIConfig) -> dict[str, str]:
    headers = {
        key: value
        for key, value in config.headers.items()
        if key.lower()
        not in {"accept", "content-type", "authorization", "cookie", "x-ds-pow-response"}
    }
    headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": config.target_url.rstrip("/"),
            "Referer": config.target_url.rstrip("/") + "/",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
        }
    )
    if config.auth_token:
        headers["Authorization"] = f"Bearer {config.auth_token}"
    if config.cookies:
        headers["Cookie"] = "; ".join(
            f"{name}={value}" for name, value in config.cookies.items()
        )
    return headers


async def probe_protocol(
    config: APIConfig,
    *,
    timeout: float = 15.0,
) -> ProtocolCapabilities:
    """Probe the PoW challenge endpoint without creating a chat session."""
    url = f"{config.target_url.rstrip('/')}{config.pow_challenge_path}"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            response = await client.post(
                url,
                headers=_probe_headers(config),
                json={"target_path": config.completion_path},
            )
    except httpx.HTTPError as error:
        return ProtocolCapabilities(
            reachable=False,
            auth_ok=False,
            pow_ok=False,
            pow_algorithm=None,
            completion_path=config.completion_path,
            session_path=config.session_path,
            pow_challenge_path=config.pow_challenge_path,
            error=str(error)[:300],
        )

    auth_ok = response.status_code not in (401, 403)
    pow_ok = False
    algorithm: Optional[str] = None
    error: Optional[str] = None
    try:
        body = response.json()
    except ValueError:
        body = None
        error = f"non-JSON response: {response.text[:200]}"

    if isinstance(body, dict):
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        biz_code = data.get("biz_code") if isinstance(data, dict) else None
        biz_data = data.get("biz_data") if isinstance(data, dict) else None
        challenge = (
            biz_data.get("challenge")
            if isinstance(biz_data, dict)
            and isinstance(biz_data.get("challenge"), dict)
            else None
        )
        top_code = body.get("code")
        auth_ok = auth_ok and top_code in (0, None) and biz_code in (0, None)
        if challenge is not None:
            algorithm_value = challenge.get("algorithm")
            algorithm = str(algorithm_value) if algorithm_value else None
            pow_ok = bool(challenge.get("challenge") and challenge.get("expire_at"))
        if not auth_ok and error is None:
            error = str(body.get("msg") or "authentication rejected")[:300]
    elif response.status_code >= 400 and error is None:
        error = f"HTTP {response.status_code}"

    return ProtocolCapabilities(
        reachable=True,
        auth_ok=auth_ok,
        pow_ok=pow_ok,
        pow_algorithm=algorithm,
        completion_path=config.completion_path,
        session_path=config.session_path,
        pow_challenge_path=config.pow_challenge_path,
        status_code=response.status_code,
        error=error,
    )
