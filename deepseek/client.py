"""DeepSeek client compatibility shell over the proven v2 protocol core.

The legacy core keeps the mature session, semantic SSE, upload, retry and
compaction behavior. This module adds three narrow boundaries around it:
self-healing auth, hybrid PoW, and spec-aware SSE transport framing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator, Optional

from ._client_legacy import (
    APIClient as _LegacyAPIClient,
    Token,
    _backoff_seconds,
    _classify_path_for_thinking,
    _strip_think_tags,
)
from .auth import AuthManager, PersistCallback
from .exceptions import AuthExpiredError
from .metrics import metrics
from .pow import solve_pow_browser, solve_pow_node
from .sse import iter_sse_events


class _FramedResponse:
    """Expose robust SSE framing through the line interface used by v2 core."""

    def __init__(self, response) -> None:
        self._response = response

    def __getattr__(self, name):
        return getattr(self._response, name)

    async def aiter_lines(self):
        async for event in iter_sse_events(self._response):
            if event.event:
                yield f"event: {event.event}"
            if event.data == "[DONE]":
                yield "data: [DONE]"
            elif isinstance(event.data, (dict, list, int, float, bool)) or event.data is None:
                yield "data: " + json.dumps(event.data, ensure_ascii=False)
            else:
                raw = str(event.data)
                if raw.startswith("data:") or raw.startswith("event:"):
                    yield raw
                else:
                    yield "data: " + raw
            yield ""


class _StreamContext:
    def __init__(self, context) -> None:
        self._context = context

    async def __aenter__(self):
        response = await self._context.__aenter__()
        return _FramedResponse(response)

    async def __aexit__(self, exc_type, exc, tb):
        return await self._context.__aexit__(exc_type, exc, tb)


class _FramedAsyncClient:
    """Delegate httpx operations while normalizing only streaming responses."""

    def __init__(self, client) -> None:
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    def stream(self, *args, **kwargs):
        return _StreamContext(self._client.stream(*args, **kwargs))

    async def post(self, *args, **kwargs):
        return await self._client.post(*args, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()


class APIClient(_LegacyAPIClient):
    """Production client with compatibility-preserving resilience upgrades."""

    def __init__(
        self,
        config,
        *,
        auth_manager: Optional[AuthManager] = None,
        auth_persist: Optional[PersistCallback] = None,
        auth_profile_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(config)
        raw_client = self.client
        self.client = _FramedAsyncClient(raw_client)
        self.auth_manager = auth_manager or AuthManager(
            config,
            persist=auth_persist,
            profile_dir=auth_profile_dir,
        )

    async def ensure_authenticated(self, *, force: bool = False) -> bool:
        """Public readiness hook for server startup, status checks and repair flows."""
        if not self._is_deepseek_mode():
            return True
        return await self.auth_manager.ensure_valid(force=force)

    async def send_message_stream(self, message: str) -> AsyncGenerator[Token, None]:
        """Retry once after auth recovery only when no output reached the caller."""
        yielded = False
        try:
            async for token in super().send_message_stream(message):
                yielded = True
                yield token
            return
        except AuthExpiredError:
            self.auth_manager.invalidate()
            if yielded or not self._is_deepseek_mode():
                raise

        if not await self.auth_manager.recover():
            raise AuthExpiredError("DeepSeek authentication expired and browser recovery failed")
        await self.reset_session()
        async for token in super().send_message_stream(message):
            yield token

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        upload_path: str = "/api/v0/file/upload_file",
    ) -> str:
        """Recover stale auth once before retrying a side-effect-safe failed upload."""
        try:
            return await super().upload_file(
                content,
                filename,
                content_type,
                upload_path,
            )
        except AuthExpiredError:
            self.auth_manager.invalidate()
        if not self._is_deepseek_mode() or not await self.auth_manager.recover():
            raise AuthExpiredError("DeepSeek authentication expired and browser recovery failed")
        return await super().upload_file(content, filename, content_type, upload_path)

    async def _generate_pow_token_for_path(self, target_path: str) -> str:
        """Solve PoW locally first, then use current web worker, then Node.

        DeepSeek mode is fail-closed: an unavailable challenge or exhausted
        solver chain is reported instead of sending a predictably invalid request.
        """
        challenge = await self._fetch_pow_challenge(target_path=target_path)
        if not challenge:
            metrics.incr("pow_failed_total")
            raise RuntimeError("DeepSeek PoW challenge is unavailable")

        algorithm = str(challenge.get("algorithm") or "")
        challenge_text = str(challenge.get("challenge") or "")
        salt = str(challenge.get("salt") or "")
        expire_at = challenge.get("expire_at")
        try:
            difficulty = int(challenge.get("difficulty", 144000))
            expire_value = int(expire_at)
        except (TypeError, ValueError):
            metrics.incr("pow_failed_total")
            raise RuntimeError("DeepSeek PoW challenge has invalid numeric fields")
        if not challenge_text or not expire_value:
            metrics.incr("pow_failed_total")
            raise RuntimeError("DeepSeek PoW challenge is incomplete")

        answer: Optional[int] = None
        if algorithm in ("", "DeepSeekHashV1"):
            answer = self._solve_pow_wasm(
                algorithm or "DeepSeekHashV1",
                challenge_text,
                salt,
                difficulty,
                expire_value,
            )
            if answer is not None:
                metrics.incr("pow_local_success_total")

        if answer is None and self.config.pow_worker_url:
            metrics.incr("pow_browser_fallback_total")
            answer = await solve_pow_browser(
                challenge,
                target_path=target_path,
                target_url=self.base_url,
                worker_url=self.config.pow_worker_url,
            )
            if answer is not None:
                metrics.incr("pow_browser_success_total")

        if answer is None and algorithm in ("", "DeepSeekHashV1"):
            answer = solve_pow_node(challenge)
            if answer is not None:
                metrics.incr("pow_node_success_total")

        if answer is None:
            metrics.incr("pow_failed_total")
            name = algorithm or "unknown"
            raise RuntimeError(f"No compatible DeepSeek PoW solver for algorithm '{name}'")
        return self._build_pow_token(challenge, answer, target_path=target_path)


__all__ = [
    "APIClient",
    "Token",
    "_backoff_seconds",
    "_classify_path_for_thinking",
    "_strip_think_tags",
]
