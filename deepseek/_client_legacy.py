"""
APIClient — DeepSeek web API client.

Handles the 2-step flow:
  1. POST {session_path}    → get session_id
  2. POST {completion_path} → SSE stream with AI response

Falls back to a generic OpenAI-style replay for non-DeepSeek endpoints.

Robustness features:
  - Retry + exponential backoff for network errors
  - Auto-recreate session on session-not-found errors (40200/40201)
  - Re-solve PoW and retry on PoW rejection (40300/40301)
  - Multi-layer PoW solver: WASM → Node.js → proceed without token
  - Flexible SSE parser handles DeepSeek format changes gracefully
  - Per-request timeout to prevent API server lock starvation

Diagnostics:
  Set ``DEEPSEEK_DUMP_SSE=1`` in the environment to write every raw SSE
  line to ``~/.deepseek_cli/sse_dump.log``. Useful when reasoning models
  leak chain-of-thought through unexpected paths and the parser needs
  updating.
"""
import asyncio
import base64
import json
import logging
import os
import random
from typing import AsyncGenerator, Optional

import httpx
from rich.console import Console

from .constants import (
    CONFIG_DIR,
    MAX_RETRIES,
    POW_RETRY_LIMIT,
    RETRY_BACKOFF,
    AUTO_COMPACT_THRESHOLD,
)
from .exceptions import AuthExpiredError, _PowExpiredError, _SessionNotFoundError
from .models import APIConfig
from .pow import DeepSeekHash, solve_pow_node

console = Console()
logger = logging.getLogger(__name__)

# Set DEEPSEEK_DUMP_SSE=1 to mirror every raw SSE line to disk for debugging
# unknown paths/types emitted by reasoning models. Outputs to
# ``~/.deepseek_cli/sse_dump.log`` (truncated on first call so each run
# starts fresh).
_SSE_DUMP_ENABLED = os.environ.get("DEEPSEEK_DUMP_SSE") == "1"
_SSE_DUMP_PATH = CONFIG_DIR / "sse_dump.log"
_SSE_DUMP_INITIALIZED = False


def _dump_sse(line: str) -> None:
    global _SSE_DUMP_INITIALIZED
    if not _SSE_DUMP_ENABLED:
        return
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        mode = "w" if not _SSE_DUMP_INITIALIZED else "a"
        with open(_SSE_DUMP_PATH, mode, encoding="utf-8") as f:
            f.write(line + "\n")
        _SSE_DUMP_INITIALIZED = True
    except OSError:
        pass


def _classify_path_for_thinking(p: str) -> bool:
    """Return True if SSE patch path indicates chain-of-thought / reasoning.

    DeepSeek's web schema uses several names interchangeably across model
    versions: ``thinking_content``, ``thinking``, ``reasoning_content``,
    ``reasoning``, ``chain_of_thought``, ``cot``, plus localized variants.
    Matching is case-insensitive and substring-based on purpose: missing a
    new name is the failure mode we want (false positives never hide real
    content because the parser still falls through other detectors).
    """
    pl = (p or "").lower()
    return (
        "think" in pl
        or "reasoning" in pl
        or "chain_of_thought" in pl
        or "/cot" in pl
        or pl.endswith("cot")
    )


# Type alias for stream tokens
Token = tuple[str, str]  # (type, text) where type is "text" or "thinking"


def _backoff_seconds(attempt: int) -> float:
    """RETRY_BACKOFF[attempt] with ±25 % jitter to avoid retry storms."""
    base = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
    return base * (0.75 + 0.5 * random.random())


# ── Stateful <think>-tag stripper ────────────────────────────────────────────
# Reasoning models (R1) sometimes route their chain-of-thought through the
# normal "text" channel rather than a dedicated "thinking" path. We strip
# anything between literal <think> and </think> tokens, even when the tags
# are split across SSE chunks. State machine, no regex.

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


async def _strip_think_tags(
    upstream: AsyncGenerator[tuple[str, str], None],
    show_thinking: bool,
) -> AsyncGenerator[tuple[str, str], None]:
    """Wrap an upstream token generator and remove ``<think>...</think>`` blocks.

    * ``thinking`` tokens pass through (already routed by the parser).
    * ``text`` tokens have any think-blocks redacted. A buffer holds bytes
      that *might* be the start of a tag so a chunk boundary inside
      ``<thi`` doesn't slip through.
    """
    in_think = False
    pending = ""  # holds an in-progress (possibly-truncated) tag prefix
    longest_tag = max(len(_THINK_OPEN), len(_THINK_CLOSE))

    async for token_type, text in upstream:
        if token_type != "text":
            yield (token_type, text)
            continue

        buffer = pending + text
        pending = ""
        out: list[str] = []

        while buffer:
            if in_think:
                # Looking for </think>
                close_idx = buffer.find(_THINK_CLOSE)
                if close_idx == -1:
                    # Hold the tail in case </think> is split across chunks.
                    keep = min(len(_THINK_CLOSE) - 1, len(buffer))
                    pending = buffer[len(buffer) - keep:] if keep else ""
                    buffer = ""
                else:
                    if show_thinking:
                        # The redacted text would have been chain-of-thought.
                        # We don't re-emit it as a thinking token because
                        # that channel is already handled by the parser; we
                        # just drop it silently here to avoid double output.
                        pass
                    buffer = buffer[close_idx + len(_THINK_CLOSE):]
                    in_think = False
            else:
                open_idx = buffer.find(_THINK_OPEN)
                if open_idx == -1:
                    # No full open tag. Hold a tail in case <think> is split.
                    keep = min(longest_tag - 1, len(buffer))
                    if keep:
                        out.append(buffer[: len(buffer) - keep])
                        pending = buffer[len(buffer) - keep:]
                    else:
                        out.append(buffer)
                        pending = ""
                    buffer = ""
                else:
                    out.append(buffer[:open_idx])
                    buffer = buffer[open_idx + len(_THINK_OPEN):]
                    in_think = True

        clean = "".join(out)
        if clean:
            yield ("text", clean)

    # Drain trailing pending bytes if we're not inside a think block.
    if pending and not in_think:
        yield ("text", pending)


class APIClient:
    """DeepSeek web API client with full resilience and fallback handling."""

    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.base_url = config.target_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.last_message_id: Optional[str] = None
        self._is_first_message: bool = True
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=15.0),
            follow_redirects=True,
        )
        self.thinking_enabled = False
        self.search_enabled = False
        self.model_type = "expert"
        self.show_thinking = False
        self._wasm_solver: Optional[DeepSeekHash] = None  # lazy-init, cached
        self._pending_file_ids: list[str] = []  # injected by server for file uploads
        self.system_prompt: str = ""  # prepended to first message of each session
        # Track flags used by current session — reset session if they change
        self._session_flags: Optional[tuple[str, bool, bool]] = None  # (model_type, thinking, search)
        # ── Auto-compact (rolling-summary) ────────────────────
        # When _turn_count reaches auto_compact_threshold, the next
        # send_message_stream() will:
        #   1. Ask the current session for a concise summary.
        #   2. Reset the session.
        #   3. Prepend that summary to the first message of the fresh session
        #      so the new session inherits context.
        # Set ``auto_compact_threshold = 0`` to disable.
        self.auto_compact_threshold: int = AUTO_COMPACT_THRESHOLD
        self._turn_count: int = 0
        self._compact_summary: str = ""  # consumed once on next first message
        # Set to True for one read after a compact; consumers (server) clear after use.
        self._just_compacted: bool = False

    # ── Header construction ───────────────────────────────────

    def _build_headers(self, accept: str = "text/event-stream") -> dict:
        skip = {"accept", "content-type", "authorization", "cookie", "x-ds-pow-response"}
        h: dict = {k: v for k, v in self.config.headers.items() if k.lower() not in skip}
        h["Accept"] = accept
        h["Content-Type"] = "application/json"
        h["Origin"] = self.base_url
        h["Referer"] = self.base_url + "/"
        h["sec-fetch-site"] = "same-origin"
        h["sec-fetch-mode"] = "cors"
        h["sec-fetch-dest"] = "empty"
        if self.config.auth_token:
            h["Authorization"] = f"Bearer {self.config.auth_token}"
        if self.config.cookies:
            h["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.config.cookies.items())
        if self.config.pow_response:
            h["x-ds-pow-response"] = self.config.pow_response
        return h

    def _is_deepseek_mode(self) -> bool:
        return "deepseek" in self.base_url.lower()

    # ── Session management ────────────────────────────────────

    async def _create_session(self) -> str:
        """Create a new DeepSeek chat session and return session_id."""
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                r = await self.client.post(
                    f"{self.base_url}{self.config.session_path}",
                    headers=self._build_headers(accept="application/json"),
                    json={},
                )
                if r.status_code == 401:
                    raise AuthExpiredError("Token expired — run /reauth")
                if r.status_code == 403:
                    raise AuthExpiredError("Access forbidden (403) — run /reauth")
                if 500 <= r.status_code < 600:
                    last_exc = RuntimeError(
                        f"Session creation: server error {r.status_code}"
                    )
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Session creation failed HTTP {r.status_code}: {r.text[:300]}"
                    )
                try:
                    data = r.json()
                except Exception:
                    raise RuntimeError(
                        f"Session creation: invalid JSON ({r.status_code}): {r.text[:300]}"
                    )
                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"Session creation: unexpected response: {r.text[:300]}"
                    )
                biz_code = data.get("code", -1)
                if biz_code not in (0, None):
                    raise RuntimeError(
                        f"Session creation failed code={biz_code}: {data.get('msg', '')}"
                    )
                sid = (
                    (data.get("data") or {}).get("biz_data") or {}
                ).get("chat_session") or {}
                sid = sid.get("id", "") if isinstance(sid, dict) else ""
                if not sid:
                    raise RuntimeError(f"Could not create session. Response: {data}")
                self.session_id = sid
                self.last_message_id = None
                self._is_first_message = True  # new session → next message gets system prompt
                return sid

            except (AuthExpiredError, RuntimeError):
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exc = e
                if attempt >= MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Session creation failed after {MAX_RETRIES} retries: {e}"
                    )
                wait = _backoff_seconds(attempt)
                await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt >= MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Session creation failed after {MAX_RETRIES} retries: {e}"
                    )
                await asyncio.sleep(_backoff_seconds(attempt))

        raise RuntimeError(
            f"Session creation failed after {MAX_RETRIES} retries: {last_exc}"
        )

    # ── PoW challenge ─────────────────────────────────────────

    async def _fetch_pow_challenge(self, target_path: Optional[str] = None) -> Optional[dict]:
        """Fetch a fresh PoW challenge. Returns challenge dict or None."""
        path = target_path or self.config.completion_path
        headers = self._build_headers(accept="application/json")
        for attempt in range(MAX_RETRIES):
            try:
                r = await self.client.post(
                    f"{self.base_url}{self.config.pow_challenge_path}",
                    headers=headers,
                    json={"target_path": path},
                )
                if r.status_code == 401:
                    raise AuthExpiredError("Token expired — run /reauth")
                if r.status_code != 200:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                data = r.json()
                biz = (
                    data.get("data", {}).get("biz_data", {}).get("challenge")
                    or data.get("biz_data", {}).get("challenge")
                    or data.get("challenge")
                    or data
                )
                if not isinstance(biz, dict):
                    return None
                return biz
            except AuthExpiredError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError):
                await asyncio.sleep(_backoff_seconds(attempt))
            except Exception:
                return None
        return None

    def _solve_pow_wasm(
        self, algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int
    ) -> Optional[int]:
        """Layer 1: solve PoW using the embedded WASM engine (cached instance)."""
        try:
            if self._wasm_solver is None:
                self._wasm_solver = DeepSeekHash().init()
            ans = self._wasm_solver.calculate_hash(
                algorithm, challenge, salt, difficulty, expire_at
            )
            if ans == 0:
                return None
            return ans
        except Exception:
            self._wasm_solver = None  # reset so next call re-inits
            return None

    def _build_pow_token(self, biz_data: dict, answer: int, target_path: Optional[str] = None) -> str:
        """Encode a solved PoW challenge into the x-ds-pow-response header value."""
        payload = {
            "algorithm": biz_data.get("algorithm", "DeepSeekHashV1"),
            "challenge": biz_data["challenge"],
            "salt": biz_data["salt"],
            "answer": answer,
            "signature": biz_data.get("signature", ""),
            "target_path": target_path or self.config.completion_path,
        }
        return base64.b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode("utf-8")

    async def _generate_pow_token(self) -> str:
        """Fetch challenge and solve PoW for the completion endpoint."""
        return await self._generate_pow_token_for_path(self.config.completion_path)

    async def _generate_pow_token_for_path(self, target_path: str) -> str:
        """Fetch challenge and solve PoW for any target path.

        Layer 1: WASM → Layer 2: Node.js → Layer 3: empty string (proceed anyway)
        """
        biz_data = await self._fetch_pow_challenge(target_path=target_path)
        if not biz_data:
            console.print("[yellow]⚠ Could not fetch PoW challenge — proceeding without token[/]")
            return ""

        algorithm = biz_data.get("algorithm", "")
        challenge = biz_data.get("challenge", "")
        salt = biz_data.get("salt", "")
        difficulty = biz_data.get("difficulty", 144000)
        expire_at = biz_data.get("expire_at")

        if not challenge or not expire_at:
            return ""

        if algorithm and algorithm != "DeepSeekHashV1":
            console.print(f"[yellow]⚠ Unknown PoW algorithm '{algorithm}' — skipping PoW[/]")
            return ""

        # Layer 1: WASM
        ans = self._solve_pow_wasm(algorithm, challenge, salt, difficulty, expire_at)

        # Layer 2: Node.js
        if ans is None:
            ans = solve_pow_node(biz_data)

        if ans is None:
            console.print("[yellow]⚠ All PoW solvers failed — proceeding without token[/]")
            return ""

        return self._build_pow_token(biz_data, ans, target_path=target_path)

    # ── Streaming ─────────────────────────────────────────────

    async def _stream_completion(self, message: str) -> AsyncGenerator[Token, None]:
        """POST to DeepSeek completion endpoint and yield (type, text) tuples."""
        # Auto-compact: if we've reached the threshold, summarize → reset → continue.
        # Skipped for the summary call itself (avoid recursion via SUMMARY_PROMPT).
        if (
            self.auto_compact_threshold > 0
            and self._turn_count >= self.auto_compact_threshold
            and self.session_id
            and message != self.SUMMARY_PROMPT
        ):
            try:
                summary = await self.request_summary()
            except Exception:
                summary = ""
            if summary:
                # Reset, then stash so the upcoming send below carries the summary.
                self.session_id = None
                self.last_message_id = None
                self._is_first_message = True
                self._session_flags = None
                self._turn_count = 0
                self._compact_summary = summary
                self._just_compacted = True
                try:
                    from .metrics import metrics
                    metrics.incr("auto_compacts_total")
                except Exception:
                    pass
                console.print(
                    f"[dim]── Auto-compact: summarized previous session "
                    f"(was {self.auto_compact_threshold}+ turns) ──[/]"
                )

        # DeepSeek binds mode (model_type / thinking / search) to a session at
        # creation time. Changing flags mid-conversation has no effect on the
        # web API, so force a fresh session whenever flags differ from the
        # ones the current session was created with.
        current_flags = (self.model_type, self.thinking_enabled, self.search_enabled)
        if self.session_id and self._session_flags != current_flags:
            self.session_id = None
            self.last_message_id = None
            self._is_first_message = True

        if not self.session_id:
            await self._create_session()
            self._session_flags = current_flags

        pow_retries = 0
        net_retries = 0
        # Snapshot so retries reuse the same payload context.
        # `first_message_at_start` may be re-read after a session reset.
        first_message_at_start = self._is_first_message
        # Consume pending file ids EAGERLY: take ownership of them now so a
        # mid-flight failure (4xx bounce, network error) doesn't leak them
        # into the next, unrelated request.
        file_ids_snapshot = list(self._pending_file_ids)
        self._pending_file_ids = []
        consumed = False

        while True:
            completion_url = f"{self.base_url}{self.config.completion_path}"
            # Build the prompt for the *first* message of a fresh session:
            # optionally prepend system prompt and/or rolling summary.
            if first_message_at_start:
                preamble_parts: list[str] = []
                if self.system_prompt:
                    preamble_parts.append(self.system_prompt)
                if self._compact_summary:
                    preamble_parts.append(
                        "[Previous conversation summary]\n" + self._compact_summary
                    )
                if preamble_parts:
                    prompt_text = "\n\n".join(preamble_parts) + "\n\n" + message
                else:
                    prompt_text = message
            else:
                prompt_text = message
            body = {
                "chat_session_id": self.session_id,
                "parent_message_id": self.last_message_id,
                "model_type": self.model_type,
                "prompt": prompt_text,
                "ref_file_ids": file_ids_snapshot,
                "thinking_enabled": self.thinking_enabled,
                "search_enabled": self.search_enabled,
            }
            headers = self._build_headers()
            pow_token = await self._generate_pow_token()
            if pow_token:
                headers["x-ds-pow-response"] = pow_token

            try:
                async for token in self._do_stream(completion_url, headers, body):
                    if not consumed:
                        # Only mark consumed once we've actually started receiving
                        consumed = True
                        self._is_first_message = False
                        # Summary was injected into this turn's prompt → consume it
                        if first_message_at_start and self._compact_summary:
                            self._compact_summary = ""
                    yield token
                # Stream finished cleanly; ensure flags consumed even if no tokens yielded
                if not consumed:
                    self._is_first_message = False
                    if first_message_at_start and self._compact_summary:
                        self._compact_summary = ""
                # Count turn unless this was the summary call itself
                if message != self.SUMMARY_PROMPT:
                    self._turn_count += 1
                return

            except _PowExpiredError:
                if consumed:
                    # Already streamed text to caller — re-sending would dup output
                    raise RuntimeError(
                        "PoW rejected mid-stream after partial output — cannot retry"
                    )
                pow_retries += 1
                if pow_retries > POW_RETRY_LIMIT:
                    raise AuthExpiredError(
                        "PoW keeps failing. Run [bold]/reauth[/] — "
                        "send a message in the browser first."
                    )
                console.print(f"[yellow]⚠ PoW rejected, re-solving (attempt {pow_retries})…[/]")
                self._wasm_solver = None
                await asyncio.sleep(1)

            except _SessionNotFoundError:
                if consumed:
                    raise RuntimeError(
                        "Session lost mid-stream after partial output — cannot retry"
                    )
                self.session_id = None
                self.last_message_id = None
                await self._create_session()
                # New session → re-send system prompt next attempt
                first_message_at_start = self._is_first_message

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if consumed:
                    raise RuntimeError(f"Network error mid-stream: {e}")
                net_retries += 1
                if net_retries > MAX_RETRIES:
                    raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {e}")
                wait = _backoff_seconds(net_retries - 1)
                console.print(f"[yellow]⚠ Network error ({e}), retrying in {wait:.1f}s…[/]")
                await asyncio.sleep(wait)

    async def _do_stream(
        self, url: str, headers: dict, body: dict
    ) -> AsyncGenerator[Token, None]:
        """Inner SSE parsing loop. Yields (type, text) tuples.

        Handles 4 SSE formats:
          A1: DeepSeek JSON-patch delta  {"p":..., "v":"text", "o":"APPEND"}
          A2: DeepSeek snapshot          {"v": {"response": {...}}}
          B:  OpenAI choices[].delta
          C:  top-level "content" key
          D:  top-level "text" key
        """
        new_message_id: Optional[str] = None
        current_event: Optional[str] = None
        json_buf = ""
        streamed_text = ""
        # Cap buffer to prevent unbounded growth on malformed SSE
        MAX_JSON_BUF = 1024 * 1024  # 1 MiB

        # Track the active fragment's type (e.g. "THINK" vs "RESPONSE") so
        # we can route content patches correctly when DeepSeek references
        # them by index (response/fragments/-1/content) without repeating
        # the type in every patch.
        active_fragment_type: str = ""

        # DeepSeek sends an initial patch with an explicit ``p`` (path) and
        # then a long run of patches with NO ``p`` at all — those implicitly
        # target the previous path. Track the last one so we can reuse it.
        last_path: str = ""

        # Only these SSE event types carry content
        CONTENT_EVENTS = {None, "message", "add_message", "completion"}

        async with self.client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code == 401:
                raise AuthExpiredError("Token expired — run /reauth")
            if response.status_code == 403:
                raise AuthExpiredError("Access forbidden (403) — run /reauth")
            if response.status_code == 429:
                raise RuntimeError("Rate limited (429) — wait a moment and try again")
            if 500 <= response.status_code < 600:
                # 5xx → treat as network-class error so retry loop can recover
                raw = await response.aread()
                raise httpx.NetworkError(
                    f"Upstream {response.status_code}: {raw[:300]}"
                )
            if response.status_code != 200:
                raw = await response.aread()
                raise RuntimeError(f"API error {response.status_code}: {raw[:800]}")

            async for line in response.aiter_lines():
                _dump_sse(line)
                if line.startswith(":"):
                    # SSE comment / keepalive — ignore
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    json_buf = ""
                    continue

                if not line.strip():
                    current_event = None
                    json_buf = ""
                    continue

                if not line.startswith("data:"):
                    self._check_error_line(line)
                    continue

                chunk = line[5:].lstrip(" ")
                if chunk.strip() in ("[DONE]", ""):
                    break

                # Buffer incomplete JSON (httpx may split long lines)
                json_buf += chunk
                if len(json_buf) > MAX_JSON_BUF:
                    # Drop runaway buffer to avoid OOM
                    json_buf = ""
                    continue
                try:
                    data = json.loads(json_buf)
                    json_buf = ""
                except json.JSONDecodeError:
                    continue  # wait for more chunks

                if not isinstance(data, dict):
                    # Server emitted non-object payload (list/scalar) — skip
                    continue

                if current_event not in CONTENT_EVENTS:
                    continue

                self._check_error_data(data)

                mid = data.get("message_id") or data.get("response_message_id")
                if mid:
                    new_message_id = mid

                # ── Format detection ──────────────────────────
                yielded = False
                v = data.get("v")

                # New-fragment notifications: a list APPEND on path
                # "response/fragments" introduces a new fragment whose
                # ``type`` we must remember so subsequent patches at
                # "response/fragments/-1/content" route to the right channel.
                if (
                    isinstance(v, list)
                    and data.get("o") == "APPEND"
                    and data.get("p", "").lower().endswith("response/fragments")
                ):
                    for frag in v:
                        if isinstance(frag, dict):
                            ft = frag.get("type")
                            if isinstance(ft, str):
                                active_fragment_type = ft
                                # If the new fragment ships with seed
                                # content, emit/route it now too.
                                seed = frag.get("content", "")
                                if isinstance(seed, str) and seed:
                                    if active_fragment_type.upper() in (
                                        "THINK", "THINKING", "REASONING", "COT"
                                    ):
                                        if self.show_thinking:
                                            yield ("thinking", seed)
                                    else:
                                        streamed_text += seed
                                        yield ("text", seed)
                    yielded = True

                if v is not None and not yielded:
                    if isinstance(v, str):
                        # Inherit the previous path when the patch omits it.
                        # DeepSeek sends one explicit ``p`` per content run,
                        # then a long stream of bare ``{"v": "..."}`` rows.
                        p = data.get("p")
                        if p:
                            last_path = p
                        elif "p" not in data:
                            # No path on this row → reuse the last one.
                            p = last_path
                        else:
                            p = ""
                        o = data.get("o", "")
                        p_lower = (p or "").lower()
                        # Path-based hint OR fragment-type-based hint:
                        # DeepSeek's R1 emits content patches with path
                        # "response/fragments/-1/content" but the fragment
                        # itself was declared as type=THINK in an earlier
                        # snapshot. We must honour either signal.
                        is_fragment_content = "fragments/" in p_lower and (
                            p_lower.endswith("/content") or p_lower.endswith("content")
                        )
                        active_is_think = active_fragment_type.upper() in (
                            "THINK", "THINKING", "REASONING", "COT"
                        )
                        is_thinking = (
                            _classify_path_for_thinking(p)
                            or (is_fragment_content and active_is_think)
                        )
                        # `is_content` only kicks in when path is empty or
                        # explicitly references content AND it's not also
                        # marked as thinking (e.g. `thinking_content`, or a
                        # fragment whose declared type is THINK).
                        is_content = (
                            not is_thinking
                            and (not p or p.endswith("/content") or p.endswith("content"))
                        )
                        is_append = o == "APPEND" or not o
                        is_replace = o == "REPLACE"
                        if v and (is_append or is_replace):
                            if is_thinking:
                                if self.show_thinking:
                                    yield ("thinking", v)
                                yielded = True
                            elif is_content:
                                if is_replace:
                                    # REPLACE: only emit the diff vs streamed_text
                                    if v != streamed_text:
                                        if streamed_text and v.startswith(streamed_text):
                                            diff = v[len(streamed_text):]
                                            if diff:
                                                streamed_text = v
                                                yield ("text", diff)
                                        else:
                                            streamed_text = v
                                            yield ("text", v)
                                else:
                                    streamed_text += v
                                    yield ("text", v)
                                yielded = True
                    elif isinstance(v, dict):
                        # A2 snapshot — extract message_id, reconcile missing prefix
                        resp = v.get("response", {})
                        if isinstance(resp, dict) and "message_id" in resp:
                            new_message_id = resp["message_id"]
                        if isinstance(resp, dict):
                            # Update active fragment type for subsequent
                            # path-only patches (`response/fragments/-1/content`).
                            frags = resp.get("fragments")
                            if isinstance(frags, list) and frags:
                                last_frag = frags[-1] if isinstance(frags[-1], dict) else {}
                                ft = last_frag.get("type")
                                if isinstance(ft, str):
                                    active_fragment_type = ft

                            snapshot = ""
                            if isinstance(frags, list):
                                # Pick the first fragment that is NOT a thinking
                                # block. Reasoning models can emit multiple
                                # fragments where index 0 is the chain-of-thought.
                                for frag in frags:
                                    if not isinstance(frag, dict):
                                        continue
                                    ftype = (frag.get("type") or "").upper()
                                    if ftype in ("THINK", "THINKING", "REASONING", "COT"):
                                        continue
                                    snapshot = frag.get("content", "") or ""
                                    if snapshot:
                                        break
                            if not snapshot:
                                snapshot = resp.get("accumulate", "") or ""
                            if snapshot and not streamed_text:
                                streamed_text = snapshot
                                yield ("text", snapshot)
                            elif snapshot and streamed_text and snapshot != streamed_text:
                                idx = snapshot.find(streamed_text)
                                if idx > 0:
                                    missing = snapshot[:idx]
                                    streamed_text = missing + streamed_text
                                    yield ("text", missing)
                        yielded = True  # A2 handled — don't fall through to B/C/D

                if not yielded:
                    choices = data.get("choices") or []
                    if choices and isinstance(choices, list):
                        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                        dtype = delta.get("type")
                        if dtype == "text":
                            text = delta.get("text", "")
                            if text:
                                yield ("text", text)
                                yielded = True
                        elif dtype == "thinking":
                            thinking = delta.get("thinking", "") or delta.get("text", "")
                            if thinking and self.show_thinking:
                                yield ("thinking", thinking)
                                yielded = True
                        elif dtype not in ("tool_call", None):
                            pass  # unknown delta type — silently ignore
                        elif "content" in delta and delta["content"]:
                            yield ("text", delta["content"])
                            yielded = True

                if not yielded and isinstance(data.get("content"), str) and data["content"]:
                    yield ("text", data["content"])
                    yielded = True

                if not yielded and isinstance(data.get("text"), str) and data["text"]:
                    yield ("text", data["text"])

        if new_message_id:
            self.last_message_id = new_message_id

    def _check_error_line(self, line: str) -> None:
        """Check a non-data SSE line for business-level error codes."""
        try:
            biz = json.loads(line)
            if isinstance(biz, dict):
                self._raise_for_code(biz.get("code", 0), biz.get("msg", ""))
        except (json.JSONDecodeError, ValueError):
            pass

    def _check_error_data(self, data: dict) -> None:
        """Check a parsed data: payload for business-level error codes."""
        if isinstance(data, dict):
            code = data.get("code")
            if code is not None:
                self._raise_for_code(code, data.get("msg", ""))

    @staticmethod
    def _raise_for_code(code, msg: str) -> None:
        # Coerce string codes to int (some upstreams return "40300")
        if isinstance(code, str):
            try:
                code = int(code)
            except ValueError:
                return  # not a numeric code — ignore
        if not isinstance(code, int):
            return
        if code in (40300, 40301):
            raise _PowExpiredError(f"PoW error code={code}")
        if code in (40200, 40201):
            raise _SessionNotFoundError(f"Session not found code={code}")
        if code in (40100, 40101, 40103, 401, 403):
            raise AuthExpiredError(f"Auth error code={code} — run /reauth")
        if code != 0:
            raise RuntimeError(f"API error {code}: {msg}")

    # ── Generic fallback ──────────────────────────────────────

    async def _stream_generic(self, message: str) -> AsyncGenerator[Token, None]:
        """Generic replay for non-DeepSeek discovered endpoints (OpenAI-style)."""
        headers = self._build_headers()
        body = dict(self.config.body_template)
        if "messages" in body and isinstance(body["messages"], list):
            body["messages"].append({"role": "user", "content": message})

        for attempt in range(MAX_RETRIES):
            try:
                async with self.client.stream(
                    method=self.config.method,
                    url=self.config.api_endpoint,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code == 401:
                        raise AuthExpiredError("Token expired — run /reauth")
                    if response.status_code != 200:
                        raw = await response.aread()
                        raise RuntimeError(f"API error {response.status_code}: {raw[:500]}")

                    if self.config.use_stream:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                chunk = line[6:]
                                if chunk.strip() in ("[DONE]", ""):
                                    break
                                try:
                                    data = json.loads(chunk)
                                    delta = (
                                        data.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                    if delta:
                                        yield ("text", delta)
                                except json.JSONDecodeError:
                                    pass
                    else:
                        raw = await response.aread()
                        data = json.loads(raw)
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        yield ("text", content)
                return

            except (AuthExpiredError, RuntimeError):
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt >= MAX_RETRIES - 1:
                    raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {e}")
                wait = _backoff_seconds(attempt)
                console.print(f"[yellow]⚠ Network error ({e}), retrying in {wait:.1f}s…[/]")
                await asyncio.sleep(wait)

    # ── Public interface ──────────────────────────────────────

    async def send_message_stream(self, message: str) -> AsyncGenerator[Token, None]:
        """Route to DeepSeek-specific or generic stream. Yields (type, text) tuples.

        Wraps the upstream stream with a stateful ``<think>...</think>`` tag
        stripper that handles tags split across SSE chunks. Reasoning models
        (R1) sometimes leak chain-of-thought into the text channel; this
        strips it so callers see only the user-facing answer (unless
        ``self.show_thinking`` is True).
        """
        if self._is_deepseek_mode():
            inner = self._stream_completion(message)
        else:
            inner = self._stream_generic(message)
        async for token_type, text in _strip_think_tags(inner, self.show_thinking):
            yield (token_type, text)

    async def reset_session(self) -> None:
        """Reset session state (for /new command or API server fresh conversation)."""
        self.session_id = None
        self.last_message_id = None
        self._wasm_solver = None
        self._is_first_message = True  # reset flag when session resets
        self._session_flags = None  # next message will create a fresh session
        self._turn_count = 0
        self._compact_summary = ""

    def set_pending_files(self, file_ids: list[str]) -> None:
        """Queue file ids for the *next* outgoing message (consumed once).

        Pass an empty list to clear. Always copies — caller's list may mutate.
        """
        self._pending_file_ids = list(file_ids)

    # ── Auto-compact (rolling-summary continuation) ───────────

    SUMMARY_PROMPT = (
        "Summarize this conversation so far in 5-10 bullet points. "
        "Capture: (1) what the user is working on, (2) decisions made, "
        "(3) facts/context the assistant should remember, (4) any pending tasks. "
        "Output ONLY the bullets, no preamble."
    )

    async def request_summary(self) -> str:
        """Ask the current session to summarize itself. Returns plain text.

        Returns empty string on failure (caller should keep using current session).
        Does NOT reset the session — caller decides what to do with the summary.
        """
        if not self.session_id:
            return ""
        # Snapshot then disable thinking output so summary is clean text
        prev_show = self.show_thinking
        self.show_thinking = False
        parts: list[str] = []
        try:
            async for token_type, token_text in self._stream_completion(self.SUMMARY_PROMPT):
                if token_type == "text":
                    parts.append(token_text)
        except Exception as e:
            console.print(f"[yellow]⚠ Could not get summary: {e}[/]")
            return ""
        finally:
            self.show_thinking = prev_show
        return "".join(parts).strip()

    async def compact_session(self) -> bool:
        """Force a compact: summarize → reset → queue summary for next first message.

        Returns True on success, False if no session or summary failed.
        Idempotent — calling on a fresh session is a no-op.
        """
        if not self.session_id:
            return False
        summary = await self.request_summary()
        if not summary:
            return False
        # Reset session, then stash the summary so the next message of the
        # fresh session begins with it as context.
        await self.reset_session()
        self._compact_summary = summary
        self._just_compacted = True
        try:
            from .metrics import metrics
            metrics.incr("auto_compacts_total")
        except Exception:
            pass
        return True

    # ── File upload ───────────────────────────────────────────

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        upload_path: str = "/api/v0/file/upload_file",
    ) -> str:
        """Upload a file via DeepSeek's upload endpoint and return the file_id.

        Raises:
            AuthExpiredError: on 401/403
            RuntimeError: on any other failure (no file_id, non-200, etc.)
        """
        cfg = self.config
        upload_url = f"{cfg.target_url.rstrip('/')}{upload_path}"

        # Build headers — strip body-related ones, set Origin/Referer
        headers = {
            k: v for k, v in cfg.headers.items()
            if k.lower() not in ("authorization", "cookie", "content-length", "host", "content-type")
        }
        if cfg.auth_token:
            headers["Authorization"] = f"Bearer {cfg.auth_token}"
        headers["Origin"] = cfg.target_url.rstrip("/")
        headers["Referer"] = cfg.target_url.rstrip("/") + "/"
        headers["sec-fetch-site"] = "same-origin"
        headers["sec-fetch-mode"] = "cors"
        headers["sec-fetch-dest"] = "empty"
        if cfg.cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cfg.cookies.items())

        pow_token = await self._generate_pow_token_for_path(upload_path)
        if pow_token:
            headers["x-ds-pow-response"] = pow_token

        resp = await self.client.post(
            upload_url,
            headers=headers,
            files={"file": (filename, content, content_type)},
        )
        if resp.status_code in (401, 403):
            raise AuthExpiredError(f"Auth expired during upload (HTTP {resp.status_code})")
        if resp.status_code != 200:
            raise RuntimeError(
                f"Upload failed: HTTP {resp.status_code} — {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Upload returned non-JSON: {resp.text[:200]}")
        if not isinstance(data, dict):
            raise RuntimeError(f"Upload returned unexpected payload: {data}")
        file_id = (
            (data.get("data") or {}).get("biz_data", {}).get("id")
            or (data.get("data") or {}).get("id")
            or data.get("id")
        )
        if not file_id:
            raise RuntimeError(f"Could not extract file_id from response: {data}")
        return str(file_id)

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass
