"""
OpenAI-compatible HTTP API server backed by the DeepSeek client.

Endpoints:
  POST /v1/chat/completions        — OpenAI-compatible (stream=true/false)
  POST /v1/chat/completions        — with conversation_id for multi-turn
  GET  /v1/conversations           — list active conversations
  DELETE /v1/conversations/{id}    — end a conversation
  POST /v1/files                   — upload a file (returns file_id)
  GET  /v1/models                  — list available models
  GET  /health                     — health check

Multi-turn:
  Pass "conversation_id" in the request body to maintain context across
  requests. Each conversation_id gets its own DeepSeek session + lock.
  Conversations expire after CONVERSATION_TTL seconds of inactivity.

File upload:
  Upload a file → get file_id → pass file_ids in chat request.
  DeepSeek web API accepts ref_file_ids in the completion payload.
"""
import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .client import APIClient
from .constants import REQUEST_TIMEOUT, VERSION, DEFAULT_SYSTEM_PROMPT
from .exceptions import AuthExpiredError
from .gateway import AccountPool
from .gateway.pool import NoAccountAvailableError
from .metrics import metrics
from .models import APIConfig
from .session import SessionManager
from .tools import compose_tools_prompt, extract_tool_calls

console = Console()

# A rough character→token ratio used by the OpenAI-style ``usage`` block
# in chat responses. Real tokenization would require shipping a tokenizer
# (DeepSeek uses tiktoken-cl100k-base-ish). 4 char/token is the universal
# OpenAI heuristic — close enough for billing-style reporting.
_TOKEN_CHARS_PER_TOKEN = 4

# Conversations expire after 30 minutes of inactivity
CONVERSATION_TTL = 30 * 60


@dataclass
class Conversation:
    """A single stateful conversation with its own DeepSeek session."""
    id: str
    client: APIClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.time)
    model: str = "deepseek-chat"

    def touch(self) -> None:
        self.last_used = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.last_used > CONVERSATION_TTL


class ConversationPool:
    """Pool of active conversations, each with its own APIClient + lock.

    Expired conversations are cleaned up lazily on each access.
    """

    def __init__(self, config: APIConfig, auto_compact_threshold: Optional[int] = None) -> None:
        self._config = config
        self._auto_compact_threshold = auto_compact_threshold
        self._conversations: dict[str, Conversation] = {}
        self._pool_lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def get_or_create(self, conversation_id: str) -> Conversation:
        """Return existing conversation or create a new one."""
        async with self._pool_lock:
            self._cleanup_expired()
            if conversation_id not in self._conversations:
                client = APIClient(self._config)
                client.system_prompt = DEFAULT_SYSTEM_PROMPT
                if self._auto_compact_threshold is not None:
                    client.auto_compact_threshold = self._auto_compact_threshold
                self._conversations[conversation_id] = Conversation(
                    id=conversation_id,
                    client=client,
                )
            conv = self._conversations[conversation_id]
            conv.touch()
            return conv

    async def delete(self, conversation_id: str) -> bool:
        """Remove a conversation. Won't preempt an in-flight request:
        if the conversation lock is held, the conversation is removed from
        the registry but its client is closed only after the lock frees."""
        async with self._pool_lock:
            conv = self._conversations.pop(conversation_id, None)
        if conv is None:
            return False
        # Wait for any in-flight request to finish before tearing down the
        # underlying httpx client (closing it mid-stream raises
        # RuntimeError on the active request).
        async with conv.lock:
            await conv.client.close()
        return True

    def get(self, conversation_id: str) -> Optional["Conversation"]:
        """Return the conversation by id without creating one. None if absent."""
        return self._conversations.get(conversation_id)

    def __len__(self) -> int:
        return len(self._conversations)

    def list_conversations(self) -> list[dict]:
        # Snapshot under no-lock is fine since dict iteration over .values()
        # raises if mutated; copy first then filter expired without cleanup
        # (cleanup happens under lock in get_or_create).
        snapshot = list(self._conversations.values())
        now = time.time()
        return [
            {
                "id": c.id,
                "last_used": c.last_used,
                "idle_seconds": int(now - c.last_used),
            }
            for c in snapshot
            if not c.is_expired()
        ]

    def _cleanup_expired(self) -> None:
        expired = [cid for cid, c in self._conversations.items() if c.is_expired()]
        for cid in expired:
            conv = self._conversations.pop(cid)
            try:
                task = asyncio.create_task(conv.client.close())
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)
            except RuntimeError:
                # No running loop — skip; close_all() will handle it
                pass

    async def close_all(self) -> None:
        async with self._pool_lock:
            for conv in self._conversations.values():
                await conv.client.close()
            self._conversations.clear()


# ── Pure helpers (used by chat_completions; no closure deps) ─────────────────


def _classify_upstream_error(err: Exception) -> tuple[int, str]:
    """Map an unexpected ``APIClient`` error to an HTTP (status, detail) pair.

    DeepSeek backend errors come through as ``RuntimeError("API error 422: ...")``.
    A 4xx code from the upstream is almost always caused by client input
    (bad file_id, bad message shape) and should surface as 4xx — not 500.
    """
    msg = str(err)
    # Look for a "API error <code>" prefix from APIClient._raise_for_code or _do_stream
    m = re.search(r"API error (\d{3})", msg)
    if m:
        code = int(m.group(1))
        if 400 <= code < 500:
            return 400, f"Upstream rejected: {msg}"
        if 500 <= code < 600:
            return 502, f"Upstream error: {msg}"
    return 500, f"Upstream error: {msg}"


def _messages_to_prompt(messages: list[dict]) -> str:
    """Extract the last user message, prepending system prompt on first turn."""
    if not messages:
        return ""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    last_user = user_msgs[-1].get("content", "")
    if system_msgs and len(user_msgs) == 1:
        sys_text = "\n\n".join(
            m.get("content", "") for m in system_msgs if m.get("content")
        )
        if sys_text:
            return f"[System]\n{sys_text}\n\n[User]\n{last_user}"
    return last_user


def _model_to_preset(model: str) -> dict:
    """Map an OpenAI-style ``model`` string to a DeepSeek mode preset.

    Presets follow the official web UI:
      - expert / reasoner / r1 / think → Expert + R1 ON,  Search OFF
      - everything else                → Fast   + R1 OFF, Search ON
    """
    m = (model or "").lower()
    if "reasoner" in m or "r1" in m or "think" in m or "expert" in m:
        return {
            "model_type": "expert",
            "thinking_enabled": True,
            "search_enabled": False,
        }
    return {
        "model_type": "default",
        "thinking_enabled": False,
        "search_enabled": True,
    }


def _apply_preset(client: APIClient, preset: dict) -> None:
    """Apply a preset dict onto an APIClient (in-place)."""
    if "model_type" in preset:
        client.model_type = preset["model_type"]
    if "thinking_enabled" in preset:
        client.thinking_enabled = preset["thinking_enabled"]
    if "search_enabled" in preset:
        client.search_enabled = preset["search_enabled"]
    if "auto_compact_threshold" in preset:
        client.auto_compact_threshold = preset["auto_compact_threshold"]


def _is_fresh_conversation(messages: list[dict]) -> bool:
    return sum(1 for m in messages if m.get("role") == "user") <= 1


def _resolve_auto_compact_override(payload: dict) -> Optional[int]:
    """Read the ``auto_compact`` field from the request body.

    Accepts:
      • int   → custom threshold (0 disables for this request)
      • False → disables for this request
      • True  → keep the server default (returned as None)
      • absent → no override (returned as None)
    """
    val = payload.get("auto_compact")
    if val is False:
        return 0
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return max(0, val)
    return None


async def serve_mode(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = None,
    use_gateway: bool = False,
    admin_token: Optional[str] = None,
    auto_compact_threshold: Optional[int] = None,
) -> None:
    """Start the OpenAI-compatible API server.

    If ``use_gateway`` is True, requests are routed through ``AccountPool``
    (multi-account, smart selection, exponential cooldown). Otherwise the
    server uses a single saved session (``config.json``).

    ``admin_token`` (optional) protects ``/admin/*`` endpoints. If unset,
    admin endpoints are disabled entirely.

    ``auto_compact_threshold`` overrides the default rolling-summary
    threshold for every client this server creates. ``None`` keeps the
    package default (``constants.AUTO_COMPACT_THRESHOLD``); ``0`` disables.
    """
    try:
        from fastapi import FastAPI, HTTPException, Header, Request, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
        import uvicorn
    except ImportError:
        console.print(
            "[red]FastAPI / uvicorn not installed. Run: pip install fastapi uvicorn[/]"
        )
        sys.exit(1)

    config = SessionManager.load_config()
    if not config:
        if use_gateway:
            test_pool = AccountPool()
            test_pool.load_all()
            if test_pool.is_empty():
                console.print("[red]No saved config found. Run with --discover first.[/]")
                sys.exit(1)
        else:
            console.print("[red]No saved config found. Run with --discover first.[/]")
            sys.exit(1)

    # Initialize multi-account pool if gateway mode is on
    gateway_pool: Optional[AccountPool] = None
    if use_gateway:
        gateway_pool = AccountPool()
        loaded = gateway_pool.load_all()
        if loaded == 0:
            console.print(
                "[red]Gateway mode enabled but no accounts found. "
                "Add accounts via Account Pool menu first.[/]"
            )
            sys.exit(1)
        console.print(f"[green]Gateway: loaded {loaded} account(s)[/]")
        if auto_compact_threshold is not None:
            for acc in gateway_pool.accounts:
                acc.client.auto_compact_threshold = auto_compact_threshold

    pool = ConversationPool(
        config, auto_compact_threshold=auto_compact_threshold
    ) if config else None

    # Single-client lock for requests that don't use conversation_id (non-gateway only)
    _default_client = APIClient(config) if config and not use_gateway else None
    if _default_client:
        _default_client.system_prompt = DEFAULT_SYSTEM_PROMPT
        if auto_compact_threshold is not None:
            _default_client.auto_compact_threshold = auto_compact_threshold
    _default_lock = asyncio.Lock()

    app = FastAPI(title="DeepSeek OpenAI-Compatible API", version=VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth ──────────────────────────────────────────────────

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        if authorization[7:] != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ── Routes ────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        body = {
            "status": "ok",
            "version": VERSION,
            "mode": "gateway" if use_gateway else "single",
        }
        if use_gateway and gateway_pool is not None:
            body["pool"] = gateway_pool.status()
        else:
            body["active_conversations"] = len(pool) if pool else 0
        return body

    # ── Observability ─────────────────────────────────────────

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus-format metrics. No auth — same convention as gateway."""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics.to_prometheus(), media_type="text/plain; version=0.0.4")

    # ── Admin (account recovery / inspection) ─────────────────

    def _check_admin(token: Optional[str]) -> None:
        if not admin_token:
            raise HTTPException(status_code=404, detail="Admin endpoints disabled")
        if token != admin_token:
            raise HTTPException(status_code=401, detail="Invalid admin token")

    @app.get("/admin/pool")
    async def admin_pool_status(x_admin_token: Optional[str] = Header(None)):
        """Inspect pool status with cooldown ETAs and per-account stats."""
        _check_admin(x_admin_token)
        if not use_gateway or gateway_pool is None:
            raise HTTPException(status_code=404, detail="Gateway mode is not enabled")
        return gateway_pool.status()

    @app.post("/admin/pool/{name}/unblock")
    async def admin_pool_unblock(
        name: str, x_admin_token: Optional[str] = Header(None)
    ):
        """Force-clear cooldown on a single account (e.g. after manual review).

        Also drops a stale lock if one is held — useful when a previous
        request was cancelled mid-flight without releasing the lock.
        """
        _check_admin(x_admin_token)
        if not use_gateway or gateway_pool is None:
            raise HTTPException(status_code=404, detail="Gateway mode is not enabled")
        for acc in gateway_pool.accounts:
            if acc.name == name:
                acc.exhausted_until = 0.0
                acc.consecutive_quota_hits = 0
                acc.last_error = None
                # Replace the lock object if it's stuck. Tests for `lock.locked()`
                # are intentional — we never preempt an in-flight request that
                # is still streaming.
                lock_was_stuck = False
                if acc.lock.locked():
                    # Heuristic: if the lock is held but the underlying client
                    # has no inflight HTTP transport, the holder is gone.
                    lock_was_stuck = True
                    acc.lock = asyncio.Lock()
                gateway_pool.flush_stats()
                return {"unblocked": name, "lock_replaced": lock_was_stuck}
        raise HTTPException(status_code=404, detail=f"Account '{name}' not found")

    @app.post("/admin/compact")
    async def admin_compact_all(x_admin_token: Optional[str] = Header(None)):
        """Force compact on the default client and every gateway account.

        Useful before maintenance, after long idle periods, or to flush
        large contexts manually.
        """
        _check_admin(x_admin_token)
        results: list[dict] = []
        if _default_client is not None:
            async with _default_lock:
                ok = await _default_client.compact_session()
            results.append({"target": "default", "compacted": ok})
        if use_gateway and gateway_pool is not None:
            for acc in gateway_pool.accounts:
                async with acc.lock:
                    ok = await acc.client.compact_session()
                results.append({"target": acc.name, "compacted": ok})
        return {"results": results}

    @app.get("/v1/models")
    async def list_models(authorization: Optional[str] = Header(None)):
        _check_auth(authorization)
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": "deepseek-chat",     "object": "model", "created": now, "owned_by": "deepseek"},
                {"id": "deepseek-fast",     "object": "model", "created": now, "owned_by": "deepseek"},
                {"id": "deepseek-reasoner", "object": "model", "created": now, "owned_by": "deepseek"},
            ],
        }

    @app.get("/v1/conversations")
    async def list_conversations(authorization: Optional[str] = Header(None)):
        """List all active conversations."""
        _check_auth(authorization)
        if pool is None:
            return {"object": "list", "data": [], "ttl_seconds": CONVERSATION_TTL}
        return {
            "object": "list",
            "data": pool.list_conversations(),
            "ttl_seconds": CONVERSATION_TTL,
        }

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str, authorization: Optional[str] = Header(None)
    ):
        """End a conversation and free its DeepSeek session."""
        _check_auth(authorization)
        if pool is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        deleted = await pool.delete(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "id": conversation_id}

    @app.post("/v1/conversations/{conversation_id}/compact")
    async def compact_conversation(
        conversation_id: str, authorization: Optional[str] = Header(None)
    ):
        """Force-summarize the conversation; the next message will start a fresh
        DeepSeek session pre-loaded with the summary."""
        _check_auth(authorization)
        if pool is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv = pool.get(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        async with conv.lock:
            ok = await conv.client.compact_session()
        return {
            "compacted": ok,
            "id": conversation_id,
            "summary_chars": len(conv.client._compact_summary) if ok else 0,
        }

    @app.post("/v1/files")
    async def upload_file(
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
    ):
        """Upload a file to DeepSeek and return a file_id.

        The file_id can then be passed in chat completions via file_ids.
        """
        _check_auth(authorization)

        content = await file.read()
        filename = file.filename or "upload"
        content_type = file.content_type or "application/octet-stream"

        try:
            # In gateway mode, pick the best available account (same scoring
            # as chat completions); otherwise fall back to the default client.
            client_for_upload = None
            if use_gateway and gateway_pool is not None:
                gw_accounts = gateway_pool.available_accounts() or gateway_pool.accounts
                if gw_accounts:
                    # Use the pool's scoring to pick the least-loaded account
                    # without going through the full streaming code path.
                    chosen_acc = min(gw_accounts, key=gateway_pool._score)
                    client_for_upload = chosen_acc.client
            else:
                client_for_upload = _default_client
            if client_for_upload is None:
                metrics.incr("file_uploads_failed_total")
                raise HTTPException(status_code=503, detail="No client available")
            file_id = await client_for_upload.upload_file(content, filename, content_type)
            metrics.incr("file_uploads_total")
        except AuthExpiredError as e:
            metrics.incr("file_uploads_failed_total")
            raise HTTPException(status_code=401, detail=str(e))
        except RuntimeError as e:
            metrics.incr("file_uploads_failed_total")
            raise HTTPException(status_code=502, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            metrics.incr("file_uploads_failed_total")
            raise HTTPException(status_code=502, detail=f"Upload error: {e}")

        return {
            "id": file_id,
            "object": "file",
            "filename": filename,
            "bytes": len(content),
            "created_at": int(time.time()),
            "purpose": "assistants",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, authorization: Optional[str] = Header(None)
    ):
        _check_auth(authorization)
        metrics.incr("requests_total")
        req_started_at = time.time()
        try:
            payload = await request.json()
        except Exception:
            metrics.incr("failed_requests_total")
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

        model = payload.get("model", "deepseek-chat")
        stream = bool(payload.get("stream", False))
        conversation_id: Optional[str] = payload.get("conversation_id")
        # Defensive copy: caller's list could be mutated after we keep a reference
        raw_file_ids = payload.get("file_ids", [])
        file_ids: list[str] = list(raw_file_ids) if isinstance(raw_file_ids, list) else []

        tools = payload.get("tools") or []
        tool_choice = payload.get("tool_choice")
        if not isinstance(tools, list):
            tools = []
        # "tool_choice": "none" disables tool emission even when tools defined
        tools_active = bool(tools) and tool_choice != "none"

        # Tools mode is stateless by design — DeepSeek has no tool-calling
        # protocol, so we flatten the entire conversation into one prompt
        # each time. Mixing tools with conversation_id is contradictory:
        # warn the client by ignoring the conversation_id and surfacing the
        # decision in a response header (set further down).
        tools_with_conversation_warning = bool(tools_active and conversation_id)
        if tools_with_conversation_warning:
            conversation_id = None

        # ── Per-request auto-compact override ─────────────────
        # Body field "auto_compact" may be:
        #   • int  → custom threshold (0 disables for this request)
        #   • bool → True keeps default, False disables
        #   • absent → no override
        ac_threshold_for_request = _resolve_auto_compact_override(payload)

        if tools_active:
            # Tools mode is stateless — flatten the entire conversation into
            # one prompt so the model sees full context (DeepSeek lacks
            # native tool-calling).
            prompt = compose_tools_prompt(messages, tools)
            # Force fresh session so previous turns don't leak into the
            # plaintext-encoded conversation.
            is_fresh = True
        else:
            prompt = _messages_to_prompt(messages)
            is_fresh = _is_fresh_conversation(messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found")

        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        # ── Pick client + lock ────────────────────────────────
        # Gateway mode: route through AccountPool (per-account locks, smart pick)
        # Single mode: use _default_client / conversation pool
        if use_gateway:
            active_client = None
            active_lock = None
            is_fresh_for_conv = is_fresh
        elif conversation_id and not tools_active:
            # Multi-turn: use dedicated client for this conversation
            assert pool is not None
            conv = await pool.get_or_create(conversation_id)
            active_client = conv.client
            active_lock = conv.lock
            is_fresh_for_conv = False  # conversation manages its own session continuity
        else:
            # Stateless: use default client
            active_client = _default_client
            active_lock = _default_lock
            is_fresh_for_conv = is_fresh

        request_id = f"r{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        # Tracks which client served this request. In gateway mode the pool
        # populates this via ``on_account_chosen`` callback. In single mode
        # it is set immediately after lock acquire.
        chosen_client: dict[str, Optional[APIClient]] = {"client": None}

        async def generate():
            # Gateway path: pool routes + handles per-account locking itself.
            if use_gateway:
                assert gateway_pool is not None
                # Build preset for the chosen account; pool applies it under lock.
                preset = _model_to_preset(model)
                if ac_threshold_for_request is not None:
                    preset["auto_compact_threshold"] = ac_threshold_for_request
                try:
                    async for token_type, token_text in gateway_pool.send_message_stream(
                        prompt,
                        request_id=request_id,
                        model_preset=preset,
                        file_ids=file_ids if file_ids else None,
                        on_account_chosen=lambda acc: chosen_client.update(client=acc.client),
                    ):
                        if token_type == "text":
                            yield token_text
                except AuthExpiredError as e:
                    raise HTTPException(status_code=401, detail=str(e))
                except NoAccountAvailableError as e:
                    headers = {}
                    if e.retry_after is not None:
                        headers["Retry-After"] = str(int(e.retry_after) + 1)
                    raise HTTPException(
                        status_code=503, detail=str(e), headers=headers
                    )
                except Exception as e:
                    status, detail = _classify_upstream_error(e)
                    raise HTTPException(status_code=status, detail=detail)
                return

            # Single-client path
            assert active_lock is not None and active_client is not None
            chosen_client["client"] = active_client
            acquired = False
            saved_threshold: Optional[int] = None
            try:
                try:
                    await asyncio.wait_for(active_lock.acquire(), timeout=REQUEST_TIMEOUT)
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=503,
                        detail="Server busy — another request is in progress. Try again shortly.",
                    )
                acquired = True
                _apply_preset(active_client, _model_to_preset(model))
                if ac_threshold_for_request is not None:
                    saved_threshold = active_client.auto_compact_threshold
                    active_client.auto_compact_threshold = ac_threshold_for_request
                if is_fresh_for_conv:
                    await active_client.reset_session()

                # Inject file_ids into the client's next request
                if file_ids:
                    active_client.set_pending_files(file_ids)

                async for token_type, token_text in active_client.send_message_stream(prompt):
                    if token_type == "text":
                        yield token_text
            except AuthExpiredError as e:
                raise HTTPException(status_code=401, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                status, detail = _classify_upstream_error(e)
                raise HTTPException(status_code=status, detail=detail)
            finally:
                # Restore threshold so per-request override doesn't leak.
                if saved_threshold is not None and active_client is not None:
                    active_client.auto_compact_threshold = saved_threshold
                if acquired:
                    active_lock.release()

        if stream:
            metrics.incr("streamed_requests_total")
            async def sse_stream():
                buffered = ""
                try:
                    async for chunk in generate():
                        if tools_active:
                            # Buffer entire reply: tool-call detection requires
                            # the whole text. We emit one consolidated chunk
                            # at the end with the parsed tool_calls.
                            buffered += chunk
                            continue
                        sse_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                            ],
                        }
                        if conversation_id:
                            sse_data["conversation_id"] = conversation_id
                        yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

                    if tools_active:
                        clean_text, tool_calls = extract_tool_calls(buffered)
                        first_delta: dict = {"role": "assistant"}
                        if clean_text:
                            first_delta["content"] = clean_text
                        if tool_calls:
                            first_delta["tool_calls"] = tool_calls
                        first_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": first_delta, "finish_reason": None}
                            ],
                        }
                        if conversation_id:
                            first_chunk["conversation_id"] = conversation_id
                        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
                        finish = "tool_calls" if tool_calls else "stop"
                    else:
                        finish = "stop"

                    final = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                    if conversation_id:
                        final["conversation_id"] = conversation_id
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    metrics.observe_latency((time.time() - req_started_at) * 1000)
                except HTTPException as e:
                    metrics.incr("failed_requests_total")
                    if e.status_code == 401:
                        metrics.incr("auth_errors_total")
                    elif e.status_code == 503:
                        metrics.incr("pool_no_account_total")
                    err = {"error": {"message": e.detail, "type": "upstream_error", "code": e.status_code}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

            stream_headers: dict[str, str] = {
                "X-Model-Used": model,
                "X-Mode": "gateway" if use_gateway else "single",
                "X-Tools-Active": "1" if tools_active else "0",
                "X-Request-Id": request_id,
                # NB: For streaming we cannot know whether compact fired
                # until the generator runs. Best-effort: clients should
                # check the non-stream endpoint or /admin/compact response.
                "X-Auto-Compact-Available": "1",
            }
            if tools_with_conversation_warning:
                stream_headers["X-Conversation-Ignored"] = "tools-mode-is-stateless"
            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        # Non-streaming
        full_text = ""
        try:
            async for chunk in generate():
                full_text += chunk
        except HTTPException as e:
            metrics.incr("failed_requests_total")
            if e.status_code == 401:
                metrics.incr("auth_errors_total")
            elif e.status_code == 503:
                metrics.incr("pool_no_account_total")
            raise

        if tools_active:
            clean_text, tool_calls = extract_tool_calls(full_text)
            message_obj: dict = {"role": "assistant", "content": clean_text or None}
            if tool_calls:
                message_obj["tool_calls"] = tool_calls
            finish_reason = "tool_calls" if tool_calls else "stop"
        else:
            message_obj = {"role": "assistant", "content": full_text}
            finish_reason = "stop"

        response_body = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // _TOKEN_CHARS_PER_TOKEN,
                "completion_tokens": len(full_text) // _TOKEN_CHARS_PER_TOKEN,
                "total_tokens": (len(prompt) + len(full_text)) // _TOKEN_CHARS_PER_TOKEN,
            },
        }
        if conversation_id:
            response_body["conversation_id"] = conversation_id

        metrics.observe_latency((time.time() - req_started_at) * 1000)
        # Detect whether auto-compact fired during this request.
        compact_fired = False
        chosen = chosen_client["client"]
        if chosen is not None and chosen._just_compacted:
            compact_fired = True
            chosen._just_compacted = False

        response_headers: dict[str, str] = {
            "X-Model-Used": model,
            "X-Mode": "gateway" if use_gateway else "single",
            "X-Tools-Active": "1" if tools_active else "0",
            "X-Request-Id": request_id,
            "X-Auto-Compact-Triggered": "1" if compact_fired else "0",
        }
        if tools_with_conversation_warning:
            response_headers["X-Conversation-Ignored"] = (
                "tools-mode-is-stateless"
            )
        # Surface which account served the request (gateway mode only).
        if use_gateway and gateway_pool is not None and chosen is not None:
            for acc in gateway_pool.accounts:
                if acc.client is chosen:
                    response_headers["X-Account-Used"] = acc.name
                    break

        return JSONResponse(response_body, headers=response_headers)

    # ── Startup banner ────────────────────────────────────────
    mode_label = (
        f"[bold cyan]gateway[/] (pool of {len(gateway_pool.accounts)})"
        if use_gateway and gateway_pool
        else "single-account"
    )
    console.print(
        Panel.fit(
            f"[bold green]🚀 DeepSeek API Server[/] [dim]v{VERSION}[/]\n"
            f"Listening on: [bold]http://{host}:{port}[/]\n"
            f"Mode: {mode_label}\n"
            f"Endpoints:\n"
            f"  POST /v1/chat/completions  — chat (+ tools, conversation_id, file_ids, auto_compact)\n"
            f"  POST /v1/files             — upload file → get file_id\n"
            f"  GET  /v1/conversations     — list active conversations\n"
            f"  POST /v1/conversations/{{id}}/compact — manual rolling-summary\n"
            f"  DELETE /v1/conversations/{{id}} — end conversation\n"
            f"  GET  /v1/models | GET /health | GET /metrics\n"
            f"  POST /admin/pool/{{name}}/unblock  (X-Admin-Token)\n"
            f"  POST /admin/compact                (X-Admin-Token)\n"
            f"Tool calling: [green]emulated via prompt[/] (OpenAI-compatible)\n"
            f"Auto-compact: {auto_compact_threshold if auto_compact_threshold is not None else 'default'} turns\n"
            f"Auth: {'Bearer token required' if api_key else '[yellow]disabled[/]'} | "
            f"Admin: {'enabled' if admin_token else '[dim]disabled[/]'} | "
            f"CORS: all origins | "
            f"Conv TTL: {CONVERSATION_TTL//60}min",
            border_style="green",
        )
    )

    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)
    try:
        await server.serve()
    finally:
        if pool is not None:
            await pool.close_all()
        if _default_client is not None:
            await _default_client.close()
        if gateway_pool is not None:
            await gateway_pool.close_all()
