"""OpenAI-compatible HTTP server backed by authenticated DeepSeek Web sessions."""
from __future__ import annotations

import asyncio
import hmac
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from ._server_legacy import (
    _apply_preset,
    _classify_upstream_error,
    _is_fresh_conversation,
    _messages_to_prompt,
    _model_to_preset,
    _resolve_auto_compact_override,
)
from .api_key import is_api_key_authorized, load_or_create_api_key
from .auth import AuthManager
from .client import APIClient
from .constants import DEFAULT_SYSTEM_PROMPT, REQUEST_TIMEOUT, VERSION
from .conversation import (
    ConversationIndex,
    ConversationState,
    UnknownPreviousResponseError,
)
from .exceptions import AuthExpiredError
from .gateway import AccountPool
from .gateway.pool import NoAccountAvailableError
from .metrics import metrics
from .models import APIConfig
from .openai_compat import (
    chat_completion_to_response,
    chat_message_to_response_output,
    response_stream_created,
    responses_request_to_chat,
)
from .protocol import probe_protocol
from .session import SessionManager
from .tools import (
    build_tool_recovery_prompt,
    compose_tools_prompt,
    extract_tool_calls,
    messages_to_prompt,
    tool_response_needs_recovery,
)

console = Console()
_TOKEN_CHARS_PER_TOKEN = 4
CONVERSATION_TTL = 30 * 60


@dataclass
class Conversation:
    id: str
    client: APIClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.time)
    model: str = "deepseek-chat"

    def touch(self) -> None:
        self.last_used = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.last_used > CONVERSATION_TTL


def _recovery_system_prompt(state: Optional[ConversationState]) -> str:
    """Replay prior canonical history only if a persisted upstream session dies."""
    if state is None or not state.history:
        return DEFAULT_SYSTEM_PROMPT
    return (
        DEFAULT_SYSTEM_PROMPT
        + "\n\n[Recovered previous conversation]\n"
        + messages_to_prompt(state.history)
    )


class ConversationPool:
    """Single-account per-conversation clients sharing one auth manager."""

    def __init__(
        self,
        config: APIConfig,
        auto_compact_threshold: Optional[int] = None,
        auth_manager: Optional[AuthManager] = None,
    ) -> None:
        self._config = config
        self._auto_compact_threshold = auto_compact_threshold
        self._auth_manager = auth_manager or AuthManager(
            config,
            persist=SessionManager.save_config,
        )
        self._conversations: dict[str, Conversation] = {}
        self._pool_lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def get_or_create(
        self,
        conversation_id: str,
        state: Optional[ConversationState] = None,
    ) -> Conversation:
        async with self._pool_lock:
            self._cleanup_expired()
            if conversation_id not in self._conversations:
                client = APIClient(self._config, auth_manager=self._auth_manager)
                client.system_prompt = _recovery_system_prompt(state)
                if self._auto_compact_threshold is not None:
                    client.auto_compact_threshold = self._auto_compact_threshold
                if state is not None and state.session_id:
                    client.session_id = state.session_id
                    client.last_message_id = state.parent_message_id
                    client.model_type = state.model_type
                    client.thinking_enabled = state.thinking_enabled
                    client.search_enabled = state.search_enabled
                    client._is_first_message = False
                    client._session_flags = (
                        client.model_type,
                        client.thinking_enabled,
                        client.search_enabled,
                    )
                self._conversations[conversation_id] = Conversation(
                    id=conversation_id,
                    client=client,
                )
            conversation = self._conversations[conversation_id]
            conversation.touch()
            return conversation

    async def delete(self, conversation_id: str) -> bool:
        async with self._pool_lock:
            conversation = self._conversations.pop(conversation_id, None)
        if conversation is None:
            return False
        async with conversation.lock:
            await conversation.client.close()
        return True

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    def __len__(self) -> int:
        return len(self._conversations)

    def list_conversations(self) -> list[dict]:
        now = time.time()
        return [
            {
                "id": conversation.id,
                "last_used": conversation.last_used,
                "idle_seconds": int(now - conversation.last_used),
            }
            for conversation in list(self._conversations.values())
            if not conversation.is_expired()
        ]

    def _cleanup_expired(self) -> None:
        expired = [
            conversation_id
            for conversation_id, conversation in self._conversations.items()
            if conversation.is_expired()
        ]
        for conversation_id in expired:
            conversation = self._conversations.pop(conversation_id)
            try:
                task = asyncio.create_task(conversation.client.close())
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)
            except RuntimeError:
                pass

    async def close_all(self) -> None:
        async with self._pool_lock:
            conversations = list(self._conversations.values())
            self._conversations.clear()
        for conversation in conversations:
            await conversation.client.close()


def _merge_prior_history(prior: list[dict], incoming: list[dict]) -> list[dict]:
    if not prior:
        return list(incoming)
    if len(incoming) >= len(prior) and incoming[: len(prior)] == prior:
        return list(incoming)
    return [*prior, *incoming]


def _payload_identifier(payload: dict, name: str) -> Optional[str]:
    value = payload.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get(name)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _usage(prompt: str, completion: str) -> dict:
    prompt_tokens = len(prompt) // _TOKEN_CHARS_PER_TOKEN
    completion_tokens = len(completion) // _TOKEN_CHARS_PER_TOKEN
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _assistant_message(text: str, tools_active: bool) -> tuple[dict, str]:
    if not tools_active:
        return {"role": "assistant", "content": text}, "stop"
    clean, calls = extract_tool_calls(text)
    message: dict = {"role": "assistant", "content": clean or None}
    if calls:
        message["tool_calls"] = calls
    return message, "tool_calls" if calls else "stop"


async def serve_mode(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = None,
    use_gateway: bool = False,
    admin_token: Optional[str] = None,
    auto_compact_threshold: Optional[int] = None,
) -> None:
    """Start the secured OpenAI-compatible DeepSeek Web gateway."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
        import uvicorn
    except ImportError:
        console.print(
            "[red]FastAPI / uvicorn not installed. Run: pip install fastapi uvicorn[/]"
        )
        sys.exit(1)

    config = SessionManager.load_config()
    if not config and not use_gateway:
        console.print("[red]No saved config found. Run --discover/--repair first.[/]")
        sys.exit(1)

    gateway_pool: Optional[AccountPool] = None
    if use_gateway:
        gateway_pool = AccountPool()
        loaded = gateway_pool.load_all()
        if loaded == 0:
            console.print("[red]Gateway mode has no configured accounts.[/]")
            sys.exit(1)
        console.print(f"[green]Gateway: loaded {loaded} account(s)[/]")
        if auto_compact_threshold is not None:
            for account in gateway_pool.accounts:
                account.client.auto_compact_threshold = auto_compact_threshold

    shared_auth = (
        AuthManager(config, persist=SessionManager.save_config)
        if config is not None
        else None
    )
    pool = (
        ConversationPool(
            config,
            auto_compact_threshold=auto_compact_threshold,
            auth_manager=shared_auth,
        )
        if config is not None
        else None
    )
    default_client = (
        APIClient(config, auth_manager=shared_auth)
        if config is not None and not use_gateway and shared_auth is not None
        else None
    )
    if default_client is not None:
        default_client.system_prompt = DEFAULT_SYSTEM_PROMPT
        if auto_compact_threshold is not None:
            default_client.auto_compact_threshold = auto_compact_threshold
    default_lock = asyncio.Lock()

    key_info = load_or_create_api_key(api_key)
    conversation_index = ConversationIndex()

    app = FastAPI(title="DeepSeek OpenAI-Compatible API", version=VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def require_api_key(
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    ) -> None:
        if not is_api_key_authorized(authorization, x_api_key, key_info.key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def check_admin(token: Optional[str]) -> None:
        if not admin_token:
            raise HTTPException(status_code=404, detail="Admin endpoints disabled")
        if not token or not hmac.compare_digest(token, admin_token):
            raise HTTPException(status_code=401, detail="Invalid admin token")

    @app.get("/health")
    async def health():
        body = {
            "status": "ok",
            "version": VERSION,
            "mode": "gateway" if use_gateway else "single",
        }
        if gateway_pool is not None:
            body["pool"] = gateway_pool.status()
        else:
            body["active_conversations"] = len(pool) if pool else 0
        return body

    @app.get("/metrics")
    async def metrics_endpoint():
        return PlainTextResponse(
            metrics.to_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/admin/probe")
    async def admin_probe(x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        if gateway_pool is not None:
            results = []
            for account in gateway_pool.accounts:
                capabilities = await probe_protocol(account.config)
                if not capabilities.auth_ok or not capabilities.pow_ok:
                    metrics.incr("protocol_probe_fail_total")
                results.append({"account": account.name, **capabilities.__dict__})
            return {"results": results}
        if config is None:
            raise HTTPException(status_code=503, detail="No config")
        capabilities = await probe_protocol(config)
        if not capabilities.auth_ok or not capabilities.pow_ok:
            metrics.incr("protocol_probe_fail_total")
        return capabilities.__dict__

    @app.get("/admin/pool")
    async def admin_pool_status(x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        if gateway_pool is None:
            raise HTTPException(status_code=404, detail="Gateway mode is not enabled")
        return gateway_pool.status()

    @app.post("/admin/pool/{name}/unblock")
    async def admin_pool_unblock(
        name: str,
        x_admin_token: Optional[str] = Header(None),
    ):
        check_admin(x_admin_token)
        if gateway_pool is None:
            raise HTTPException(status_code=404, detail="Gateway mode is not enabled")
        for account in gateway_pool.accounts:
            if account.name == name:
                account.exhausted_until = 0.0
                account.consecutive_quota_hits = 0
                account.last_error = None
                gateway_pool.flush_stats()
                return {"unblocked": name}
        raise HTTPException(status_code=404, detail=f"Account '{name}' not found")

    @app.post("/admin/compact")
    async def admin_compact_all(x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        results: list[dict] = []
        if default_client is not None:
            async with default_lock:
                results.append(
                    {"target": "default", "compacted": await default_client.compact_session()}
                )
        if gateway_pool is not None:
            for account in gateway_pool.accounts:
                async with account.lock:
                    results.append(
                        {
                            "target": account.name,
                            "compacted": await account.client.compact_session(),
                        }
                    )
        return {"results": results}

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": "deepseek-chat", "object": "model", "created": now, "owned_by": "deepseek"},
                {"id": "deepseek-fast", "object": "model", "created": now, "owned_by": "deepseek"},
                {"id": "deepseek-reasoner", "object": "model", "created": now, "owned_by": "deepseek"},
            ],
        }

    @app.get("/v1/conversations", dependencies=[Depends(require_api_key)])
    async def list_conversations():
        data = (
            gateway_pool.list_conversations()
            if gateway_pool is not None
            else conversation_index.list_states()
        )
        return {"object": "list", "data": data, "ttl_seconds": CONVERSATION_TTL}

    @app.delete(
        "/v1/conversations/{conversation_id}",
        dependencies=[Depends(require_api_key)],
    )
    async def delete_conversation(conversation_id: str):
        deleted = conversation_index.delete(conversation_id)
        if gateway_pool is not None:
            deleted = await gateway_pool.delete_conversation(conversation_id) or deleted
        elif pool is not None:
            deleted = await pool.delete(conversation_id) or deleted
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "id": conversation_id}

    @app.post(
        "/v1/conversations/{conversation_id}/compact",
        dependencies=[Depends(require_api_key)],
    )
    async def compact_conversation(conversation_id: str):
        if gateway_pool is not None:
            compacted = await gateway_pool.compact_conversation(conversation_id)
        elif pool is not None:
            state = conversation_index.state(conversation_id)
            conversation = pool.get(conversation_id)
            if conversation is None and state is not None:
                conversation = await pool.get_or_create(conversation_id, state)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            async with conversation.lock:
                compacted = await conversation.client.compact_session()
        else:
            compacted = False
        if not compacted:
            raise HTTPException(status_code=409, detail="Conversation could not be compacted")
        return {"compacted": True, "id": conversation_id}

    @app.post("/v1/files", dependencies=[Depends(require_api_key)])
    async def upload_file(file: UploadFile = File(...)):
        content = await file.read()
        filename = file.filename or "upload"
        content_type = file.content_type or "application/octet-stream"
        try:
            if gateway_pool is not None:
                file_id, account_name = await gateway_pool.upload_file(
                    content,
                    filename,
                    content_type,
                )
            else:
                if default_client is None:
                    raise HTTPException(status_code=503, detail="No client available")
                async with default_lock:
                    file_id = await default_client.upload_file(
                        content,
                        filename,
                        content_type,
                    )
                account_name = None
            metrics.incr("file_uploads_total")
        except AuthExpiredError as error:
            metrics.incr("file_uploads_failed_total")
            raise HTTPException(status_code=401, detail=str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            metrics.incr("file_uploads_failed_total")
            raise HTTPException(status_code=502, detail=f"Upload error: {error}") from error
        body = {
            "id": file_id,
            "object": "file",
            "filename": filename,
            "bytes": len(content),
            "created_at": int(time.time()),
            "purpose": "assistants",
        }
        if account_name:
            body["account"] = account_name
        return body

    async def handle_chat_payload(payload: dict, *, style: str):
        metrics.incr("requests_total")
        started = time.time()
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")
        messages = [message for message in messages if isinstance(message, dict)]
        if not messages:
            raise HTTPException(status_code=400, detail="No valid messages")

        model = str(payload.get("model") or "deepseek-chat")
        stream = bool(payload.get("stream", False))
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        tools_active = bool(tools) and payload.get("tool_choice") != "none"
        file_ids = (
            list(payload.get("file_ids"))
            if isinstance(payload.get("file_ids"), list)
            else []
        )

        try:
            resolution = conversation_index.resolve(
                messages=messages,
                conversation_id=_payload_identifier(payload, "conversation_id"),
                chat_session_id=_payload_identifier(payload, "chat_session_id"),
                previous_response_id=_payload_identifier(payload, "previous_response_id"),
            )
        except UnknownPreviousResponseError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        conversation_id = resolution.conversation_id
        prior_state = conversation_index.state(conversation_id)
        prior_history = prior_state.history if prior_state is not None else []
        effective_messages = _merge_prior_history(prior_history, messages)

        if tools_active:
            prompt = compose_tools_prompt(effective_messages, tools)
        elif resolution.is_new and len(effective_messages) > 1:
            prompt = messages_to_prompt(effective_messages)
        else:
            prompt = _messages_to_prompt(effective_messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found")

        auto_compact = _resolve_auto_compact_override(payload)
        preset = _model_to_preset(model)
        if auto_compact is not None:
            preset["auto_compact_threshold"] = auto_compact

        request_id = f"r{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())
        chosen: dict[str, object] = {"client": None, "account": None}
        transport_conversation = None if tools_active else conversation_id

        def remember_choice(account, client) -> None:
            chosen["account"] = account
            chosen["client"] = client
            client.system_prompt = (
                DEFAULT_SYSTEM_PROMPT
                if tools_active
                else _recovery_system_prompt(prior_state)
            )

        async def generate_once(
            prompt_text: str,
            *,
            reset_session: bool = False,
        ):
            if gateway_pool is not None:
                try:
                    async for token_type, token_text in gateway_pool.send_message_stream(
                        prompt_text,
                        request_id=request_id,
                        conversation_id=transport_conversation,
                        model_preset=preset,
                        file_ids=file_ids or None,
                        reset_session=reset_session or tools_active,
                        on_client_chosen=remember_choice,
                    ):
                        if token_type == "text":
                            yield token_text
                except NoAccountAvailableError as error:
                    headers = {}
                    if error.retry_after is not None:
                        headers["Retry-After"] = str(int(error.retry_after) + 1)
                    raise HTTPException(status_code=503, detail=str(error), headers=headers) from error
                except AuthExpiredError as error:
                    raise HTTPException(status_code=401, detail=str(error)) from error
                except Exception as error:
                    status, detail = _classify_upstream_error(error)
                    raise HTTPException(status_code=status, detail=detail) from error
                return

            if tools_active:
                client = default_client
                lock = default_lock
                state = None
            else:
                if pool is None:
                    raise HTTPException(status_code=503, detail="No client available")
                conversation = await pool.get_or_create(conversation_id, prior_state)
                client = conversation.client
                lock = conversation.lock
                state = prior_state
            if client is None:
                raise HTTPException(status_code=503, detail="No client available")
            chosen["client"] = client
            acquired = False
            previous_threshold: Optional[int] = None
            try:
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=REQUEST_TIMEOUT)
                except asyncio.TimeoutError as error:
                    raise HTTPException(status_code=503, detail="Server busy") from error
                acquired = True
                _apply_preset(client, preset)
                if auto_compact is not None:
                    previous_threshold = client.auto_compact_threshold
                    client.auto_compact_threshold = auto_compact
                if reset_session or tools_active:
                    await client.reset_session()
                client.system_prompt = (
                    DEFAULT_SYSTEM_PROMPT
                    if tools_active
                    else _recovery_system_prompt(state)
                )
                if file_ids:
                    client.set_pending_files(file_ids)
                async for token_type, token_text in client.send_message_stream(prompt_text):
                    if token_type == "text":
                        yield token_text
            except AuthExpiredError as error:
                raise HTTPException(status_code=401, detail=str(error)) from error
            except HTTPException:
                raise
            except Exception as error:
                status, detail = _classify_upstream_error(error)
                raise HTTPException(status_code=status, detail=detail) from error
            finally:
                if previous_threshold is not None:
                    client.auto_compact_threshold = previous_threshold
                if acquired:
                    lock.release()

        async def collect_with_tool_recovery() -> str:
            text = ""
            async for chunk in generate_once(prompt):
                text += chunk
            if tools_active and tool_response_needs_recovery(text):
                metrics.incr("tool_recovery_total")
                retry_text = ""
                async for chunk in generate_once(
                    build_tool_recovery_prompt(prompt),
                    reset_session=True,
                ):
                    retry_text += chunk
                text = retry_text
                if not tool_response_needs_recovery(text):
                    metrics.incr("tool_recovery_success_total")
            return text

        def remember_response(assistant: dict, public_response_id: str) -> None:
            client = chosen.get("client")
            session_id = None
            parent_id = None
            model_type = preset["model_type"]
            thinking_enabled = bool(preset["thinking_enabled"])
            search_enabled = bool(preset["search_enabled"])
            if isinstance(client, APIClient) and not tools_active:
                session_id = client.session_id
                parent_id = client.last_message_id
                model_type = client.model_type
                thinking_enabled = client.thinking_enabled
                search_enabled = client.search_enabled
            conversation_index.remember(
                conversation_id=conversation_id,
                response_id=public_response_id,
                request_messages=effective_messages,
                assistant_message=assistant,
                session_id=session_id,
                parent_message_id=parent_id,
                model_type=model_type,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            )

        if not stream:
            try:
                full_text = await collect_with_tool_recovery()
            except HTTPException as error:
                metrics.incr("failed_requests_total")
                if error.status_code == 401:
                    metrics.incr("auth_errors_total")
                elif error.status_code == 503:
                    metrics.incr("pool_no_account_total")
                raise
            assistant, finish_reason = _assistant_message(full_text, tools_active)
            chat_body = {
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": assistant,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _usage(prompt, full_text),
                "conversation_id": conversation_id,
            }
            public_id = response_id if style == "responses" else chat_id
            remember_response(assistant, public_id)
            metrics.observe_latency((time.time() - started) * 1000)
            headers = {
                "X-Request-Id": request_id,
                "X-Mode": "gateway" if gateway_pool is not None else "single",
                "X-Conversation-Id": conversation_id,
            }
            account = chosen.get("account")
            if account is not None and hasattr(account, "name"):
                headers["X-Account-Used"] = str(account.name)
            if style == "responses":
                body = chat_completion_to_response(
                    chat_body,
                    response_id=response_id,
                    conversation_id=conversation_id,
                )
                body["previous_response_id"] = _payload_identifier(payload, "previous_response_id")
                return JSONResponse(body, headers=headers)
            return JSONResponse(chat_body, headers=headers)

        metrics.incr("streamed_requests_total")
        if style == "responses":
            async def responses_stream():
                full_text = ""
                assistant: dict = {"role": "assistant", "content": ""}
                try:
                    yield "data: " + json.dumps(
                        response_stream_created(response_id, model, conversation_id),
                        ensure_ascii=False,
                    ) + "\n\n"
                    if tools_active:
                        full_text = await collect_with_tool_recovery()
                        assistant, _ = _assistant_message(full_text, True)
                        for index, item in enumerate(chat_message_to_response_output(assistant)):
                            yield "data: " + json.dumps(
                                {
                                    "type": "response.output_item.added",
                                    "output_index": index,
                                    "item": item,
                                },
                                ensure_ascii=False,
                            ) + "\n\n"
                            yield "data: " + json.dumps(
                                {
                                    "type": "response.output_item.done",
                                    "output_index": index,
                                    "item": item,
                                },
                                ensure_ascii=False,
                            ) + "\n\n"
                    else:
                        item_id = f"msg_{uuid.uuid4().hex}"
                        yield "data: " + json.dumps(
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "id": item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                },
                            }
                        ) + "\n\n"
                        async for chunk in generate_once(prompt):
                            full_text += chunk
                            yield "data: " + json.dumps(
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": chunk,
                                },
                                ensure_ascii=False,
                            ) + "\n\n"
                        assistant = {"role": "assistant", "content": full_text}
                        yield "data: " + json.dumps(
                            {
                                "type": "response.output_text.done",
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": full_text,
                            },
                            ensure_ascii=False,
                        ) + "\n\n"
                    chat_body = {
                        "id": chat_id,
                        "created": created,
                        "model": model,
                        "choices": [{"message": assistant}],
                        "usage": _usage(prompt, full_text),
                    }
                    completed = chat_completion_to_response(
                        chat_body,
                        response_id=response_id,
                        conversation_id=conversation_id,
                    )
                    completed["previous_response_id"] = _payload_identifier(
                        payload, "previous_response_id"
                    )
                    remember_response(assistant, response_id)
                    metrics.observe_latency((time.time() - started) * 1000)
                    yield "data: " + json.dumps(
                        {"type": "response.completed", "response": completed},
                        ensure_ascii=False,
                    ) + "\n\n"
                except HTTPException as error:
                    metrics.incr("failed_requests_total")
                    yield "data: " + json.dumps(
                        {
                            "type": "error",
                            "code": error.status_code,
                            "message": str(error.detail),
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
            return StreamingResponse(responses_stream(), media_type="text/event-stream")

        async def chat_stream():
            full_text = ""
            try:
                if tools_active:
                    full_text = await collect_with_tool_recovery()
                    assistant, finish_reason = _assistant_message(full_text, True)
                    delta = {"role": "assistant", **assistant}
                    delta.pop("role", None)
                    yield "data: " + json.dumps(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                            "conversation_id": conversation_id,
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
                else:
                    finish_reason = "stop"
                    async for chunk in generate_once(prompt):
                        full_text += chunk
                        yield "data: " + json.dumps(
                            {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk},
                                        "finish_reason": None,
                                    }
                                ],
                                "conversation_id": conversation_id,
                            },
                            ensure_ascii=False,
                        ) + "\n\n"
                    assistant = {"role": "assistant", "content": full_text}
                if tools_active:
                    assistant, finish_reason = _assistant_message(full_text, True)
                remember_response(assistant, chat_id)
                metrics.observe_latency((time.time() - started) * 1000)
                yield "data: " + json.dumps(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                        "conversation_id": conversation_id,
                    },
                    ensure_ascii=False,
                ) + "\n\n"
                yield "data: [DONE]\n\n"
            except HTTPException as error:
                metrics.incr("failed_requests_total")
                yield "data: " + json.dumps(
                    {
                        "error": {
                            "message": str(error.detail),
                            "type": "upstream_error",
                            "code": error.status_code,
                        }
                    },
                    ensure_ascii=False,
                ) + "\n\n"
        return StreamingResponse(chat_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        return await handle_chat_payload(payload, style="chat")

    @app.post("/v1/responses", dependencies=[Depends(require_api_key)])
    async def responses(request: Request):
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        chat_payload = responses_request_to_chat(payload)
        return await handle_chat_payload(chat_payload, style="responses")

    key_location = str(key_info.path) if key_info.path else key_info.source
    mode_label = (
        f"gateway ({len(gateway_pool.accounts)} accounts)"
        if gateway_pool is not None
        else "single-account"
    )
    console.print(
        Panel.fit(
            f"[bold green]DeepSeek API Server[/] [dim]v{VERSION}[/]\n"
            f"Listening on: [bold]http://{host}:{port}[/]\n"
            f"Mode: {mode_label}\n"
            f"API auth: required ({key_info.source}; {key_location})\n"
            f"POST /v1/chat/completions\n"
            f"POST /v1/responses\n"
            f"POST /v1/files\n"
            f"GET /v1/models | GET /v1/conversations\n"
            f"GET /health | GET /metrics\n"
            f"GET /admin/probe (X-Admin-Token)",
            border_style="green",
        )
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    try:
        await server.serve()
    finally:
        if pool is not None:
            await pool.close_all()
        if default_client is not None:
            await default_client.close()
        if gateway_pool is not None:
            await gateway_pool.close_all()


__all__ = [
    "Conversation",
    "ConversationPool",
    "serve_mode",
    "_apply_preset",
    "_classify_upstream_error",
    "_is_fresh_conversation",
    "_messages_to_prompt",
    "_model_to_preset",
    "_resolve_auto_compact_override",
    "_merge_prior_history",
]
