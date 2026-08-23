"""Pure OpenAI Chat Completions ↔ Responses API compatibility helpers."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            value = part.get("text")
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(part.get("content"), str):
                parts.append(part["content"])
        return "\n".join(parts)
    return str(content)


def _normalize_responses_tool(tool: object) -> object:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return tool
    if isinstance(tool.get("function"), dict):
        return tool
    name = tool.get("name")
    if not name:
        return tool
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {},
        },
    }


def responses_request_to_chat(payload: dict) -> dict:
    """Normalize the supported Responses API request surface to chat messages."""
    messages: list[dict] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    raw_input = payload.get("input", "")
    items = raw_input if isinstance(raw_input, list) else [raw_input]
    for item in items:
        if isinstance(item, str):
            if item:
                messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": _content_text(item.get("output")),
                }
            )
        elif item_type == "function_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
        elif role in {"system", "developer", "user", "assistant", "tool"} or item_type == "message":
            resolved_role = "system" if role == "developer" else (role or "user")
            message = {
                "role": resolved_role,
                "content": _content_text(item.get("content")),
            }
            if item.get("tool_call_id"):
                message["tool_call_id"] = item["tool_call_id"]
            messages.append(message)

    tools = payload.get("tools")
    normalized_tools = (
        [_normalize_responses_tool(tool) for tool in tools]
        if isinstance(tools, list)
        else []
    )
    result = {
        "model": payload.get("model", "deepseek-chat"),
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
        "tools": normalized_tools,
        "tool_choice": payload.get("tool_choice"),
        "previous_response_id": payload.get("previous_response_id"),
        "conversation_id": payload.get("conversation_id"),
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }
    return result


def chat_message_to_response_output(message: dict) -> list[dict]:
    output: list[dict] = []
    content = message.get("content")
    if content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": str(content),
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function["name"],
                "arguments": arguments,
            }
        )
    return output


def chat_completion_to_response(
    chat_body: dict,
    *,
    response_id: str,
    conversation_id: str,
) -> dict:
    choices = chat_body.get("choices") or []
    message = (
        choices[0].get("message", {})
        if choices and isinstance(choices[0], dict)
        else {}
    )
    usage = chat_body.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(chat_body.get("created") or time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": chat_body.get("model", "deepseek-chat"),
        "output": chat_message_to_response_output(message),
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "conversation": {"id": conversation_id},
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }


def response_stream_created(response_id: str, model: str, conversation_id: str) -> dict:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
            "conversation": {"id": conversation_id},
        },
    }
