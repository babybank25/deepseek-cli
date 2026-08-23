"""OpenAI-compatible tool shim plus one-shot malformed-response recovery."""
from __future__ import annotations

import json

from ._tools_legacy import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    _TOOL_CALL_RE,
    _coerce_content,
    build_tools_system_prompt,
    compose_tools_prompt,
    extract_tool_calls,
    messages_to_prompt,
)


def tool_response_needs_recovery(text: str) -> bool:
    """Retry only empty output or text that attempted but failed tool syntax."""
    if not text.strip():
        return True
    if TOOL_CALL_OPEN not in text and TOOL_CALL_CLOSE not in text:
        return False
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (value.get("name") or value.get("function")):
            return False
    return True


def build_tool_recovery_prompt(original_prompt: str) -> str:
    """Ask for one clean replay without expanding the tool protocol surface."""
    return (
        original_prompt.rstrip()
        + "\n\n[System]\n"
        + "The previous response was empty or used malformed tool-call syntax. "
        + "Retry once. If a tool is needed, emit only the exact "
        + '<tool_call>{"name":"...","arguments":{...}}</tool_call> format. '
        + "Otherwise answer normally. Do not mention this retry instruction."
    )


__all__ = [
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "build_tools_system_prompt",
    "messages_to_prompt",
    "extract_tool_calls",
    "compose_tools_prompt",
    "tool_response_needs_recovery",
    "build_tool_recovery_prompt",
]
