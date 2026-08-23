"""
OpenAI-compatible tool/function-calling shim for DeepSeek web.

DeepSeek's web API has no native function-calling protocol, so we emulate
it via prompt injection:

  1. ``build_tools_system_prompt`` describes the available tools and the
     output format the model must use: ``<tool_call>{...}</tool_call>``.
  2. ``messages_to_prompt`` flattens an OpenAI ``messages`` list — including
     ``role: "tool"`` results — into a plaintext conversation the model can
     read.
  3. ``extract_tool_calls`` scans the model's reply, lifts every tool-call
     block, and returns OpenAI-formatted ``tool_calls`` plus the cleaned
     remaining text.

The resulting wire-format is compatible with OpenAI's
``/v1/chat/completions`` tool-calling clients (Claude Code's OpenAI-mode,
LangChain ``ChatOpenAI`` with tools, LiteLLM proxy, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# Match `<tool_call>{...}</tool_call>` blocks, allowing whitespace and
# newlines inside. Greedy `{.*?}` keeps each call self-contained.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


# ── System-prompt builder ─────────────────────────────────────

def build_tools_system_prompt(tools: list[dict]) -> str:
    """Return a system-prompt fragment teaching the model the tool-call format.

    Returns an empty string if ``tools`` is empty or contains no valid
    function definitions.
    """
    if not isinstance(tools, list):
        return ""
    descs: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") not in (None, "function"):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or ""
        if not name:
            continue
        desc = fn.get("description") or ""
        params = fn.get("parameters") or {}
        try:
            params_json = json.dumps(params, ensure_ascii=False)
        except (TypeError, ValueError):
            params_json = "{}"
        descs.append(f"- {name}: {desc}\n  parameters: {params_json}")
    if not descs:
        return ""
    tools_block = "\n".join(descs)
    return (
        "You have access to the following tools.\n"
        f"{tools_block}\n\n"
        "When you decide to use a tool, emit a tool call in this EXACT format "
        "and nothing else for that call:\n"
        '<tool_call>{"name": "<function_name>", "arguments": {<json args>}}</tool_call>\n'
        "Rules:\n"
        "- Use raw JSON for arguments (no string-escaping the object).\n"
        "- Emit one <tool_call>...</tool_call> per call. Multiple calls allowed.\n"
        "- Don't wrap tool calls in markdown fences.\n"
        "- Only call tools when needed; otherwise answer normally.\n"
        "- After a tool result is provided, continue with the final answer."
    )


# ── Message → prompt ──────────────────────────────────────────

def _coerce_content(content: Any) -> str:
    """Normalize OpenAI message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "content" in p:
                    parts.append(str(p["content"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def messages_to_prompt(messages: list[dict]) -> str:
    """Flatten an OpenAI messages list — including tool results — to plaintext.

    System messages are emitted as ``[System]`` blocks. Assistant messages
    that contain ``tool_calls`` are rendered using the same
    ``<tool_call>...</tool_call>`` syntax the model will produce, so the
    conversation stays self-consistent.
    """
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = _coerce_content(m.get("content"))

        if role == "system":
            if content:
                parts.append(f"[System]\n{content}")
        elif role == "user":
            if content:
                parts.append(f"[User]\n{content}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            rendered = []
            if content:
                rendered.append(content)
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    # Already a JSON string per OpenAI spec
                    args_json = args
                else:
                    try:
                        args_json = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_json = "{}"
                rendered.append(
                    f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'
                )
            if rendered:
                parts.append("[Assistant]\n" + "\n".join(rendered))
        elif role == "tool":
            tcid = m.get("tool_call_id", "") or ""
            label = f"[Tool result for {tcid}]" if tcid else "[Tool result]"
            parts.append(f"{label}\n{content}")
    return "\n\n".join(parts)


# ── Tool-call extractor ───────────────────────────────────────

def extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse ``<tool_call>`` blocks from ``text``.

    Returns ``(clean_text, tool_calls)``:

    * ``clean_text``  — ``text`` with every ``<tool_call>...</tool_call>``
      block removed and surrounding whitespace stripped.
    * ``tool_calls``  — list of OpenAI-format dicts:
        ``{"id": "call_<n>", "type": "function",
           "function": {"name": "...", "arguments": "<json string>"}}``
    """
    if not text:
        return "", []
    calls: list[dict] = []
    failures: list[str] = []
    for idx, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        raw = m.group(1)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            failures.append("invalid_json")
            logger.warning(
                "tool_call: invalid JSON dropped (%s): %s", e, raw[:120]
            )
            try:
                from .metrics import metrics
                metrics.incr("tool_calls_invalid_json_total")
            except Exception:
                pass
            continue
        if not isinstance(data, dict):
            failures.append("not_object")
            continue
        # Allow common synonyms ("function" / "args") just in case.
        name = data.get("name") or data.get("function") or ""
        if not name:
            failures.append("no_name")
            logger.warning("tool_call: missing 'name' field, dropping: %s", raw[:120])
            try:
                from .metrics import metrics
                metrics.incr("tool_calls_no_name_total")
            except Exception:
                pass
            continue
        args = data.get("arguments")
        if args is None:
            args = data.get("args", {})
        # OpenAI spec: arguments is a JSON STRING (not a dict).
        if isinstance(args, str):
            arg_str = args
        else:
            try:
                arg_str = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                arg_str = "{}"
        calls.append(
            {
                "id": f"call_{idx}_{abs(hash(name + arg_str)) % 10**9:09d}",
                "type": "function",
                "function": {"name": str(name), "arguments": arg_str},
            }
        )
    if calls:
        try:
            from .metrics import metrics
            metrics.incr("tool_calls_total", len(calls))
        except Exception:
            pass
    clean_text = _TOOL_CALL_RE.sub("", text).strip()
    return clean_text, calls


# ── Compose final prompt for tools mode ──────────────────────

def compose_tools_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Build the full plain-text prompt used for tools-mode requests.

    Combines (in order): user system messages, the tool instruction prompt,
    and the rest of the conversation.
    """
    base_system = "\n\n".join(
        _coerce_content(m.get("content"))
        for m in messages
        if isinstance(m, dict)
        and m.get("role") == "system"
        and _coerce_content(m.get("content"))
    )
    tool_system = build_tools_system_prompt(tools)
    rest = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") != "system"
    ]
    body = messages_to_prompt(rest)

    system_block = "\n\n".join(s for s in (base_system, tool_system) if s)
    if system_block and body:
        return f"[System]\n{system_block}\n\n{body}"
    return system_block or body
