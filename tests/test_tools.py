"""Tests for deepseek.tools — emulated tool-calling shim."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.tools import (
    build_tools_system_prompt,
    compose_tools_prompt,
    extract_tool_calls,
    messages_to_prompt,
)


# ── build_tools_system_prompt ─────────────────────────────────

class TestBuildToolsSystemPrompt:
    def test_empty_returns_empty(self):
        assert build_tools_system_prompt([]) == ""

    def test_non_list_returns_empty(self):
        assert build_tools_system_prompt(None) == ""  # type: ignore[arg-type]
        assert build_tools_system_prompt("nope") == ""  # type: ignore[arg-type]

    def test_includes_function_name_and_description(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        prompt = build_tools_system_prompt(tools)
        assert "get_weather" in prompt
        assert "Get weather for a city" in prompt
        assert "<tool_call>" in prompt
        assert "</tool_call>" in prompt

    def test_skips_invalid_entries(self):
        tools = [
            {"type": "function", "function": {}},  # no name
            "not a dict",                          # invalid type
            {"type": "function", "function": {"name": "ok", "description": "d"}},
        ]
        prompt = build_tools_system_prompt(tools)
        assert "ok" in prompt
        # Only one valid tool means only one bullet
        assert prompt.count("- ok") == 1

    def test_unknown_type_filtered(self):
        tools = [{"type": "retrieval", "function": {"name": "x"}}]
        assert build_tools_system_prompt(tools) == ""


# ── extract_tool_calls ────────────────────────────────────────

class TestExtractToolCalls:
    def test_no_tool_calls(self):
        clean, calls = extract_tool_calls("just a normal answer")
        assert clean == "just a normal answer"
        assert calls == []

    def test_single_call(self):
        text = (
            "Let me check.\n"
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Bangkok"}}</tool_call>'
        )
        clean, calls = extract_tool_calls(text)
        assert clean == "Let me check."
        assert len(calls) == 1
        c = calls[0]
        assert c["type"] == "function"
        assert c["function"]["name"] == "get_weather"
        # arguments must be a JSON STRING per OpenAI spec
        assert isinstance(c["function"]["arguments"], str)
        assert json.loads(c["function"]["arguments"]) == {"city": "Bangkok"}
        assert c["id"].startswith("call_")

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            ' middle '
            '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
        )
        clean, calls = extract_tool_calls(text)
        assert "<tool_call>" not in clean
        assert clean == "middle"
        assert [c["function"]["name"] for c in calls] == ["a", "b"]

    def test_malformed_json_skipped(self):
        text = '<tool_call>{not json}</tool_call>'
        clean, calls = extract_tool_calls(text)
        assert calls == []
        # Malformed call still gets stripped from text
        assert "<tool_call>" not in clean

    def test_arguments_already_string(self):
        """If model emits arguments as a JSON string, keep it as-is."""
        text = '<tool_call>{"name": "x", "arguments": "{\\"k\\":1}"}</tool_call>'
        _, calls = extract_tool_calls(text)
        assert calls[0]["function"]["arguments"] == '{"k":1}'

    def test_empty_input(self):
        assert extract_tool_calls("") == ("", [])

    def test_multiline_arguments(self):
        text = (
            "<tool_call>{\n"
            '  "name": "search",\n'
            '  "arguments": {"q": "deepseek"}\n'
            "}</tool_call>"
        )
        _, calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_unique_ids_across_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        )
        _, calls = extract_tool_calls(text)
        # Even identical calls should have distinct ids (idx differs)
        assert calls[0]["id"] != calls[1]["id"]

    def test_increments_metrics_on_invalid_json(self):
        from deepseek.metrics import metrics
        before = metrics.snapshot()["tool_calls_invalid_json_total"]
        extract_tool_calls('<tool_call>{not_json}</tool_call>')
        after = metrics.snapshot()["tool_calls_invalid_json_total"]
        assert after == before + 1

    def test_increments_metrics_on_no_name(self):
        from deepseek.metrics import metrics
        before = metrics.snapshot()["tool_calls_no_name_total"]
        extract_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
        after = metrics.snapshot()["tool_calls_no_name_total"]
        assert after == before + 1

    def test_increments_metrics_on_success(self):
        from deepseek.metrics import metrics
        before = metrics.snapshot()["tool_calls_total"]
        extract_tool_calls('<tool_call>{"name": "x", "arguments": {}}</tool_call>')
        after = metrics.snapshot()["tool_calls_total"]
        assert after == before + 1


# ── messages_to_prompt ────────────────────────────────────────

class TestMessagesToPrompt:
    def test_simple_user(self):
        out = messages_to_prompt([{"role": "user", "content": "hello"}])
        assert "[User]" in out
        assert "hello" in out

    def test_includes_tool_results(self):
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city":"BKK"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 30C"},
            {"role": "user", "content": "thanks"},
        ]
        out = messages_to_prompt(msgs)
        assert "weather?" in out
        assert "<tool_call>" in out
        assert "get_weather" in out
        assert "Sunny, 30C" in out
        assert "Tool result for call_1" in out
        assert "thanks" in out

    def test_content_array_parts(self):
        """OpenAI also allows content as a list of parts (text/image)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        out = messages_to_prompt(msgs)
        assert "hi" in out

    def test_skips_non_dict_messages(self):
        out = messages_to_prompt(["bad", None, {"role": "user", "content": "ok"}])
        assert "ok" in out

    def test_assistant_with_only_tool_calls_no_content(self):
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
        ]
        out = messages_to_prompt(msgs)
        assert "<tool_call>" in out
        assert '"name": "f"' in out


# ── compose_tools_prompt ──────────────────────────────────────

class TestComposeToolsPrompt:
    def test_combines_system_and_tools_and_history(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "weather?"},
        ]
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "d"}}]
        out = compose_tools_prompt(msgs, tools)
        assert "You are helpful." in out
        assert "get_weather" in out
        assert "weather?" in out
        # System block precedes user block
        assert out.index("You are helpful.") < out.index("weather?")
        assert out.index("get_weather") < out.index("weather?")

    def test_no_system_no_tools_falls_back_to_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = compose_tools_prompt(msgs, [])
        assert "[User]" in out
        assert "hi" in out

    def test_only_system_no_user(self):
        msgs = [{"role": "system", "content": "rules"}]
        out = compose_tools_prompt(msgs, [])
        assert "rules" in out
