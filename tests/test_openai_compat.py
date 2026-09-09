"""Tests for Responses API normalization and tool recovery policy."""
from deepseek.openai_compat import (
    chat_completion_to_response,
    responses_request_to_chat,
)
from deepseek.tools import build_tool_recovery_prompt, tool_response_needs_recovery


def test_responses_string_input_becomes_user_message():
    chat = responses_request_to_chat(
        {"model": "deepseek-chat", "instructions": "be concise", "input": "hello"}
    )
    assert chat["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    ]


def test_responses_function_tool_shape_is_normalized():
    chat = responses_request_to_chat(
        {
            "input": "weather",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "description": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
        }
    )
    assert chat["tools"][0]["function"]["name"] == "weather"


def test_chat_tool_call_maps_to_response_function_call():
    body = {
        "id": "chat-1",
        "created": 1,
        "model": "deepseek-chat",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"q\":1}"},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    response = chat_completion_to_response(
        body,
        response_id="resp-1",
        conversation_id="conv-1",
    )
    call = response["output"][0]
    assert call["type"] == "function_call"
    assert call["call_id"] == "call-1"
    assert response["usage"]["total_tokens"] == 3


def test_tool_recovery_only_for_empty_or_malformed_attempt():
    assert tool_response_needs_recovery("") is True
    assert tool_response_needs_recovery("normal answer") is False
    assert tool_response_needs_recovery("<tool_call>{bad}</tool_call>") is True
    assert (
        tool_response_needs_recovery(
            '<tool_call>{"name":"lookup","arguments":{"q":1}}</tool_call>'
        )
        is False
    )


def test_tool_recovery_prompt_keeps_original_request():
    prompt = build_tool_recovery_prompt("ORIGINAL")
    assert prompt.startswith("ORIGINAL")
    assert "Retry once" in prompt


def test_responses_developer_message_normalizes_to_system():
    chat = responses_request_to_chat(
        {
            "input": [
                {"type": "message", "role": "developer", "content": "follow project rules"},
                {"type": "message", "role": "user", "content": "hello"},
            ]
        }
    )
    assert chat["messages"][0] == {"role": "system", "content": "follow project rules"}
    assert chat["messages"][1] == {"role": "user", "content": "hello"}


def test_responses_function_call_output_preserves_call_id_and_content():
    chat = responses_request_to_chat(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-weather",
                    "output": "sunny 32C",
                }
            ]
        }
    )
    assert chat["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny 32C",
        }
    ]
