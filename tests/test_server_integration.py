"""Route-level regression tests for secured Chat/Responses compatibility."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import deepseek.server as server_module
from deepseek.api_key import APIKeyInfo
from deepseek.conversation import ConversationIndex
from deepseek.models import APIConfig


def _config() -> APIConfig:
    return APIConfig.from_dict(
        {
            "target_url": "https://chat.deepseek.com",
            "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
            "method": "POST",
            "headers": {},
            "body_template": {},
            "auth_token": "token",
        }
    )


async def _capture_app(monkeypatch, tmp_path, stream_impl=None):
    captured = {}
    monkeypatch.setattr(server_module.SessionManager, "load_config", lambda: _config())
    monkeypatch.setattr(
        server_module,
        "load_or_create_api_key",
        lambda _explicit=None: APIKeyInfo("secret", "test"),
    )
    monkeypatch.setattr(
        server_module,
        "ConversationIndex",
        lambda: ConversationIndex(path=tmp_path / "sessions.json"),
    )

    if stream_impl is None:
        async def stream_impl(self, _message):
            self.session_id = self.session_id or "session-test"
            self.last_message_id = "message-test"
            yield ("text", "hello from deepseek")

    monkeypatch.setattr(server_module.APIClient, "send_message_stream", stream_impl)

    import uvicorn

    class FakeServer:
        def __init__(self, config):
            captured["app"] = config.app

        async def serve(self):
            return None

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    await server_module.serve_mode()
    return captured["app"]


@pytest.mark.asyncio
async def test_v1_routes_require_api_key(monkeypatch, tmp_path):
    app = await _capture_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models",
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        assert client.get("/health").status_code == 200


@pytest.mark.asyncio
async def test_responses_api_and_previous_response_resume(monkeypatch, tmp_path):
    app = await _capture_app(monkeypatch, tmp_path)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        first = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "deepseek-chat", "input": "hello"},
        )
        assert first.status_code == 200
        body = first.json()
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "hello from deepseek"
        conversation_id = body["conversation"]["id"]

        second = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "previous_response_id": body["id"],
                "input": "continue",
            },
        )
        assert second.status_code == 200
        assert second.json()["conversation"]["id"] == conversation_id


@pytest.mark.asyncio
async def test_previous_response_replays_stateless_tool_history(monkeypatch, tmp_path):
    prompts = []

    async def fake_stream(self, message):
        prompts.append(message)
        if len(prompts) == 1:
            yield (
                "text",
                '<tool_call>{"name":"lookup","arguments":{"city":"Bangkok"}}</tool_call>',
            )
        else:
            yield ("text", "final answer")

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        first = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "input": "What is the weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        call = next(item for item in first_body["output"] if item["type"] == "function_call")

        second = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "previous_response_id": first_body["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "sunny 32C",
                    }
                ],
            },
        )
        assert second.status_code == 200
        assert len(prompts) == 2
        assert "sunny 32C" in prompts[1]
        assert "lookup" in prompts[1]


@pytest.mark.asyncio
async def test_previous_response_replays_multiple_tool_outputs(monkeypatch, tmp_path):
    prompts = []

    async def fake_stream(self, message):
        prompts.append(message)
        if len(prompts) == 1:
            yield (
                "text",
                '<tool_call>{"name":"weather","arguments":{"city":"Bangkok"}}</tool_call>'
                '<tool_call>{"name":"time","arguments":{"city":"Bangkok"}}</tool_call>',
            )
        else:
            yield ("text", "done")

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        first = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "input": "Plan my morning",
                "tools": [
                    {"type": "function", "name": "weather", "parameters": {"type": "object"}},
                    {"type": "function", "name": "time", "parameters": {"type": "object"}},
                ],
            },
        )
        calls = [item for item in first.json()["output"] if item["type"] == "function_call"]
        assert len(calls) == 2

        second = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "previous_response_id": first.json()["id"],
                "input": [
                    {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "sunny"},
                    {"type": "function_call_output", "call_id": calls[1]["call_id"], "output": "08:00"},
                ],
            },
        )

    assert second.status_code == 200
    assert "weather" in prompts[1]
    assert "time" in prompts[1]
    assert "sunny" in prompts[1]
    assert "08:00" in prompts[1]


@pytest.mark.asyncio
async def test_responses_stream_emits_created_delta_and_completed(monkeypatch, tmp_path):
    async def fake_stream(self, _message):
        yield ("text", "hello ")
        yield ("text", "world")

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={"model": "deepseek-chat", "input": "hello", "stream": True},
        ) as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert response.status_code == 200
    joined = "\n".join(events)
    assert "response.created" in joined
    assert "response.output_text.delta" in joined
    assert "hello " in joined
    assert "world" in joined
    assert "response.completed" in joined


@pytest.mark.asyncio
async def test_responses_tool_stream_buffers_split_tool_block(monkeypatch, tmp_path):
    async def fake_stream(self, _message):
        yield ("text", '<tool_call>{"name":"look')
        yield ("text", 'up","arguments":{"q":1}}</tool_call>')

    app = await _capture_app(monkeypatch, tmp_path, stream_impl=fake_stream)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "input": "lookup",
                "stream": True,
                "tools": [
                    {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
                ],
            },
        ) as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]

    joined = "\n".join(events)
    assert "response.output_text.delta" not in joined
    assert "response.output_item.added" in joined
    assert '"type": "function_call"' in joined
    assert '"name": "lookup"' in joined


@pytest.mark.asyncio
async def test_x_api_key_header_supported(monkeypatch, tmp_path):
    app = await _capture_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"x-api-key": "secret"})
        assert response.status_code == 200
