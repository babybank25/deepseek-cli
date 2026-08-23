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


async def _capture_app(monkeypatch, tmp_path):
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

    async def fake_stream(self, _message):
        self.session_id = self.session_id or "session-test"
        self.last_message_id = "message-test"
        yield ("text", "hello from deepseek")

    monkeypatch.setattr(server_module.APIClient, "send_message_stream", fake_stream)

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
async def test_x_api_key_header_supported(monkeypatch, tmp_path):
    app = await _capture_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"x-api-key": "secret"})
        assert response.status_code == 200
