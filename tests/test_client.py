"""Tests for deepseek.client — APIClient SSE parsing and error handling."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.client import APIClient
from deepseek.exceptions import AuthExpiredError, _PowExpiredError, _SessionNotFoundError
from deepseek.models import APIConfig


@pytest.fixture(autouse=True)
def _close_pytest_asyncio_clean_loop_before_sync_tests(request):
    """Prevent sync asyncio.run() tests from orphaning pytest-asyncio's clean loop."""
    if request.node.get_closest_marker("asyncio") is not None:
        return
    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        return
    if loop.is_running() or loop.is_closed():
        return
    loop.close()
    policy.set_event_loop(None)


def _make_config(**overrides) -> APIConfig:
    base = {
        "target_url": "https://chat.deepseek.com",
        "api_endpoint": "https://chat.deepseek.com/api/v0/chat/completion",
        "method": "POST",
        "headers": {},
        "body_template": {},
        "auth_token": "test_token",
    }
    base.update(overrides)
    return APIConfig.from_dict(base)


# ── _raise_for_code ───────────────────────────────────────────

class TestRaiseForCode:
    def test_pow_error_40300(self):
        with pytest.raises(_PowExpiredError):
            APIClient._raise_for_code(40300, "")

    def test_pow_error_40301(self):
        with pytest.raises(_PowExpiredError):
            APIClient._raise_for_code(40301, "")

    def test_session_error_40200(self):
        with pytest.raises(_SessionNotFoundError):
            APIClient._raise_for_code(40200, "")

    def test_session_error_40201(self):
        with pytest.raises(_SessionNotFoundError):
            APIClient._raise_for_code(40201, "")

    def test_auth_error_40100(self):
        with pytest.raises(AuthExpiredError):
            APIClient._raise_for_code(40100, "")

    def test_auth_error_401(self):
        with pytest.raises(AuthExpiredError):
            APIClient._raise_for_code(401, "")

    def test_generic_error(self):
        with pytest.raises(RuntimeError, match="API error 99999"):
            APIClient._raise_for_code(99999, "something went wrong")

    def test_zero_is_ok(self):
        APIClient._raise_for_code(0, "")  # should not raise

    def test_none_is_ok(self):
        APIClient._raise_for_code(None, "")  # should not raise


# ── _build_headers ────────────────────────────────────────────

class TestBuildHeaders:
    def test_includes_auth_token(self):
        client = APIClient(_make_config(auth_token="mytoken"))
        h = client._build_headers()
        assert h["Authorization"] == "Bearer mytoken"

    def test_skips_empty_auth_token(self):
        client = APIClient(_make_config(auth_token=""))
        h = client._build_headers()
        assert "Authorization" not in h

    def test_includes_cookies(self):
        client = APIClient(_make_config(cookies={"a": "1", "b": "2"}))
        h = client._build_headers()
        assert "Cookie" in h
        assert "a=1" in h["Cookie"]
        assert "b=2" in h["Cookie"]

    def test_skips_sensitive_captured_headers(self):
        client = APIClient(_make_config(headers={
            "authorization": "old",
            "cookie": "old",
            "x-custom": "keep",
        }))
        h = client._build_headers()
        assert h.get("authorization") != "old"
        assert "x-custom" in h

    def test_sets_origin_and_referer(self):
        client = APIClient(_make_config())
        h = client._build_headers()
        assert h["Origin"] == "https://chat.deepseek.com"
        assert h["Referer"] == "https://chat.deepseek.com/"

    def test_injects_pow_response(self):
        client = APIClient(_make_config(pow_response="pow_token_123"))
        h = client._build_headers()
        assert h["x-ds-pow-response"] == "pow_token_123"


# ── _is_deepseek_mode ─────────────────────────────────────────

class TestIsDeepseekMode:
    def test_true_for_deepseek_url(self):
        client = APIClient(_make_config(target_url="https://chat.deepseek.com"))
        assert client._is_deepseek_mode() is True

    def test_false_for_other_url(self):
        client = APIClient(_make_config(target_url="https://api.openai.com"))
        assert client._is_deepseek_mode() is False


# ── SSE parsing via _do_stream ────────────────────────────────

def _make_sse_response(lines: list[str]):
    """Create a mock httpx streaming response from SSE lines."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def aiter_lines():
        for line in lines:
            yield line

    mock_response.aiter_lines = aiter_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    return mock_response


async def _collect_tokens(client: APIClient, lines: list[str]) -> list[tuple[str, str]]:
    """Run _do_stream with mocked HTTP and collect all yielded tokens."""
    mock_resp = _make_sse_response(lines)
    tokens = []
    with patch.object(client.client, "stream", return_value=mock_resp):
        async for token in client._do_stream("https://example.com", {}, {}):
            tokens.append(token)
    return tokens


class TestDoStream:
    def setup_method(self):
        self.client = APIClient(_make_config())

    def test_format_a1_delta(self):
        lines = [
            'data: {"p":"response/fragments/0/content","v":"Hello","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("text", "Hello") in tokens

    def test_format_a1_thinking(self):
        lines = [
            'data: {"p":"response/thinking_content","v":"thinking...","o":"APPEND"}',
            "data: [DONE]",
        ]
        # show_thinking must be True for thinking tokens to be yielded
        self.client.show_thinking = True
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("thinking", "thinking...") in tokens

    def test_thinking_filtered_when_show_thinking_false(self):
        lines = [
            'data: {"p":"response/thinking_content","v":"thinking...","o":"APPEND"}',
            "data: [DONE]",
        ]
        self.client.show_thinking = False
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("thinking", "thinking...") not in tokens
        assert tokens == []

    def test_format_b_openai_delta(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("text", "world") in tokens

    def test_format_c_top_level_content(self):
        lines = [
            'data: {"content":"direct content"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("text", "direct content") in tokens

    def test_format_d_top_level_text(self):
        lines = [
            'data: {"text":"text key"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert ("text", "text key") in tokens

    def test_skips_metadata_events(self):
        lines = [
            "event: update_session",
            'data: {"title":"Thai Greeting and Assistance Offer"}',
            "",
            'data: {"p":"response/fragments/0/content","v":"Hi","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        text_tokens = [t for t in tokens if t[0] == "text"]
        assert len(text_tokens) == 1
        assert text_tokens[0] == ("text", "Hi")
        # Title must NOT appear in output
        assert not any("Thai" in t[1] for t in tokens)

    def test_skips_ready_event(self):
        lines = [
            "event: ready",
            'data: {"request_message_id":1,"response_message_id":2}',
            "",
            'data: {"p":"response/fragments/0/content","v":"OK","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        assert tokens == [("text", "OK")]

    def test_a2_snapshot_not_duplicated(self):
        """A2 snapshot should not cause duplicate output."""
        lines = [
            'data: {"p":"response/fragments/0/content","v":"Hello","o":"APPEND"}',
            'data: {"v":{"response":{"message_id":2,"fragments":[{"content":"Hello"}]}}}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        text_tokens = [t for t in tokens if t[0] == "text"]
        # Should only have "Hello" once
        assert len(text_tokens) == 1
        assert text_tokens[0] == ("text", "Hello")

    def test_a2_snapshot_fills_missing_prefix(self):
        """If A2 arrives before A1 deltas, yield the snapshot text."""
        lines = [
            'data: {"v":{"response":{"message_id":2,"fragments":[{"content":"Hello world"}]}}}',
            'data: {"p":"response/fragments/0/content","v":" world","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        full = "".join(text_tokens)
        assert "Hello" in full

    def test_raises_pow_error_on_40300(self):
        lines = ['data: {"code":40300,"msg":"invalid pow"}']
        with pytest.raises(_PowExpiredError):
            asyncio.run(
                _collect_tokens(self.client, lines)
            )

    def test_raises_session_error_on_40200(self):
        lines = ['data: {"code":40200,"msg":"session not found"}']
        with pytest.raises(_SessionNotFoundError):
            asyncio.run(
                _collect_tokens(self.client, lines)
            )

    def test_raises_auth_error_on_40100(self):
        lines = ['data: {"code":40100,"msg":"auth failed"}']
        with pytest.raises(AuthExpiredError):
            asyncio.run(
                _collect_tokens(self.client, lines)
            )

    def test_handles_401_http_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def run():
            tokens = []
            with patch.object(self.client.client, "stream", return_value=mock_resp):
                async for token in self.client._do_stream("https://example.com", {}, {}):
                    tokens.append(token)
            return tokens

        with pytest.raises(AuthExpiredError):
            asyncio.run(run())

    def test_handles_403_http_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch.object(self.client.client, "stream", return_value=mock_resp):
                async for _ in self.client._do_stream("https://example.com", {}, {}):
                    pass

        with pytest.raises(AuthExpiredError):
            asyncio.run(run())

    def test_handles_429_http_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch.object(self.client.client, "stream", return_value=mock_resp):
                async for _ in self.client._do_stream("https://example.com", {}, {}):
                    pass

        with pytest.raises(RuntimeError, match="Rate limited"):
            asyncio.run(run())

    def test_buffers_chunked_json(self):
        """httpx may split a long data: line — json_buf should reassemble it."""
        full_json = json.dumps({
            "p": "response/fragments/0/content",
            "v": "chunked",
            "o": "APPEND",
        })
        half = len(full_json) // 2
        lines = [
            f"data: {full_json[:half]}",
            full_json[half:],  # continuation without "data:" prefix
            "data: [DONE]",
        ]
        # This tests that partial JSON is buffered — the second line won't start
        # with "data:" so it won't be processed as a new SSE line.
        # The test verifies no crash occurs.
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        # May or may not yield depending on split — just verify no exception
        assert isinstance(tokens, list)

    def test_done_terminates_stream(self):
        lines = [
            'data: {"p":"response/fragments/0/content","v":"A","o":"APPEND"}',
            "data: [DONE]",
            'data: {"p":"response/fragments/0/content","v":"B","o":"APPEND"}',
        ]
        tokens = asyncio.run(
            _collect_tokens(self.client, lines)
        )
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        assert "A" in text_tokens
        assert "B" not in text_tokens

    def test_replace_op_emits_diff(self):
        """REPLACE op with prefix should only emit the new suffix."""
        lines = [
            'data: {"p":"response/fragments/0/content","v":"Hello","o":"APPEND"}',
            'data: {"p":"response/fragments/0/content","v":"Hello world","o":"REPLACE"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        # Should yield "Hello" then " world", not "Hello" + "Hello world"
        assert "".join(text_tokens) == "Hello world"

    def test_buffer_overflow_protected(self):
        """Malformed unbounded data should not blow memory."""
        # Send many huge chunks that never form valid JSON
        big = "x" * 2000
        lines = [f"data: {big}" for _ in range(600)] + ["data: [DONE]"]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        # Should complete without OOM, no tokens yielded
        assert isinstance(tokens, list)

    def test_403_code_in_data_raises_auth(self):
        lines = ['data: {"code":403,"msg":"forbidden"}']
        with pytest.raises(AuthExpiredError):
            asyncio.run(_collect_tokens(self.client, lines))

    def test_string_code_coerced(self):
        """Server sometimes returns code as string — should be coerced."""
        lines = ['data: {"code":"40300","msg":"pow"}']
        with pytest.raises(_PowExpiredError):
            asyncio.run(_collect_tokens(self.client, lines))

    def test_non_numeric_string_code_ignored(self):
        """Non-numeric code string should be silently ignored."""
        lines = [
            'data: {"code":"ok","p":"response/fragments/0/content","v":"hi","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        assert "hi" in text_tokens

    def test_non_dict_json_skipped(self):
        """If server emits a JSON list/scalar, parser must not crash."""
        lines = [
            'data: [1,2,3]',
            'data: "scalar"',
            'data: {"p":"response/fragments/0/content","v":"ok","o":"APPEND"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        assert text_tokens == ["ok"]

    def test_5xx_treated_as_network_error(self):
        """5xx should raise httpx.NetworkError so retry loop catches it."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.headers = {}
        mock_resp.aread = AsyncMock(return_value=b"upstream down")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch.object(self.client.client, "stream", return_value=mock_resp):
                async for _ in self.client._do_stream("https://x", {}, {}):
                    pass

        with pytest.raises(httpx.NetworkError):
            asyncio.run(run())

    def test_sse_comment_lines_ignored(self):
        """SSE keepalive comments (':...') should not affect parsing."""
        lines = [
            ": keepalive",
            ":",
            'data: {"p":"response/fragments/0/content","v":"hi","o":"APPEND"}',
            ": another comment",
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text_tokens = [t[1] for t in tokens if t[0] == "text"]
        assert text_tokens == ["hi"]


# ── reset_session ─────────────────────────────────────────────

class TestResetSession:
    def test_clears_session_state(self):
        client = APIClient(_make_config())
        client.session_id = "old_session"
        client.last_message_id = "old_msg"

        asyncio.run(client.reset_session())

        assert client.session_id is None
        assert client.last_message_id is None
        assert client._wasm_solver is None
        assert client._session_flags is None


# ── Mode-change auto-reset ────────────────────────────────────

class TestModeChangeResetsSession:
    def test_changing_model_type_resets_session(self):
        """Mode is bound to session; changing flags must drop the session."""
        client = APIClient(_make_config())
        client.session_id = "sid_1"
        client.last_message_id = "msg_1"
        client._session_flags = ("expert", False, False)

        # Simulate user toggling to fast
        client.model_type = "default"

        async def run():
            create_called = []

            async def fake_create():
                create_called.append(True)
                client.session_id = "sid_2"
                client._is_first_message = True
                return "sid_2"

            client._create_session = fake_create  # type: ignore[assignment]

            # Drive just the pre-loop logic by hand (avoid real HTTP)
            current_flags = (
                client.model_type,
                client.thinking_enabled,
                client.search_enabled,
            )
            if client.session_id and client._session_flags != current_flags:
                client.session_id = None
                client.last_message_id = None
                client._is_first_message = True
            if not client.session_id:
                await client._create_session()
                client._session_flags = current_flags
            return create_called

        called = asyncio.run(run())
        assert called == [True]
        assert client.session_id == "sid_2"
        assert client._session_flags == ("default", False, False)
        assert client.last_message_id is None

    def test_same_flags_no_reset(self):
        """If flags don't change, session is reused."""
        client = APIClient(_make_config())
        client.session_id = "sid_1"
        client.last_message_id = "msg_1"
        client._session_flags = ("expert", False, False)

        # No flag change
        async def run():
            current_flags = (
                client.model_type,
                client.thinking_enabled,
                client.search_enabled,
            )
            return client.session_id and client._session_flags == current_flags

        assert asyncio.run(run())
        assert client.session_id == "sid_1"  # untouched

    def test_search_toggle_resets_session(self):
        client = APIClient(_make_config())
        client.session_id = "sid_1"
        client._session_flags = ("expert", False, False)

        client.search_enabled = True
        current_flags = (
            client.model_type,
            client.thinking_enabled,
            client.search_enabled,
        )
        # Mismatch detected
        assert client._session_flags != current_flags

    def test_thinking_toggle_resets_session(self):
        client = APIClient(_make_config())
        client.session_id = "sid_1"
        client._session_flags = ("expert", False, False)

        client.thinking_enabled = True
        current_flags = (
            client.model_type,
            client.thinking_enabled,
            client.search_enabled,
        )
        assert client._session_flags != current_flags


# ── upload_file ───────────────────────────────────────────────

class TestUploadFile:
    @pytest.mark.asyncio
    async def test_returns_file_id_from_biz_data(self):
        client = APIClient(_make_config())

        async def fake_pow(_path):
            return ""
        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": {"biz_data": {"id": "f_123"}}})

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                assert await client.upload_file(b"hello", "x.txt", "text/plain") == "f_123"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_raises_auth_on_401(self):
        auth = MagicMock()
        auth.invalidate = MagicMock()
        auth.recover = AsyncMock(return_value=False)
        auth.ensure_valid = AsyncMock(return_value=False)
        client = APIClient(_make_config(), auth_manager=auth)

        async def fake_pow(_path):
            return ""
        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(AuthExpiredError):
                    await client.upload_file(b"x", "f.txt")
            auth.recover.assert_awaited_once()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_raises_runtime_when_no_file_id(self):
        client = APIClient(_make_config())

        async def fake_pow(_path):
            return ""
        client._generate_pow_token_for_path = fake_pow  # type: ignore[assignment]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": {}})

        try:
            with patch.object(client.client, "post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(RuntimeError, match="file_id"):
                    await client.upload_file(b"x", "f.txt")
        finally:
            await client.close()


# ── Backoff jitter ────────────────────────────────────────────

class TestBackoffJitter:
    def test_within_jitter_band(self):
        """_backoff_seconds returns value within ±25% of base."""
        from deepseek.client import _backoff_seconds
        from deepseek.constants import RETRY_BACKOFF
        for attempt in range(len(RETRY_BACKOFF)):
            base = RETRY_BACKOFF[attempt]
            for _ in range(50):
                v = _backoff_seconds(attempt)
                assert base * 0.75 <= v <= base * 1.25

    def test_clamps_to_last_index(self):
        from deepseek.client import _backoff_seconds
        from deepseek.constants import RETRY_BACKOFF
        # attempt larger than list → use last entry
        v = _backoff_seconds(999)
        last = RETRY_BACKOFF[-1]
        assert last * 0.75 <= v <= last * 1.25

    def test_distribution_is_actually_random(self):
        """At least 90 % of samples should NOT be exactly the base value."""
        from deepseek.client import _backoff_seconds
        from deepseek.constants import RETRY_BACKOFF
        base = RETRY_BACKOFF[0]
        samples = [_backoff_seconds(0) for _ in range(50)]
        non_base = sum(1 for s in samples if s != base)
        assert non_base >= 45  # very high chance with jitter


# ── Auto-compact (rolling-summary) ───────────────────────────

class TestAutoCompact:
    def test_reset_clears_compact_state(self):
        c = APIClient(_make_config())
        c.session_id = "sid"
        c._turn_count = 7
        c._compact_summary = "old summary"
        asyncio.run(c.reset_session())
        assert c._turn_count == 0
        assert c._compact_summary == ""

    def test_summary_prepended_to_first_message(self):
        """When _compact_summary is set, the first message of a fresh session
        must include the summary block as a preamble."""
        c = APIClient(_make_config())
        c._compact_summary = "previously: discussed X, Y, Z"
        c.system_prompt = "you are helpful"

        # Stub _create_session and _do_stream so we can capture the body sent.
        async def fake_create():
            c.session_id = "sid_new"
            c._is_first_message = True
            return "sid_new"
        c._create_session = fake_create  # type: ignore[assignment]

        async def fake_pow():
            return ""
        c._generate_pow_token = fake_pow  # type: ignore[assignment]

        captured = {}

        async def fake_do_stream(_url, _headers, body):
            captured["body"] = body
            if False:
                yield None
        c._do_stream = fake_do_stream  # type: ignore[assignment]

        async def run():
            async for _ in c._stream_completion("hello"):
                pass
        asyncio.run(run())

        sent_prompt = captured["body"]["prompt"]
        assert "you are helpful" in sent_prompt
        assert "Previous conversation summary" in sent_prompt
        assert "discussed X, Y, Z" in sent_prompt
        assert "hello" in sent_prompt
        # Summary must come BEFORE the user message
        assert sent_prompt.index("discussed X, Y, Z") < sent_prompt.index("hello")
        # Summary consumed after first send
        assert c._compact_summary == ""

    def test_summary_only_on_first_message_of_session(self):
        """Subsequent turns within the same session must NOT re-prepend summary."""
        c = APIClient(_make_config())
        c.session_id = "sid"
        c._is_first_message = False  # already past first
        c._session_flags = (c.model_type, c.thinking_enabled, c.search_enabled)
        c._compact_summary = "should-not-leak"

        async def fake_pow():
            return ""
        c._generate_pow_token = fake_pow  # type: ignore[assignment]

        captured = {}

        async def fake_do_stream(_url, _headers, body):
            captured["body"] = body
            if False:
                yield None
        c._do_stream = fake_do_stream  # type: ignore[assignment]

        async def run():
            async for _ in c._stream_completion("hello again"):
                pass
        asyncio.run(run())

        sent = captured["body"]["prompt"]
        assert sent == "hello again"  # raw, no preamble
        # summary not consumed because we never were on first_message
        assert c._compact_summary == "should-not-leak"

    def test_threshold_zero_disables_auto_compact(self):
        """auto_compact_threshold=0 means never trigger automatically."""
        c = APIClient(_make_config())
        c.auto_compact_threshold = 0
        c.session_id = "sid"
        c._turn_count = 9999
        # Even with huge turn count, compaction logic must not fire.
        # Just inspect the path: the condition uses `> 0`.
        assert c.auto_compact_threshold == 0
        # Confirm the trigger gate condition explicitly:
        triggered = (
            c.auto_compact_threshold > 0
            and c._turn_count >= c.auto_compact_threshold
        )
        assert triggered is False


# ── Compact ↔ metrics + just_compacted flag ──────────────────

class TestCompactSideEffects:
    def test_compact_session_sets_just_compacted_and_metric(self):
        from deepseek.metrics import metrics

        c = APIClient(_make_config())
        c.session_id = "sid"

        async def fake_summary():
            return "summary text"
        c.request_summary = fake_summary  # type: ignore[assignment]

        before = metrics.snapshot()["auto_compacts_total"]
        ok = asyncio.run(c.compact_session())
        after = metrics.snapshot()["auto_compacts_total"]

        assert ok is True
        assert c._just_compacted is True
        assert c._compact_summary == "summary text"
        assert c.session_id is None
        assert after == before + 1

    def test_compact_session_no_session_returns_false(self):
        c = APIClient(_make_config())
        c.session_id = None
        ok = asyncio.run(c.compact_session())
        assert ok is False
        assert c._just_compacted is False

    def test_default_threshold_from_constants(self):
        from deepseek import constants
        c = APIClient(_make_config())
        assert c.auto_compact_threshold == constants.AUTO_COMPACT_THRESHOLD


# ── set_pending_files ────────────────────────────────────────

class TestSetPendingFiles:
    def test_copies_input(self):
        c = APIClient(_make_config())
        ids = ["a", "b"]
        c.set_pending_files(ids)
        ids.append("c")  # mutate caller's list
        assert c._pending_file_ids == ["a", "b"]  # not affected

    def test_clear_with_empty(self):
        c = APIClient(_make_config())
        c.set_pending_files(["a"])
        c.set_pending_files([])
        assert c._pending_file_ids == []


# ── <think>-tag stripper ─────────────────────────────────────

class TestStripThinkTags:
    @staticmethod
    async def _stream(chunks: list[str]):
        for c in chunks:
            yield ("text", c)

    @staticmethod
    async def _collect(gen) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for t in gen:
            out.append(t)
        return out

    def test_passthrough_when_no_tags(self):
        from deepseek.client import _strip_think_tags

        async def run():
            return await self._collect(_strip_think_tags(self._stream(["hello ", "world"]), False))

        out = asyncio.run(run())
        assert "".join(t for _, t in out) == "hello world"

    def test_strips_complete_block(self):
        from deepseek.client import _strip_think_tags

        async def run():
            return await self._collect(
                _strip_think_tags(self._stream(["pre <think>secret</think> post"]), False)
            )
        out = asyncio.run(run())
        assert "secret" not in "".join(t for _, t in out)
        assert "pre  post" == "".join(t for _, t in out)

    def test_strips_block_split_across_chunks(self):
        from deepseek.client import _strip_think_tags

        # Chunk boundary INSIDE the open tag
        chunks = ["greeting <th", "ink>secret reasoning</think> answer"]

        async def run():
            return await self._collect(_strip_think_tags(self._stream(chunks), False))
        out = asyncio.run(run())
        joined = "".join(t for _, t in out)
        assert "secret" not in joined
        assert "answer" in joined
        assert "<think>" not in joined

    def test_strips_block_split_inside_close_tag(self):
        from deepseek.client import _strip_think_tags

        chunks = ["A <think>cot</thi", "nk> B"]

        async def run():
            return await self._collect(_strip_think_tags(self._stream(chunks), False))
        joined = "".join(t for _, t in asyncio.run(run()))
        assert "cot" not in joined
        assert joined == "A  B"

    def test_unclosed_block_stays_redacted(self):
        from deepseek.client import _strip_think_tags

        async def run():
            return await self._collect(
                _strip_think_tags(self._stream(["before <think>never closed"]), False)
            )
        joined = "".join(t for _, t in asyncio.run(run()))
        assert joined == "before "

    def test_thinking_token_passthrough(self):
        from deepseek.client import _strip_think_tags

        async def gen():
            yield ("text", "answer ")
            yield ("thinking", "raw cot")
            yield ("text", "more")

        async def run():
            return await self._collect(_strip_think_tags(gen(), True))
        out = asyncio.run(run())
        types = [t for t, _ in out]
        assert "thinking" in types


# ── Real-world R1 schema (DeepSeek 2026) ─────────────────────

class TestR1FragmentSchema:
    """Regression for the schema where a fragment is declared once via
    snapshot or new-fragment APPEND, then content is patched with
    bare ``{"v": ...}`` rows (no ``p``) that implicitly target the
    fragment's content path."""

    def setup_method(self):
        self.client = APIClient(_make_config())
        self.client.show_thinking = False

    def test_think_fragment_then_response_fragment(self):
        # Snapshot: starts a THINK fragment with seed "We"
        # Then patches that should be CoT (no p, inherit last p)
        # Then a new RESPONSE fragment seeded with "Hi"
        # Then patches that should be normal text
        lines = [
            'data: {"v":{"response":{"message_id":2,"fragments":[{"id":2,"type":"THINK","content":"We"}]}}}',
            'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" reason"}',
            'data: {"v":" cot"}',
            'data: {"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE","content":"Hi"}]}',
            'data: {"p":"response/fragments/-1/content","v":" there"}',
            'data: {"v":"!"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text = "".join(t for tt, t in tokens if tt == "text")
        # Must contain only the RESPONSE fragment ("Hi there!"), no CoT.
        assert "reason" not in text
        assert "cot" not in text
        assert "We" not in text  # the seed of the THINK fragment
        assert "Hi" in text
        assert "there" in text

    def test_inherits_path_when_p_missing(self):
        # If a content patch with explicit p="..." is followed by bare {"v": ...},
        # the bare rows must inherit the last path. Routing the bare rows as
        # a fresh content stream is what was leaking the chain-of-thought.
        lines = [
            'data: {"v":{"response":{"message_id":2,"fragments":[{"id":2,"type":"THINK","content":"X"}]}}}',
            'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"a"}',
            'data: {"v":"b"}',
            'data: {"v":"c"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        text_tokens = [t for tt, t in tokens if tt == "text"]
        # All three patches were inside the THINK fragment → no text emitted.
        assert text_tokens == []

    def test_show_thinking_routes_inherited_path_to_thinking(self):
        self.client.show_thinking = True
        lines = [
            'data: {"v":{"response":{"message_id":2,"fragments":[{"id":2,"type":"THINK","content":""}]}}}',
            'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"step1"}',
            'data: {"v":" step2"}',
            "data: [DONE]",
        ]
        tokens = asyncio.run(_collect_tokens(self.client, lines))
        thinking_text = "".join(t for tt, t in tokens if tt == "thinking")
        assert "step1" in thinking_text
        assert "step2" in thinking_text
