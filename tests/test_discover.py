"""Tests for deepseek.discover — internal helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.discover import _detect_api_paths, _extract_auth
from deepseek.models import CapturedRequest
from deepseek.sniffer import NetworkSniffer


def _make_cap(url: str, method: str = "POST", headers=None) -> CapturedRequest:
    return CapturedRequest(
        url=url,
        method=method,
        request_headers=headers or {},
        request_body="",
        response_status=200,
        response_headers={},
        response_body="",
    )


# ── _extract_auth ─────────────────────────────────────────────

class TestExtractAuth:
    def test_lowercase_header(self):
        cap = _make_cap("https://x", headers={"authorization": "Bearer xyz"})
        token, kind = _extract_auth(cap)
        assert token == "xyz"
        assert kind == "bearer"

    def test_capitalized_header(self):
        cap = _make_cap("https://x", headers={"Authorization": "Bearer abc"})
        token, kind = _extract_auth(cap)
        assert token == "abc"
        assert kind == "bearer"

    def test_mixed_case_header(self):
        cap = _make_cap("https://x", headers={"AUTHORIZATION": "Bearer mix"})
        token, kind = _extract_auth(cap)
        assert token == "mix"
        assert kind == "bearer"

    def test_non_bearer_token(self):
        cap = _make_cap("https://x", headers={"authorization": "Token raw_value"})
        token, kind = _extract_auth(cap)
        assert token == "Token raw_value"
        assert kind == "header"

    def test_no_auth(self):
        cap = _make_cap("https://x", headers={})
        token, kind = _extract_auth(cap)
        assert token == ""
        assert kind == "bearer"

    def test_lowercase_bearer_scheme(self):
        """Some clients send 'bearer' instead of 'Bearer'."""
        cap = _make_cap("https://x", headers={"authorization": "bearer xyz"})
        token, kind = _extract_auth(cap)
        assert token == "xyz"
        assert kind == "bearer"


# ── _detect_api_paths ─────────────────────────────────────────

class TestDetectApiPaths:
    def test_falls_back_to_defaults_when_empty(self):
        s = NetworkSniffer()
        s.captured = []
        comp, sess, pow_p = _detect_api_paths(s)
        assert comp == "/api/v0/chat/completion"
        assert sess == "/api/v0/chat_session/create"
        assert pow_p == "/api/v0/chat/create_pow_challenge"

    def test_detects_completion_path(self):
        s = NetworkSniffer()
        s.captured = [_make_cap("https://x.com/api/v1/chat/completion")]
        comp, _, _ = _detect_api_paths(s)
        assert comp == "/api/v1/chat/completion"

    def test_does_not_overwrite_session_with_history(self):
        """Path /api/.../session_history should NOT clobber session_path."""
        s = NetworkSniffer()
        s.captured = [
            _make_cap("https://x.com/api/v0/chat_session/create"),
            _make_cap("https://x.com/api/v0/chat_session_history/list"),
        ]
        _, sess, _ = _detect_api_paths(s)
        assert sess == "/api/v0/chat_session/create"

    def test_detects_pow_challenge_path(self):
        s = NetworkSniffer()
        s.captured = [_make_cap("https://x.com/api/v2/chat/create_pow_challenge")]
        _, _, pow_p = _detect_api_paths(s)
        assert pow_p == "/api/v2/chat/create_pow_challenge"

    def test_ignores_get_requests(self):
        s = NetworkSniffer()
        s.captured = [_make_cap("https://x.com/api/v0/chat/completion", method="GET")]
        comp, _, _ = _detect_api_paths(s)
        # Should fall back to default since GET is ignored
        assert comp == "/api/v0/chat/completion"
