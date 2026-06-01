"""Tests for deepseek.sniffer — NetworkSniffer heuristics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.models import CapturedRequest
from deepseek.sniffer import NetworkSniffer


def _make_cap(url: str, method: str = "POST", body: str = "", status: int = 200) -> CapturedRequest:
    return CapturedRequest(
        url=url,
        method=method,
        request_headers={},
        request_body="",
        response_status=status,
        response_headers={},
        response_body=body,
    )


class TestDetectChatApi:
    def setup_method(self):
        self.sniffer = NetworkSniffer()

    def test_detects_deepseek_json_patch_format(self):
        self.sniffer.captured = [
            _make_cap(
                "https://chat.deepseek.com/api/v0/chat/completion",
                body='data: {"v":"hello","p":"response/fragments/0/content","o":"APPEND"}',
            )
        ]
        result = self.sniffer.detect_chat_api()
        assert result is not None
        assert "completion" in result.url

    def test_detects_openai_format(self):
        self.sniffer.captured = [
            _make_cap(
                "https://api.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"hi"}}]}',
            )
        ]
        result = self.sniffer.detect_chat_api()
        assert result is not None

    def test_ignores_get_requests(self):
        self.sniffer.captured = [
            _make_cap(
                "https://chat.deepseek.com/api/v0/chat/completion",
                method="GET",
                body='{"v":"hello"}',
            )
        ]
        result = self.sniffer.detect_chat_api()
        assert result is None

    def test_ignores_low_score_requests(self):
        self.sniffer.captured = [
            _make_cap("https://example.com/api/data", body='{"status":"ok"}')
        ]
        result = self.sniffer.detect_chat_api()
        assert result is None

    def test_returns_highest_score(self):
        low = _make_cap("https://example.com/api/other", body='{"message":"x"}')
        high = _make_cap(
            "https://chat.deepseek.com/api/v0/chat/completion",
            body='{"v":"text","p":"content","o":"APPEND","choices":[{"delta":{"content":"hi"}}]}',
        )
        self.sniffer.captured = [low, high]
        result = self.sniffer.detect_chat_api()
        assert result is high

    def test_returns_none_when_empty(self):
        self.sniffer.captured = []
        assert self.sniffer.detect_chat_api() is None


class TestListCandidates:
    def setup_method(self):
        self.sniffer = NetworkSniffer()

    def test_returns_post_with_body(self):
        self.sniffer.captured = [
            _make_cap("https://example.com/api", body="some response"),
            _make_cap("https://example.com/api", method="GET", body="ignored"),
            _make_cap("https://example.com/api", body=""),  # empty body — excluded
        ]
        candidates = self.sniffer.list_candidates()
        assert len(candidates) == 1
        assert candidates[0].method == "POST"
