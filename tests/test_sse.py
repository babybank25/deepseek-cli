"""Transport framing regressions for DeepSeek SSE."""
import json

import pytest

from deepseek.sse import SSEFrameParser


def test_parser_handles_arbitrary_chunk_boundaries():
    parser = SSEFrameParser()
    payload = 'event: message\ndata: {"value":"hello"}\n\n'
    events = []
    for chunk in (payload[:7], payload[7:19], payload[19:31], payload[31:]):
        events.extend(parser.feed(chunk))
    assert len(events) == 1
    assert events[0].event == "message"
    assert events[0].data == {"value": "hello"}


def test_parser_joins_multiline_data():
    parser = SSEFrameParser()
    events = parser.feed('data: {"a": 1,\ndata: "b": 2}\n\n')
    assert events[0].data == {"a": 1, "b": 2}


def test_parser_preserves_done_sentinel():
    parser = SSEFrameParser()
    assert parser.feed("data: [DONE]\n\n")[0].data == "[DONE]"


def test_parser_flushes_final_block_without_blank_line():
    parser = SSEFrameParser()
    parser.feed('data: {"ok":true}')
    events = parser.finish()
    assert events[0].data == {"ok": True}


def test_parser_preserves_unprefixed_json_error():
    parser = SSEFrameParser()
    event = parser.feed('{"code":40300,"msg":"pow"}\n\n')[0]
    assert event.data["code"] == 40300


def test_parser_buffer_limit_fails_closed():
    parser = SSEFrameParser(max_buffer_bytes=16)
    with pytest.raises(ValueError, match="safety limit"):
        parser.feed("x" * 17)
