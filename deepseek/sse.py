"""Transport-only Server-Sent Events framing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Optional

MAX_SSE_BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SSEEvent:
    event: Optional[str]
    data: object


class SSEFrameParser:
    """Incrementally parse SSE blocks across arbitrary network chunk boundaries."""

    def __init__(self, max_buffer_bytes: int = MAX_SSE_BUFFER_BYTES) -> None:
        self._buffer = ""
        self._max_buffer_bytes = max_buffer_bytes

    def feed(self, chunk: str) -> list[SSEEvent]:
        self._buffer += chunk
        if len(self._buffer.encode("utf-8", errors="ignore")) > self._max_buffer_bytes:
            self._buffer = ""
            raise ValueError("SSE frame buffer exceeded safety limit")
        events: list[SSEEvent] = []
        while True:
            boundary = self._next_boundary()
            if boundary is None:
                break
            index, width = boundary
            block = self._buffer[:index]
            self._buffer = self._buffer[index + width :]
            event = self._parse_block(block)
            if event is not None:
                events.append(event)
        return events

    def finish(self) -> list[SSEEvent]:
        block = self._buffer
        self._buffer = ""
        event = self._parse_block(block)
        return [event] if event is not None else []

    def _next_boundary(self) -> Optional[tuple[int, int]]:
        boundaries = []
        for token in ("\r\n\r\n", "\n\n"):
            index = self._buffer.find(token)
            if index >= 0:
                boundaries.append((index, len(token)))
        return min(boundaries, default=None, key=lambda item: item[0])

    @staticmethod
    def _parse_block(block: str) -> Optional[SSEEvent]:
        if not block.strip():
            return None
        event: Optional[str] = None
        data_lines: list[str] = []
        unknown_lines: list[str] = []
        for raw_line in block.splitlines():
            if raw_line.startswith(":"):
                continue
            if raw_line.startswith("event:"):
                event = raw_line[6:].strip() or None
            elif raw_line.startswith("data:"):
                data_lines.append(raw_line[5:].lstrip())
            elif raw_line.strip():
                unknown_lines.append(raw_line)

        if data_lines:
            raw_data = "\n".join(data_lines)
        elif unknown_lines:
            raw_data = "\n".join(unknown_lines)
        else:
            return None
        if raw_data.strip() == "[DONE]":
            return SSEEvent(event=event, data="[DONE]")
        try:
            return SSEEvent(event=event, data=json.loads(raw_data))
        except json.JSONDecodeError:
            return SSEEvent(event=event, data=raw_data)


async def iter_sse_events(response) -> AsyncIterator[SSEEvent]:
    parser = SSEFrameParser()
    async for chunk in response.aiter_text():
        if chunk:
            for event in parser.feed(chunk):
                yield event
    for event in parser.finish():
        yield event
