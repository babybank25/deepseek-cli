"""
Process-wide counters and latency tracker, exported in Prometheus text format.

Stays intentionally simple — atomic counters + lock-protected latency average.
Idea adopted from ThaiLLM Gateway: structured ops counters that operators can
scrape with `curl /metrics`. We do NOT persist across restarts (gauges reset).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _State:
    requests_total: int = 0
    streamed_requests_total: int = 0
    failed_requests_total: int = 0
    auth_errors_total: int = 0
    pool_no_account_total: int = 0
    tool_calls_total: int = 0
    tool_calls_invalid_json_total: int = 0
    tool_calls_no_name_total: int = 0
    file_uploads_total: int = 0
    file_uploads_failed_total: int = 0
    auto_compacts_total: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    started_at: float = 0.0


class Metrics:
    """Thread-safe in-memory metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s = _State(started_at=time.time())

    # ── Mutators ──────────────────────────────────────────────

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            current = getattr(self._s, name, None)
            if current is None or not isinstance(current, int):
                return
            setattr(self._s, name, current + by)

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._s.latency_sum_ms += ms
            self._s.latency_count += 1

    # ── Snapshot ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            avg = (
                self._s.latency_sum_ms / self._s.latency_count
                if self._s.latency_count
                else 0.0
            )
            return {
                "requests_total": self._s.requests_total,
                "streamed_requests_total": self._s.streamed_requests_total,
                "failed_requests_total": self._s.failed_requests_total,
                "auth_errors_total": self._s.auth_errors_total,
                "pool_no_account_total": self._s.pool_no_account_total,
                "tool_calls_total": self._s.tool_calls_total,
                "tool_calls_invalid_json_total": self._s.tool_calls_invalid_json_total,
                "tool_calls_no_name_total": self._s.tool_calls_no_name_total,
                "file_uploads_total": self._s.file_uploads_total,
                "file_uploads_failed_total": self._s.file_uploads_failed_total,
                "auto_compacts_total": self._s.auto_compacts_total,
                "avg_latency_ms": avg,
                "uptime_seconds": time.time() - self._s.started_at,
            }

    def to_prometheus(self) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP deepseek_requests_total Total /v1/chat/completions requests received",
            "# TYPE deepseek_requests_total counter",
            f"deepseek_requests_total {snap['requests_total']}",
            "# HELP deepseek_streamed_requests_total Streaming requests",
            "# TYPE deepseek_streamed_requests_total counter",
            f"deepseek_streamed_requests_total {snap['streamed_requests_total']}",
            "# HELP deepseek_failed_requests_total Failed (non-2xx) requests",
            "# TYPE deepseek_failed_requests_total counter",
            f"deepseek_failed_requests_total {snap['failed_requests_total']}",
            "# HELP deepseek_auth_errors_total Auth-expired errors propagated",
            "# TYPE deepseek_auth_errors_total counter",
            f"deepseek_auth_errors_total {snap['auth_errors_total']}",
            "# HELP deepseek_pool_no_account_total 503 (no account) responses",
            "# TYPE deepseek_pool_no_account_total counter",
            f"deepseek_pool_no_account_total {snap['pool_no_account_total']}",
            "# HELP deepseek_tool_calls_total Tool calls successfully parsed",
            "# TYPE deepseek_tool_calls_total counter",
            f"deepseek_tool_calls_total {snap['tool_calls_total']}",
            "# HELP deepseek_tool_calls_invalid_json_total Tool-call blocks with malformed JSON",
            "# TYPE deepseek_tool_calls_invalid_json_total counter",
            f"deepseek_tool_calls_invalid_json_total {snap['tool_calls_invalid_json_total']}",
            "# HELP deepseek_tool_calls_no_name_total Tool-call blocks missing 'name' field",
            "# TYPE deepseek_tool_calls_no_name_total counter",
            f"deepseek_tool_calls_no_name_total {snap['tool_calls_no_name_total']}",
            "# HELP deepseek_file_uploads_total File uploads accepted",
            "# TYPE deepseek_file_uploads_total counter",
            f"deepseek_file_uploads_total {snap['file_uploads_total']}",
            "# HELP deepseek_file_uploads_failed_total File uploads that errored",
            "# TYPE deepseek_file_uploads_failed_total counter",
            f"deepseek_file_uploads_failed_total {snap['file_uploads_failed_total']}",
            "# HELP deepseek_auto_compacts_total Times auto-compact rolled the session",
            "# TYPE deepseek_auto_compacts_total counter",
            f"deepseek_auto_compacts_total {snap['auto_compacts_total']}",
            "# HELP deepseek_avg_latency_ms Average request latency in ms",
            "# TYPE deepseek_avg_latency_ms gauge",
            f"deepseek_avg_latency_ms {snap['avg_latency_ms']:.2f}",
            "# HELP deepseek_uptime_seconds Server uptime",
            "# TYPE deepseek_uptime_seconds gauge",
            f"deepseek_uptime_seconds {snap['uptime_seconds']:.2f}",
        ]
        return "\n".join(lines) + "\n"


# Global registry (single server process)
metrics = Metrics()
