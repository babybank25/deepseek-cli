"""Process-wide counters and latency tracker exported in Prometheus format."""
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
    auth_refresh_total: int = 0
    auth_refresh_success_total: int = 0
    auth_refresh_failed_total: int = 0
    pool_no_account_total: int = 0
    conversation_affinity_hits: int = 0
    conversation_bindings_total: int = 0
    session_resume_previous_response: int = 0
    session_resume_history: int = 0
    tool_calls_total: int = 0
    tool_calls_invalid_json_total: int = 0
    tool_calls_no_name_total: int = 0
    tool_recovery_total: int = 0
    tool_recovery_success_total: int = 0
    file_uploads_total: int = 0
    file_uploads_failed_total: int = 0
    auto_compacts_total: int = 0
    pow_local_success_total: int = 0
    pow_browser_fallback_total: int = 0
    pow_browser_success_total: int = 0
    pow_node_success_total: int = 0
    pow_failed_total: int = 0
    protocol_probe_fail_total: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    started_at: float = 0.0


class Metrics:
    """Thread-safe in-memory metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s = _State(started_at=time.time())

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            current = getattr(self._s, name, None)
            if isinstance(current, int):
                setattr(self._s, name, current + by)

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._s.latency_sum_ms += ms
            self._s.latency_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            values = {
                name: getattr(self._s, name)
                for name in self._s.__dataclass_fields__
                if name not in {"latency_sum_ms", "latency_count", "started_at"}
            }
            values["avg_latency_ms"] = (
                self._s.latency_sum_ms / self._s.latency_count
                if self._s.latency_count
                else 0.0
            )
            values["uptime_seconds"] = time.time() - self._s.started_at
            return values

    def to_prometheus(self) -> str:
        snapshot = self.snapshot()
        help_text = {
            "requests_total": "Total compatibility API requests received",
            "streamed_requests_total": "Streaming requests",
            "failed_requests_total": "Failed requests",
            "auth_errors_total": "Auth errors propagated",
            "auth_refresh_total": "Browser auth recovery attempts",
            "auth_refresh_success_total": "Successful auth recoveries",
            "auth_refresh_failed_total": "Failed auth recoveries",
            "pool_no_account_total": "Requests with no safe account available",
            "conversation_affinity_hits": "Requests resolved to an existing account binding",
            "conversation_bindings_total": "New conversation/account bindings",
            "session_resume_previous_response": "Sessions resumed via previous_response_id",
            "session_resume_history": "Sessions resumed via exact history fingerprint",
            "tool_calls_total": "Tool calls successfully parsed",
            "tool_calls_invalid_json_total": "Malformed tool-call blocks",
            "tool_calls_no_name_total": "Tool-call blocks missing a function name",
            "tool_recovery_total": "Tool response recovery attempts",
            "tool_recovery_success_total": "Successful tool response recoveries",
            "file_uploads_total": "File uploads accepted",
            "file_uploads_failed_total": "File uploads that errored",
            "auto_compacts_total": "Automatic or manual session compactions",
            "pow_local_success_total": "PoW challenges solved by embedded WASM",
            "pow_browser_fallback_total": "PoW attempts delegated to browser worker fallback",
            "pow_browser_success_total": "PoW challenges solved by browser worker fallback",
            "pow_node_success_total": "PoW challenges solved by Node fallback",
            "pow_failed_total": "PoW challenges that could not be solved",
            "protocol_probe_fail_total": "Protocol capability probes that failed",
        }
        lines: list[str] = []
        for name, description in help_text.items():
            metric = f"deepseek_{name}"
            lines.extend(
                [
                    f"# HELP {metric} {description}",
                    f"# TYPE {metric} counter",
                    f"{metric} {snapshot[name]}",
                ]
            )
        lines.extend(
            [
                "# HELP deepseek_avg_latency_ms Average request latency in ms",
                "# TYPE deepseek_avg_latency_ms gauge",
                f"deepseek_avg_latency_ms {snapshot['avg_latency_ms']:.2f}",
                "# HELP deepseek_uptime_seconds Server uptime",
                "# TYPE deepseek_uptime_seconds gauge",
                f"deepseek_uptime_seconds {snapshot['uptime_seconds']:.2f}",
            ]
        )
        return "\n".join(lines) + "\n"


metrics = Metrics()
