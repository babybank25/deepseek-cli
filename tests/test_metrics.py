"""Tests for deepseek.metrics — Prometheus-format counters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.metrics import Metrics


def test_initial_snapshot_zero():
    m = Metrics()
    s = m.snapshot()
    assert s["requests_total"] == 0
    assert s["avg_latency_ms"] == 0.0
    assert s["uptime_seconds"] >= 0


def test_incr_known_field():
    m = Metrics()
    m.incr("requests_total")
    m.incr("requests_total", by=4)
    assert m.snapshot()["requests_total"] == 5


def test_incr_unknown_field_silently_ignored():
    m = Metrics()
    m.incr("nonexistent_counter")  # must not raise
    assert m.snapshot()["requests_total"] == 0


def test_observe_latency_avg():
    m = Metrics()
    m.observe_latency(100.0)
    m.observe_latency(300.0)
    assert m.snapshot()["avg_latency_ms"] == 200.0


def test_to_prometheus_contains_known_metrics():
    m = Metrics()
    m.incr("requests_total")
    m.incr("tool_calls_total", by=3)
    text = m.to_prometheus()
    assert "deepseek_requests_total 1" in text
    assert "deepseek_tool_calls_total 3" in text
    assert "# TYPE deepseek_requests_total counter" in text


def test_to_prometheus_is_well_formed():
    m = Metrics()
    text = m.to_prometheus()
    # Each non-comment line must be `name value`
    for line in text.strip().splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        assert len(parts) >= 2
