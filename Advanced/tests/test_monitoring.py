"""Tests for monitoring module."""
from __future__ import annotations

import time

import pytest

from Advanced.monitoring import (
    LatencyStats,
    PipelineMetrics,
    PipelineLogger,
    get_metrics,
    reset_metrics,
    track_latency,
)


class TestLatencyStats:
    def test_empty_stats(self):
        stats = LatencyStats()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.median == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0

    def test_single_measurement(self):
        stats = LatencyStats()
        stats.record(100.0)
        assert stats.count == 1
        assert stats.mean == 100.0
        assert stats.median == 100.0
        assert stats.min_val == 100.0
        assert stats.max_val == 100.0

    def test_multiple_measurements(self):
        stats = LatencyStats()
        for i in range(100):
            stats.record(float(i))
        assert stats.count == 100
        assert stats.mean == pytest.approx(49.5)
        assert stats.median == pytest.approx(49.5)
        assert stats.min_val == 0.0
        assert stats.max_val == 99.0

    def test_p95(self):
        stats = LatencyStats()
        for i in range(100):
            stats.record(float(i))
        assert stats.p95 == pytest.approx(95.0)

    def test_to_dict(self):
        stats = LatencyStats()
        stats.record(10.0)
        stats.record(20.0)
        d = stats.to_dict()
        assert d["count"] == 2
        assert d["mean_ms"] == 15.0
        assert "median_ms" in d
        assert "p95_ms" in d
        assert "p99_ms" in d


class TestTrackLatency:
    def test_tracks_time(self):
        stats = LatencyStats()
        with track_latency(stats):
            time.sleep(0.01)  # 10ms
        assert stats.count == 1
        assert stats.measurements[0] >= 10  # at least 10ms

    def test_tracks_on_exception(self):
        stats = LatencyStats()
        with pytest.raises(ValueError):
            with track_latency(stats):
                raise ValueError("test")
        assert stats.count == 1


class TestPipelineMetrics:
    def test_initial_state(self):
        metrics = PipelineMetrics()
        assert metrics.total_queries == 0
        assert metrics.error_count == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_cache_hit_rate(self):
        metrics = PipelineMetrics()
        metrics.cache_hits = 7
        metrics.cache_misses = 3
        d = metrics.to_dict()
        assert d["cache_hit_rate"] == pytest.approx(0.7)

    def test_cache_hit_rate_no_accesses(self):
        metrics = PipelineMetrics()
        d = metrics.to_dict()
        assert d["cache_hit_rate"] == 0.0

    def test_to_dict_structure(self):
        metrics = PipelineMetrics()
        d = metrics.to_dict()
        assert "total_queries" in d
        assert "error_count" in d
        assert "cache_hit_rate" in d
        assert "retrieval" in d
        assert "rerank" in d
        assert "query_expansion" in d
        assert "llm" in d
        assert "total" in d
        assert "embedding" in d

    def test_export(self, tmp_path):
        metrics = PipelineMetrics()
        metrics.total_queries = 5
        out_path = tmp_path / "metrics.json"
        metrics.export(out_path)
        assert out_path.exists()

        import json
        with open(out_path) as f:
            data = json.load(f)
        assert data["total_queries"] == 5


class TestGlobalMetrics:
    def test_get_metrics_singleton(self):
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_metrics(self):
        m1 = get_metrics()
        m1.total_queries = 42
        reset_metrics()
        m2 = get_metrics()
        assert m2.total_queries == 0
        assert m2 is not m1


class TestPipelineLogger:
    def test_logger_creation(self):
        logger = PipelineLogger("test")
        assert logger is not None

    def test_query_start(self, capsys):
        logger = PipelineLogger("test")
        # Should not raise
        logger.query_start("test question", "abc123")

    def test_cache_event(self, capsys):
        logger = PipelineLogger("test")
        logger.cache_event("abc123", True, "some_key")
