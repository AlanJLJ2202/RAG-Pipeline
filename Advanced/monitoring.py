"""Monitoring and observability for Advanced RAG Pipeline.

Provides:
- Structured JSON logging via structlog
- Latency tracking decorators
- Pipeline metrics accumulator
- Metrics export to file
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Generator

import structlog

from .config import MonitoringConfig


# ── Structured Logging Setup ─────────────────────────────────

def setup_logging(config: MonitoringConfig) -> None:
    """Configure structlog with JSON or console output."""
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure stdlib logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    # Configure structlog processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)


# ── Latency Tracking ─────────────────────────────────────────

@dataclass
class LatencyStats:
    """Accumulate latency measurements."""
    measurements: list[float] = field(default_factory=list)

    def record(self, duration_ms: float) -> None:
        self.measurements.append(duration_ms)

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def mean(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.measurements) if self.measurements else 0.0

    @property
    def p95(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_vals = sorted(self.measurements)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def p99(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_vals = sorted(self.measurements)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def min_val(self) -> float:
        return min(self.measurements) if self.measurements else 0.0

    @property
    def max_val(self) -> float:
        return max(self.measurements) if self.measurements else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "min_ms": round(self.min_val, 2),
            "max_ms": round(self.max_val, 2),
        }


@dataclass
class PipelineMetrics:
    """Aggregate metrics for the entire pipeline."""
    retrieval_latency: LatencyStats = field(default_factory=LatencyStats)
    rerank_latency: LatencyStats = field(default_factory=LatencyStats)
    query_expansion_latency: LatencyStats = field(default_factory=LatencyStats)
    llm_latency: LatencyStats = field(default_factory=LatencyStats)
    total_latency: LatencyStats = field(default_factory=LatencyStats)
    embedding_latency: LatencyStats = field(default_factory=LatencyStats)
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "error_count": self.error_count,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "retrieval": self.retrieval_latency.to_dict(),
            "rerank": self.rerank_latency.to_dict(),
            "query_expansion": self.query_expansion_latency.to_dict(),
            "llm": self.llm_latency.to_dict(),
            "total": self.total_latency.to_dict(),
            "embedding": self.embedding_latency.to_dict(),
        }

    def export(self, path: Path) -> None:
        """Export metrics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ── Global metrics instance ──────────────────────────────────

_metrics: PipelineMetrics | None = None


def get_metrics() -> PipelineMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset global metrics (useful for tests)."""
    global _metrics
    _metrics = PipelineMetrics()


# ── Context Manager for Timing ────────────────────────────────

@contextmanager
def track_latency(stats: LatencyStats) -> Generator[None, None, None]:
    """Context manager that records elapsed time in milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats.record(elapsed_ms)


def timed(stats: LatencyStats):
    """Decorator that tracks function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with track_latency(stats):
                return func(*args, **kwargs)
        return wrapper

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with track_latency(stats):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# ── Pipeline Logger Helper ────────────────────────────────────

class PipelineLogger:
    """Convenience wrapper for pipeline step logging."""

    def __init__(self, name: str = "rag_pipeline"):
        self.log = get_logger(name)

    def query_start(self, question: str, query_id: str) -> None:
        self.log.info("query_started", question=question, query_id=query_id)

    def query_end(self, query_id: str, answer_preview: str, total_ms: float) -> None:
        self.log.info(
            "query_completed",
            query_id=query_id,
            answer_preview=answer_preview[:100],
            total_ms=round(total_ms, 2),
        )

    def query_error(self, query_id: str, error: str) -> None:
        self.log.error("query_failed", query_id=query_id, error=error)

    def retrieval_step(self, query_id: str, num_docs: int, strategy: str, ms: float) -> None:
        self.log.info(
            "retrieval_completed",
            query_id=query_id,
            num_docs=num_docs,
            strategy=strategy,
            latency_ms=round(ms, 2),
        )

    def rerank_step(self, query_id: str, input_docs: int, output_docs: int, ms: float) -> None:
        self.log.info(
            "rerank_completed",
            query_id=query_id,
            input_docs=input_docs,
            output_docs=output_docs,
            latency_ms=round(ms, 2),
        )

    def expansion_step(self, query_id: str, strategy: str, num_variants: int, ms: float) -> None:
        self.log.info(
            "query_expansion_completed",
            query_id=query_id,
            strategy=strategy,
            num_variants=num_variants,
            latency_ms=round(ms, 2),
        )

    def llm_step(self, query_id: str, model: str, ms: float) -> None:
        self.log.info(
            "llm_completed",
            query_id=query_id,
            model=model,
            latency_ms=round(ms, 2),
        )

    def cache_event(self, query_id: str, hit: bool, key: str) -> None:
        self.log.debug(
            "cache_event",
            query_id=query_id,
            hit=hit,
            key=key[:50],
        )
