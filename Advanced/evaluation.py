"""Evaluation module for Advanced RAG Pipeline.

Provides:
- Classic IR metrics: MRR, NDCG@K, Hit Rate
- RAGAS integration for faithfulness/relevancy
- EvaluationSuite that runs full benchmark
- JSON export for tracking over time
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import EvaluationConfig, RAGConfig
from .monitoring import PipelineLogger, get_metrics


# ── Classic IR Metrics ────────────────────────────────────────

def mean_reciprocal_rank(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    MRR = (1/|Q|) * sum(1/rank_i) where rank_i is the rank of the
    first relevant document for query i.

    Args:
        retrieved_ids: List of retrieved doc ID lists, one per query.
        relevant_ids: List of sets of relevant doc IDs, one per query.

    Returns:
        MRR score in [0, 1].
    """
    if not retrieved_ids:
        return 0.0

    total_rr = 0.0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                total_rr += 1.0 / rank
                break

    return total_rr / len(retrieved_ids)


def ndcg_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    Args:
        retrieved_ids: List of retrieved doc ID lists, one per query.
        relevant_ids: List of sets of relevant doc IDs, one per query.
        k: Cutoff rank.

    Returns:
        NDCG@K score in [0, 1].
    """
    if not retrieved_ids:
        return 0.0

    def dcg(retrieved: list[str], relevant: set[str], k: int) -> float:
        score = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                score += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        return score

    def idcg(relevant_count: int, k: int) -> float:
        score = 0.0
        for i in range(min(relevant_count, k)):
            score += 1.0 / math.log2(i + 2)
        return score

    total_ndcg = 0.0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        dcg_val = dcg(retrieved, relevant, k)
        idcg_val = idcg(len(relevant), k)
        if idcg_val > 0:
            total_ndcg += dcg_val / idcg_val

    return total_ndcg / len(retrieved_ids)


def hit_rate_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int = 10,
) -> float:
    """Compute Hit Rate at K.

    Fraction of queries where at least one relevant doc appears in top-K.

    Args:
        retrieved_ids: List of retrieved doc ID lists, one per query.
        relevant_ids: List of sets of relevant doc IDs, one per query.
        k: Cutoff rank.

    Returns:
        Hit Rate in [0, 1].
    """
    if not retrieved_ids:
        return 0.0

    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k = set(retrieved[:k])
        if top_k & relevant:
            hits += 1

    return hits / len(retrieved_ids)


# ── Document ID Helpers ───────────────────────────────────────

def doc_to_id(doc: Document) -> str:
    """Generate a stable ID for a document based on content hash."""
    import hashlib
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:12]


# ── RAGAS Wrapper ─────────────────────────────────────────────

def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Run RAGAS evaluation and return metrics as dict."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[context_recall, context_precision, answer_relevancy],
        )

        return {
            "context_recall": result["context_recall"],
            "context_precision": result["context_precision"],
            "answer_relevancy": result["answer_relevancy"],
        }
    except ImportError:
        return {"error": "ragas not installed"}
    except Exception as e:
        return {"error": str(e)}


# ── Evaluation Suite ──────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of a single query evaluation."""
    question: str
    answer: str
    ground_truth: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: set[str]
    latency_ms: float = 0.0


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""
    timestamp: str
    total_questions: int
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float
    ragas_metrics: dict[str, float]
    avg_latency_ms: float
    p95_latency_ms: float
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            "ragas": self.ragas_metrics,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "details": self.details,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  EVALUATION RESULTS",
            "=" * 60,
            f"  Questions:        {self.total_questions}",
            f"  MRR:              {self.mrr:.4f}",
            f"  NDCG@K:           {self.ndcg_at_k:.4f}",
            f"  Hit Rate@K:       {self.hit_rate_at_k:.4f}",
            f"  Avg Latency:      {self.avg_latency_ms:.1f} ms",
            f"  P95 Latency:      {self.p95_latency_ms:.1f} ms",
            "-" * 60,
            "  RAGAS Metrics:",
        ]
        for key, val in self.ragas_metrics.items():
            if isinstance(val, float):
                lines.append(f"    {key:20s}: {val:.4f}")
            else:
                lines.append(f"    {key:20s}: {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


class EvaluationSuite:
    """Run full evaluation of the RAG pipeline.

    Usage:
        suite = EvaluationSuite(config)
        summary = suite.run(pipeline_query_fn)
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.eval_config = config.evaluation
        self.logger = PipelineLogger("evaluation")

    def run(
        self,
        query_fn: Callable[[str], tuple[str, list[Document], float]],
        relevance_fn: Callable[[str, str], bool] | None = None,
    ) -> EvalSummary:
        """Run evaluation on configured test questions.

        Args:
            query_fn: Function that takes a question and returns
                      (answer, retrieved_docs, latency_ms).
            relevance_fn: Optional function (question, doc_content) -> bool
                          to determine if a doc is relevant. If None, uses
                          a simple keyword overlap heuristic.

        Returns:
            EvalSummary with all metrics.
        """
        if relevance_fn is None:
            relevance_fn = self._default_relevance

        test_questions = self.eval_config.test_questions
        if not test_questions:
            self.logger.log.warning("no_test_questions configured")
            return EvalSummary(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_questions=0,
                mrr=0, ndcg_at_k=0, hit_rate_at_k=0,
                ragas_metrics={}, avg_latency_ms=0, p95_latency_ms=0,
            )

        results: list[EvalResult] = []
        questions_text: list[str] = []
        answers_text: list[str] = []
        contexts_list: list[list[str]] = []
        ground_truths: list[str] = []

        for i, item in enumerate(test_questions, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]

            self.logger.log.info(
                "eval_question",
                index=i,
                total=len(test_questions),
                question=question[:80],
            )

            answer, docs, latency_ms = query_fn(question)

            # Build doc IDs and relevance
            doc_ids = [doc_to_id(d) for d in docs]
            relevant = {
                doc_to_id(d)
                for d in docs
                if relevance_fn(question, d.page_content)
            }

            results.append(EvalResult(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                retrieved_doc_ids=doc_ids,
                relevant_doc_ids=relevant,
                latency_ms=latency_ms,
            ))

            questions_text.append(question)
            answers_text.append(answer)
            contexts_list.append([d.page_content for d in docs])
            ground_truths.append(ground_truth)

        # Compute IR metrics
        all_retrieved = [r.retrieved_doc_ids for r in results]
        all_relevant = [r.relevant_doc_ids for r in results]
        k = self.config.retriever.final_k

        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
        ndcg = ndcg_at_k(all_retrieved, all_relevant, k=k)
        hr = hit_rate_at_k(all_retrieved, all_relevant, k=k)

        # Latency stats
        latencies = [r.latency_ms for r in results]
        sorted_lat = sorted(latencies)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0

        # RAGAS (if enabled)
        ragas_metrics: dict[str, float] = {}
        if "ragas" in self.eval_config.metrics:
            ragas_metrics = run_ragas_evaluation(
                questions_text, answers_text, contexts_list, ground_truths,
            )

        # Build details
        details = [
            {
                "question": r.question,
                "answer_preview": r.answer[:200],
                "num_retrieved": len(r.retrieved_doc_ids),
                "num_relevant": len(r.relevant_doc_ids),
                "latency_ms": round(r.latency_ms, 2),
            }
            for r in results
        ]

        summary = EvalSummary(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_questions=len(results),
            mrr=mrr,
            ndcg_at_k=ndcg,
            hit_rate_at_k=hr,
            ragas_metrics=ragas_metrics,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            details=details,
        )

        # Export if configured
        if self.eval_config.output_file:
            out_path = self.config.project_root / self.eval_config.output_file
            self._export_results(summary, out_path)

        return summary

    def _export_results(self, summary: EvalSummary, path: Path) -> None:
        """Export results to JSON, appending to existing history."""
        history: list[dict] = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        history = data
                    elif isinstance(data, dict):
                        history = [data]
            except (json.JSONDecodeError, OSError):
                pass

        history.append(summary.to_dict())

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        self.logger.log.info("evaluation_exported", path=str(path))

    @staticmethod
    def _default_relevance(question: str, doc_content: str) -> bool:
        """Simple keyword overlap heuristic for relevance.

        A document is considered relevant if >30% of question keywords
        appear in the document content.
        """
        import re

        def tokenize(text: str) -> set[str]:
            words = re.findall(r"\w+", text.lower())
            # Remove very short words and common stopwords
            stopwords = {
                "el", "la", "los", "las", "un", "una", "de", "del", "en",
                "que", "es", "y", "a", "por", "con", "para", "no", "se",
                "su", "al", "lo", "como", "más", "o", "pero", "sus",
                "the", "is", "a", "an", "and", "or", "but", "in", "on",
                "at", "to", "for", "of", "with", "by", "it", "this",
            }
            return {w for w in words if len(w) > 2 and w not in stopwords}

        q_tokens = tokenize(question)
        d_tokens = tokenize(doc_content)

        if not q_tokens:
            return True  # fallback: consider relevant

        overlap = len(q_tokens & d_tokens) / len(q_tokens)
        return overlap > 0.3
