"""Tests for evaluation module — IR metrics."""
from __future__ import annotations

import math

import pytest

from Advanced.evaluation import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    doc_to_id,
)
from Advanced.config import RAGConfig, EvaluationConfig
from Advanced.evaluation import EvaluationSuite


class TestMRR:
    def test_perfect_mrr(self):
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_at_rank_2(self):
        retrieved = [["doc2", "doc1", "doc3"]]
        relevant = [{"doc1"}]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_mrr_at_rank_3(self):
        retrieved = [["doc2", "doc3", "doc1"]]
        relevant = [{"doc1"}]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_mrr_no_relevant(self):
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc99"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_mrr_multiple_queries(self):
        retrieved = [
            ["doc1", "doc2"],
            ["doc2", "doc1"],
        ]
        relevant = [{"doc1"}, {"doc1"}]
        expected = (1.0 + 0.5) / 2
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(expected)

    def test_mrr_empty(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2", "doc3"}]
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert score == pytest.approx(1.0)

    def test_ndcg_single_relevant(self):
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1"}]
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert score == pytest.approx(1.0)

    def test_ndcg_relevant_at_end(self):
        retrieved = [["doc2", "doc3", "doc1"]]
        relevant = [{"doc1"}]
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert score == pytest.approx(expected)

    def test_ndcg_no_relevant(self):
        retrieved = [["doc1", "doc2"]]
        relevant = [{"doc99"}]
        score = ndcg_at_k(retrieved, relevant, k=2)
        assert score == 0.0

    def test_ndcg_empty(self):
        assert ndcg_at_k([], []) == 0.0


class TestHitRate:
    def test_hit_rate_perfect(self):
        retrieved = [["doc1", "doc2"], ["doc3", "doc4"]]
        relevant = [{"doc1"}, {"doc3"}]
        assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0

    def test_hit_rate_partial(self):
        retrieved = [["doc1", "doc2"], ["doc3", "doc4"]]
        relevant = [{"doc1"}, [{"doc99"}][0]]
        assert hit_rate_at_k(retrieved, relevant, k=2) == 0.5

    def test_hit_rate_at_k_cutoff(self):
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc3"}]
        # doc3 is at rank 3, k=2 should miss
        assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
        # k=3 should hit
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_hit_rate_empty(self):
        assert hit_rate_at_k([], []) == 0.0


class TestDocToId:
    def test_stable_id(self, sample_documents):
        doc = sample_documents[0]
        id1 = doc_to_id(doc)
        id2 = doc_to_id(doc)
        assert id1 == id2

    def test_different_docs_different_ids(self, sample_documents):
        id1 = doc_to_id(sample_documents[0])
        id2 = doc_to_id(sample_documents[1])
        assert id1 != id2

    def test_id_length(self, sample_documents):
        doc_id = doc_to_id(sample_documents[0])
        assert len(doc_id) == 12


class TestEvaluationSuite:
    def test_default_relevance(self):
        question = "What is the DRY principle?"
        content = "DRY stands for Don't Repeat Yourself and avoids code duplication."
        assert EvaluationSuite._default_relevance(question, content) is True

    def test_default_relevance_no_match(self):
        question = "What is the DRY principle?"
        content = "The weather is sunny today with clear skies."
        assert EvaluationSuite._default_relevance(question, content) is False
