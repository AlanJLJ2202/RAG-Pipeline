"""Tests for reranker module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from Advanced.config import RerankerConfig
from Advanced.reranker import CrossEncoderReranker, RerankerRetriever
from Advanced.monitoring import reset_metrics


class TestCrossEncoderReranker:
    @patch("Advanced.reranker._load_cross_encoder")
    def test_rerank_returns_top_n(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.5, 0.7, 0.3]
        mock_load.return_value = mock_model

        from langchain_core.documents import Document

        docs = [
            Document(page_content=f"Doc {i}", metadata={"page": i})
            for i in range(5)
        ]

        config = RerankerConfig(enabled=True, top_n=3)
        reranker = CrossEncoderReranker(config)
        result = reranker.rerank("test query", docs)

        assert len(result) == 3
        # Should be sorted by score descending
        scores = [d.metadata["rerank_score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    @patch("Advanced.reranker._load_cross_encoder")
    def test_rerank_disabled_passthrough(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        from langchain_core.documents import Document

        docs = [
            Document(page_content=f"Doc {i}", metadata={"page": i})
            for i in range(5)
        ]

        config = RerankerConfig(enabled=False, top_n=3)
        reranker = CrossEncoderReranker(config)
        result = reranker.rerank("test query", docs)

        assert len(result) == 3
        # Should be original order (no reranking)
        assert result[0].page_content == "Doc 0"

    @patch("Advanced.reranker._load_cross_encoder")
    def test_rerank_empty_docs(self, mock_load):
        mock_load.return_value = MagicMock()
        config = RerankerConfig(enabled=True, top_n=3)
        reranker = CrossEncoderReranker(config)
        result = reranker.rerank("test query", [])
        assert result == []

    @patch("Advanced.reranker._load_cross_encoder")
    def test_rerank_with_scores(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.2]
        mock_load.return_value = mock_model

        from langchain_core.documents import Document

        docs = [
            Document(page_content="High relevance"),
            Document(page_content="Low relevance"),
        ]

        config = RerankerConfig(enabled=True, top_n=2)
        reranker = CrossEncoderReranker(config)
        result = reranker.rerank_with_scores("test query", docs)

        assert len(result) == 2
        assert result[0][1] > result[1][1]  # first has higher score
