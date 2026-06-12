"""Reranker module for Advanced RAG Pipeline.

Uses a CrossEncoder model from sentence-transformers to rerank
retrieved documents by relevance to the query.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~50ms inference)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import RerankerConfig
from .monitoring import PipelineLogger, get_metrics, track_latency


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> "CrossEncoder":
    """Load and cache the CrossEncoder model."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, max_length=512)


class CrossEncoderReranker:
    """Reranks documents using a CrossEncoder model.

    Scores each (query, document) pair and returns top_n documents
    sorted by relevance score descending.
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = _load_cross_encoder(config.model_name)
        self.logger = PipelineLogger("reranker")

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        query_id: str = "",
    ) -> list[Document]:
        """Rerank documents by relevance to query.

        Args:
            query: The user question.
            documents: Retrieved documents to rerank.
            query_id: Optional tracking ID.

        Returns:
            Top-n documents sorted by relevance score.
        """
        if not documents or not self.config.enabled:
            return list(documents[: self.config.top_n])

        metrics = get_metrics()

        with track_latency(metrics.rerank_latency):
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Attach scores to documents
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: float(x[0]), reverse=True)

            result = []
            for score, doc in scored_docs[: self.config.top_n]:
                doc.metadata["rerank_score"] = float(score)
                result.append(doc)

        self.logger.rerank_step(
            query_id=query_id,
            input_docs=len(documents),
            output_docs=len(result),
            ms=0,  # tracked by context manager
        )

        return result

    def rerank_with_scores(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[tuple[Document, float]]:
        """Rerank and return (document, score) pairs."""
        if not documents or not self.config.enabled:
            return [(doc, 0.0) for doc in documents[: self.config.top_n]]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [(doc, float(score)) for doc, score in scored[: self.config.top_n]]


class RerankerRetriever(BaseRetriever):
    """LangChain retriever wrapper that applies reranking after a base retriever.

    Use as a drop-in replacement for any retriever to add reranking.
    """

    base_retriever: BaseRetriever
    reranker: CrossEncoderReranker
    query_id: str = ""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        return self.reranker.rerank(query, docs, query_id=self.query_id)
