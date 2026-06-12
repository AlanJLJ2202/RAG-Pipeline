"""Query Expansion module for Advanced RAG Pipeline.

Two strategies:
1. HyDE (Hypothetical Document Embeddings): generates hypothetical answers,
   embeds them, and uses them for retrieval.
2. Multi-Query: reformulates the question in N variants, retrieves for each,
   and deduplicates results.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import QueryExpansionConfig
from .monitoring import PipelineLogger, get_metrics, track_latency


_HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Eres un experto en el tema. Genera un párrafo corto que responda "
    "directamente a la siguiente pregunta. El párrafo debe ser informativo "
    "y contener datos específicos (nombres, cifras, conceptos).\n\n"
    "Pregunta: {question}\n\nPárrafo:"
)

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "Dada la siguiente pregunta del usuario, genera {n} reformulaciones "
    "alternativas que capturen diferentes ángulos de la misma intención. "
    "Cada reformulación debe ser una sola oración. "
    "Devuelve SOLO las reformulaciones, una por línea, sin numerar.\n\n"
    "Pregunta original: {question}\n\nReformulaciones:"
)


class QueryExpander:
    """Expands queries to improve retrieval recall."""

    def __init__(self, config: QueryExpansionConfig, llm: BaseChatModel):
        self.config = config
        self.llm = llm
        self.logger = PipelineLogger("query_expansion")

    def expand(self, question: str, query_id: str = "") -> list[str]:
        """Expand question into multiple queries.

        Returns list of query strings (original + expansions).
        If disabled or strategy=none, returns [question].
        """
        if not self.config.enabled or self.config.strategy == "none":
            return [question]

        metrics = get_metrics()

        with track_latency(metrics.query_expansion_latency):
            if self.config.strategy == "hyde":
                expansions = self._hyde(question)
            elif self.config.strategy == "multi_query":
                expansions = self._multi_query(question)
            else:
                expansions = [question]

        self.logger.expansion_step(
            query_id=query_id,
            strategy=self.config.strategy,
            num_variants=len(expansions),
            ms=0,
        )

        return expansions

    def _hyde(self, question: str) -> list[str]:
        """HyDE: generate hypothetical answers to use as queries."""
        chain = _HYDE_PROMPT | self.llm

        hypotheticals: list[str] = [question]  # always include original
        for _ in range(self.config.hyde_num_hypotheticals):
            try:
                response = chain.invoke({"question": question})
                text = response.content.strip()
                if text:
                    hypotheticals.append(text)
            except Exception:
                continue

        return hypotheticals

    def _multi_query(self, question: str) -> list[str]:
        """Multi-query: reformulate the question in N variants."""
        chain = _MULTI_QUERY_PROMPT | self.llm

        try:
            response = chain.invoke({
                "question": question,
                "n": self.config.multi_query_num_variants,
            })
            variants = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
        except Exception:
            variants = []

        # Always include original question first
        return [question] + variants


def deduplicate_documents(
    documents: Sequence[Document],
    max_docs: int | None = None,
) -> list[Document]:
    """Remove duplicate documents based on content hash.

    Preserves order (first occurrence kept).
    """
    seen: set[str] = set()
    unique: list[Document] = []

    for doc in documents:
        content_hash = hashlib.md5(
            doc.page_content.encode("utf-8")
        ).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)

    if max_docs is not None:
        unique = unique[:max_docs]

    return unique


def merge_retrieval_results(
    results_per_query: list[list[Document]],
    max_docs: int | None = None,
) -> list[Document]:
    """Merge and deduplicate results from multiple query retrievals.

    Interleaves results from each query to maintain diversity.
    """
    merged: list[Document] = []
    max_len = max(len(r) for r in results_per_query) if results_per_query else 0

    # Round-robin merge
    for i in range(max_len):
        for query_results in results_per_query:
            if i < len(query_results):
                merged.append(query_results[i])

    return deduplicate_documents(merged, max_docs=max_docs)
