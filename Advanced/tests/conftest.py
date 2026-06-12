"""Shared fixtures for Advanced RAG Pipeline tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Advanced.config import (
    ChunkingConfig,
    EmbeddingsConfig,
    EvaluationConfig,
    LLMConfig,
    LoaderConfig,
    MonitoringConfig,
    PromptConfig,
    QueryExpansionConfig,
    RAGConfig,
    RerankerConfig,
    RetrieverConfig,
    VectorstoreConfig,
    load_config,
)
from Advanced.monitoring import PipelineMetrics, reset_metrics


@pytest.fixture(autouse=True)
def reset_global_metrics():
    """Reset metrics before each test."""
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def sample_config() -> RAGConfig:
    """Minimal config for unit tests (no real API calls)."""
    return RAGConfig(
        loader=LoaderConfig(source_pdf="test.pdf"),
        chunking=ChunkingConfig(
            child_chunk_size=200,
            child_chunk_overlap=20,
            parent_chunk_size=500,
            parent_chunk_overlap=50,
        ),
        embeddings=EmbeddingsConfig(model="text-embedding-3-small"),
        vectorstore=VectorstoreConfig(
            collection_name="test_collection",
            persist_directory="/tmp/test_chroma",
        ),
        retriever=RetrieverConfig(
            strategy="hybrid",
            vector_k=4,
            bm25_k=4,
            final_k=3,
            bm25_weight=0.35,
            vector_weight=0.65,
        ),
        reranker=RerankerConfig(
            enabled=False,  # disabled for fast unit tests
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=3,
        ),
        query_expansion=QueryExpansionConfig(
            enabled=False,  # disabled for fast unit tests
            strategy="hyde",
        ),
        llm=LLMConfig(model="gpt-4o-mini", temperature=0),
        evaluation=EvaluationConfig(
            test_questions=[
                {
                    "question": "Test question?",
                    "ground_truth": "Test answer.",
                }
            ],
            metrics=["mrr", "ndcg_at_k", "hit_rate"],
            output_file="/tmp/test_eval_results.json",
        ),
        monitoring=MonitoringConfig(
            log_level="WARNING",
            log_format="console",
            log_file="/tmp/test_logs/pipeline.log",
            metrics_export_file="/tmp/test_metrics.json",
        ),
    )


@pytest.fixture
def sample_documents():
    """Sample LangChain documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="DRY stands for Don't Repeat Yourself. It avoids code duplication.",
            metadata={"page": 1, "fuente": "test"},
        ),
        Document(
            page_content="The Boy Scout Rule: leave the code better than you found it.",
            metadata={"page": 2, "fuente": "test"},
        ),
        Document(
            page_content="Technical debt accumulates from quick patches and poor naming.",
            metadata={"page": 3, "fuente": "test"},
        ),
        Document(
            page_content="Programmers spend 90% of time reading code and only 10% writing.",
            metadata={"page": 4, "fuente": "test"},
        ),
        Document(
            page_content="Dan Abramov wrote about clean code in his essay Goodbye Clean Code.",
            metadata={"page": 5, "fuente": "test"},
        ),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "This is a mock response about programming."
    llm.invoke.return_value = response
    llm.__or__ = lambda self, other: other  # allow LCEL chaining
    return llm
