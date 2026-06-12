"""Advanced RAG Pipeline package.

Provides production-ready RAG with:
- CrossEncoder reranking
- HyDE / multi-query expansion
- Monitoring and structured logging
- IR evaluation metrics (MRR, NDCG, Hit Rate)
- Embedding caching
"""
from .config import RAGConfig, load_config
from .monitoring import PipelineMetrics, PipelineLogger, reset_metrics

# Lazy imports for modules that depend on chromadb/langchain
# These are imported on first access to avoid import errors
# on systems with older sqlite3 versions
__all__ = [
    "AdvancedRAGPipeline",
    "RAGConfig",
    "load_config",
    "CrossEncoderReranker",
    "QueryExpander",
    "EvaluationSuite",
    "PipelineMetrics",
    "PipelineLogger",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "hit_rate_at_k",
    "reset_metrics",
]


def __getattr__(name: str):
    if name == "AdvancedRAGPipeline":
        from .rag_pipeline import AdvancedRAGPipeline
        return AdvancedRAGPipeline
    elif name == "CrossEncoderReranker":
        from .reranker import CrossEncoderReranker
        return CrossEncoderReranker
    elif name == "QueryExpander":
        from .query_expansion import QueryExpander
        return QueryExpander
    elif name == "EvaluationSuite":
        from .evaluation import EvaluationSuite
        return EvaluationSuite
    elif name == "mean_reciprocal_rank":
        from .evaluation import mean_reciprocal_rank
        return mean_reciprocal_rank
    elif name == "ndcg_at_k":
        from .evaluation import ndcg_at_k
        return ndcg_at_k
    elif name == "hit_rate_at_k":
        from .evaluation import hit_rate_at_k
        return hit_rate_at_k
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
