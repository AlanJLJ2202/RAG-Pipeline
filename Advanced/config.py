"""Configuration system for Advanced RAG Pipeline.

Loads settings from config.yaml with env var overrides.
Env vars: RAG_<SECTION>_<KEY> (uppercase, underscores).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(cfg: dict, prefix: str = "RAG") -> dict:
    """Override config values with env vars like RAG_LLM_MODEL."""
    for key, value in os.environ.items():
        if not key.startswith(prefix + "_"):
            continue
        parts = key[len(prefix) + 1 :].lower().split("_")
        if len(parts) < 2:
            continue
        section, param = parts[0], "_".join(parts[1:])
        if section in cfg and isinstance(cfg[section], dict):
            # Try to cast to original type
            original = cfg[section].get(param)
            if original is not None:
                try:
                    value = type(original)(value)  # type: ignore[call-arg]
                except (ValueError, TypeError):
                    pass
            cfg[section][param] = value
    return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class LoaderConfig:
    source_pdf: str = "100cosas-es.pdf"
    metadata_fields: dict = field(default_factory=lambda: {
        "fuente": "100cosas-es",
        "seccion_threshold_page": 20,
    })


@dataclass
class ChunkingConfig:
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1800
    parent_chunk_overlap: int = 100


@dataclass
class EmbeddingsConfig:
    model: str = "text-embedding-3-small"
    cache_max_size: int = 2048
    batch_size: int = 64


@dataclass
class VectorstoreConfig:
    provider: str = "chroma"
    collection_name: str = "rag_v2"
    persist_directory: str = "chroma_db"


@dataclass
class RetrieverConfig:
    strategy: str = "hybrid"
    vector_k: int = 8
    bm25_k: int = 8
    final_k: int = 6
    bm25_weight: float = 0.35
    vector_weight: float = 0.65


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 6
    batch_size: int = 32


@dataclass
class QueryExpansionConfig:
    enabled: bool = True
    strategy: str = "hyde"
    hyde_num_hypotheticals: int = 3
    multi_query_num_variants: int = 3


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0
    max_tokens: int = 1024


@dataclass
class PromptConfig:
    template: str = (
        "Responde la pregunta basándote SOLO en el siguiente contexto.\n"
        "Si no sabes la respuesta, di \"No encontré esa información en el documento.\"\n\n"
        "Contexto: {context}\nPregunta: {question}"
    )


@dataclass
class EvaluationConfig:
    test_questions: list[dict] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: ["mrr", "ndcg_at_k", "hit_rate", "ragas"])
    output_file: str = "evaluation_results.json"


@dataclass
class MonitoringConfig:
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/pipeline.log"
    enable_latency_tracking: bool = True
    enable_metrics_export: bool = True
    metrics_export_file: str = "logs/metrics.json"


@dataclass
class RAGConfig:
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    vectorstore: VectorstoreConfig = field(default_factory=VectorstoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def pdf_path(self) -> Path:
        return self.project_root / self.loader.source_pdf

    @property
    def chroma_path(self) -> Path:
        return self.project_root / self.vectorstore.persist_directory

    @property
    def log_dir(self) -> Path:
        return self.project_root / self.monitoring.log_file.rsplit("/", 1)[0]


def _dict_to_dataclass(dataclass_cls: type, data: dict) -> Any:
    """Convert a dict to a dataclass, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(dataclass_cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dataclass_cls(**filtered)


def load_config(config_path: Path | None = None) -> RAGConfig:
    """Load config from YAML, apply env overrides, return RAGConfig."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = _load_yaml(config_path)
        raw = _apply_env_overrides(raw)

    return RAGConfig(
        loader=_dict_to_dataclass(LoaderConfig, raw.get("loader", {})),
        chunking=_dict_to_dataclass(ChunkingConfig, raw.get("chunking", {})),
        embeddings=_dict_to_dataclass(EmbeddingsConfig, raw.get("embeddings", {})),
        vectorstore=_dict_to_dataclass(VectorstoreConfig, raw.get("vectorstore", {})),
        retriever=_dict_to_dataclass(RetrieverConfig, raw.get("retriever", {})),
        reranker=_dict_to_dataclass(RerankerConfig, raw.get("reranker", {})),
        query_expansion=_dict_to_dataclass(QueryExpansionConfig, raw.get("query_expansion", {})),
        llm=_dict_to_dataclass(LLMConfig, raw.get("llm", {})),
        prompt=_dict_to_dataclass(PromptConfig, raw.get("prompt", {})),
        evaluation=_dict_to_dataclass(EvaluationConfig, raw.get("evaluation", {})),
        monitoring=_dict_to_dataclass(MonitoringConfig, raw.get("monitoring", {})),
    )
