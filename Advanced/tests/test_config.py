"""Tests for config module."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from Advanced.config import (
    RAGConfig,
    _apply_env_overrides,
    _deep_merge,
    load_config,
)


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"section": {"x": 1, "y": 2}}
        override = {"section": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"section": {"x": 1, "y": 3, "z": 4}}

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestEnvOverrides:
    def test_simple_override(self, monkeypatch):
        cfg = {"llm": {"model": "gpt-4", "temperature": 0}}
        monkeypatch.setenv("RAG_LLM_MODEL", "gpt-4o-mini")
        result = _apply_env_overrides(cfg)
        assert result["llm"]["model"] == "gpt-4o-mini"

    def test_type_casting(self, monkeypatch):
        cfg = {"llm": {"temperature": 0.5}}
        monkeypatch.setenv("RAG_LLM_TEMPERATURE", "0.8")
        result = _apply_env_overrides(cfg)
        assert result["llm"]["temperature"] == 0.8
        assert isinstance(result["llm"]["temperature"], float)

    def test_no_override(self):
        cfg = {"llm": {"model": "gpt-4"}}
        result = _apply_env_overrides(cfg)
        assert result["llm"]["model"] == "gpt-4"


class TestLoadConfig:
    def test_default_config(self):
        config = RAGConfig()
        assert config.llm.model == "gpt-4o-mini"
        assert config.reranker.enabled is True
        assert config.query_expansion.strategy == "hyde"

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "llm": {"model": "gpt-4", "temperature": 0.5},
            "reranker": {"enabled": False},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_file)
        assert config.llm.model == "gpt-4"
        assert config.llm.temperature == 0.5
        assert config.reranker.enabled is False

    def test_config_properties(self):
        config = RAGConfig()
        assert config.pdf_path.name == "100cosas-es.pdf"
        assert config.chroma_path.name == "chroma_db"


class TestRAGConfig:
    def test_default_values(self):
        config = RAGConfig()
        assert config.chunking.child_chunk_size == 400
        assert config.chunking.parent_chunk_size == 1800
        assert config.retriever.bm25_weight == 0.35
        assert config.retriever.vector_weight == 0.65

    def test_query_expansion_config(self):
        config = RAGConfig()
        assert config.query_expansion.hyde_num_hypotheticals == 3
        assert config.query_expansion.multi_query_num_variants == 3
