# RAG-Pipeline

End-to-end Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB and OpenAI. Transforms any PDF into a conversational knowledge base — eliminating hallucinations by grounding LLM responses in real document content.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Advanced RAG Pipeline           │
                    └─────────────────────────────────────────────┘

PDF ──► Loader ──► Smart Chunker ──► Embeddings ──► ChromaDB
                    (parent/child)    (cached)       (HNSW)

Question ──► Query Expansion ──► Hybrid Retriever ──► Reranker ──► Prompt + LLM ──► Answer
             (HyDE/Multi-query)    (BM25 + Vector)    (CrossEncoder)

┌─────────────────────────────────────────────────────────────────┐
│  Monitoring Layer: structured logs, latency tracking, metrics   │
│  Evaluation Layer: MRR, NDCG, Hit Rate, RAGAS                  │
│  Config Layer: YAML config, env vars, defaults                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
RAG-Pipeline/
├── Beginner/
│   └── rag_test.py              # Basic RAG (vector search only)
├── Intermediate/
│   ├── rag_test_v2.py           # Hybrid RAG (BM25 + vector + parent/child)
│   └── eval_rag.py              # RAGAS evaluation (3 metrics)
├── Advanced/
│   ├── __init__.py              # Package exports
│   ├── __main__.py              # CLI entry point
│   ├── config.py                # Configuration system (YAML + env vars)
│   ├── config.yaml              # Default configuration
│   ├── monitoring.py            # Structured logging & latency tracking
│   ├── reranker.py              # CrossEncoder reranking
│   ├── query_expansion.py       # HyDE & multi-query expansion
│   ├── evaluation.py            # MRR, NDCG, Hit Rate, RAGAS
│   ├── rag_pipeline.py          # Main pipeline orchestrator
│   └── tests/
│       ├── conftest.py          # Shared fixtures
│       ├── test_config.py       # Config tests
│       ├── test_evaluation.py   # IR metrics tests
│       ├── test_monitoring.py   # Monitoring tests
│       ├── test_query_expansion.py
│       └── test_reranker.py
├── chroma_db/                   # Persisted vector store
├── 100cosas-es.pdf              # Source document
├── requirements.txt             # Python dependencies
└── README.md
```

## Pipeline Levels

### Beginner (`Beginner/rag_test.py`)
- Simple vector search with ChromaDB
- Basic chunking (1000 chars)
- GPT-4o-mini for generation

### Intermediate (`Intermediate/rag_test_v2.py`)
- Hybrid retrieval: BM25 (35%) + Vector (65%)
- Parent/child document retrieval (400/1800 chars)
- RAGAS evaluation with 3 metrics

### Advanced (`Advanced/`)
- **CrossEncoder reranking** (ms-marco-MiniLM, ~50ms)
- **Query expansion**: HyDE (Hypothetical Document Embeddings) or multi-query
- **Monitoring**: structured JSON logging, latency tracking (mean, P95, P99)
- **Evaluation**: MRR, NDCG@K, Hit Rate + RAGAS
- **Caching**: LRU embedding cache to reduce API calls
- **Config system**: YAML config with env var overrides
- **Async support**: `aquery()` for concurrent requests

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### Configuration

Edit `Advanced/config.yaml` to customize:
- Chunk sizes, retrieval strategy
- Reranker model and top_n
- Query expansion strategy (hyde / multi_query / none)
- LLM model and temperature
- Monitoring and log settings

Override any config value via environment variables:
```bash
RAG_LLM_MODEL=gpt-4
RAG_RERANKER_ENABLED=false
RAG_QUERY_EXPANSION_STRATEGY=multi_query
```

## Usage

### Advanced Pipeline — Interactive CLI

```bash
python -m Advanced
```

### Single Query

```bash
python -m Advanced --query "¿Qué es el principio DRY?"
```

### Run Evaluation

```bash
python -m Advanced --eval
```

### Disable Reranker or Query Expansion

```bash
python -m Advanced --no-reranker
python -m Advanced --no-expansion
```

### Show Last Metrics

```bash
python -m Advanced --metrics
```

### Programmatic Usage

```python
from Advanced import load_config, AdvancedRAGPipeline

config = load_config()
pipeline = AdvancedRAGPipeline(config)
pipeline.initialize()

answer, docs = pipeline.query("What is technical debt?")
print(answer)

# Async
answer, docs = await pipeline.aquery("What is DRY?")

# Metrics
print(pipeline.get_metrics_summary())
```

## Advanced Components

### Reranker (`Advanced/reranker.py`)
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers. Scores each (query, document) pair and returns top_n by relevance. ~50ms inference per batch.

### Query Expansion (`Advanced/query_expansion.py`)
- **HyDE**: Generates 3 hypothetical answers, embeds them, uses for retrieval
- **Multi-Query**: Reformulates the question in 3 variants, merges and deduplicates results

### Evaluation (`Advanced/evaluation.py`)
- **MRR** (Mean Reciprocal Rank): measures rank of first relevant result
- **NDCG@K**: normalized discounted cumulative gain at K
- **Hit Rate@K**: fraction of queries with at least one relevant result in top-K
- **RAGAS**: context_recall, context_precision, answer_relevancy

Results are exported to `evaluation_results.json` for tracking over time.

### Monitoring (`Advanced/monitoring.py`)
- Structured JSON logs via `structlog`
- Latency tracking per pipeline step (retrieval, rerank, expansion, LLM)
- PipelineMetrics with mean, median, P95, P99 statistics
- Metrics export to `logs/metrics.json`

## Running Tests

```bash
pytest Advanced/tests/ -v
```

## Metrics Interpretation

| Metric | Good | Excellent |
|--------|------|-----------|
| MRR | > 0.7 | > 0.9 |
| NDCG@K | > 0.6 | > 0.8 |
| Hit Rate@K | > 0.8 | > 0.95 |
| Context Recall (RAGAS) | > 0.85 | > 0.95 |
| Context Precision (RAGAS) | > 0.75 | > 0.90 |
| Answer Relevancy (RAGAS) | > 0.80 | > 0.95 |
