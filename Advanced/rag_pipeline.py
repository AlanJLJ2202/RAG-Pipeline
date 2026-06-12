"""Advanced RAG Pipeline — main orchestrator.

Integrates:
- Smart chunking (parent/child)
- Hybrid retrieval (BM25 + vector)
- Query expansion (HyDE / multi-query)
- CrossEncoder reranking
- Monitoring and metrics
- Embedding cache
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig
from .monitoring import (
    PipelineLogger,
    PipelineMetrics,
    get_metrics,
    reset_metrics,
    setup_logging,
    track_latency,
)
from .query_expansion import QueryExpander, merge_retrieval_results
from .reranker import CrossEncoderReranker


# ── Embedding Cache ───────────────────────────────────────────

class EmbeddingCache:
    """LRU cache for embeddings to avoid redundant API calls."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, text: str) -> list[float] | None:
        key = self._hash(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        key = self._hash(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


class CachedOpenAIEmbeddings:
    """Wrapper around OpenAIEmbeddings with LRU caching."""

    def __init__(self, embeddings: OpenAIEmbeddings, cache: EmbeddingCache):
        self._embeddings = embeddings
        self._cache = cache

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            get_metrics().cache_hits += 1
            return cached
        get_metrics().cache_misses += 1
        result = self._embeddings.embed_query(text)
        self._cache.put(text, result)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        to_compute: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                get_metrics().cache_hits += 1
                results.append(cached)
            else:
                get_metrics().cache_misses += 1
                results.append([])
                to_compute.append((i, text))

        if to_compute:
            batch_texts = [t for _, t in to_compute]
            batch_embeddings = self._embeddings.embed_documents(batch_texts)
            for (idx, text), emb in zip(to_compute, batch_embeddings):
                self._cache.put(text, emb)
                results[idx] = emb

        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


# ── Main Pipeline ─────────────────────────────────────────────

class AdvancedRAGPipeline:
    """Production-ready RAG pipeline with reranking, query expansion,
    monitoring, and caching.

    Usage:
        config = load_config()
        pipeline = AdvancedRAGPipeline(config)
        pipeline.initialize()
        answer, docs = pipeline.query("What is DRY?")
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = PipelineLogger("rag_pipeline")
        self.metrics = get_metrics()

        # Components (initialized in initialize())
        self._embeddings: OpenAIEmbeddings | None = None
        self._cached_embeddings: CachedOpenAIEmbeddings | None = None
        self._vectorstore: Chroma | None = None
        self._hybrid_retriever: EnsembleRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._query_expander: QueryExpander | None = None
        self._llm: ChatOpenAI | None = None
        self._chain: Any = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        from dotenv import load_dotenv
        load_dotenv(self.config.project_root / ".env")

        setup_logging(self.config.monitoring)
        self.logger.log.info("pipeline_initializing")

        # Embeddings with cache
        self._embeddings = OpenAIEmbeddings(model=self.config.embeddings.model)
        cache = EmbeddingCache(max_size=self.config.embeddings.cache_max_size)
        self._cached_embeddings = CachedOpenAIEmbeddings(self._embeddings, cache)

        # Vector store
        self._vectorstore = Chroma(
            collection_name=self.config.vectorstore.collection_name,
            embedding_function=self._embeddings,  # Chroma needs the raw embeddings
            persist_directory=str(self.config.chroma_path),
        )

        # Load and index documents
        self._build_retriever()

        # Reranker
        if self.config.reranker.enabled:
            self._reranker = CrossEncoderReranker(self.config.reranker)

        # Query expansion
        self._llm = ChatOpenAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        if self.config.query_expansion.enabled:
            self._query_expander = QueryExpander(
                self.config.query_expansion, self._llm,
            )

        # Build chain
        self._build_chain()

        self._initialized = True
        self.logger.log.info("pipeline_initialized")

    def _build_retriever(self) -> None:
        """Build the hybrid retriever (BM25 + vector)."""
        pdf_path = self.config.pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        documentos = loader.load()

        # Enrich metadata
        for doc in documentos:
            doc.metadata["fuente"] = self.config.loader.metadata_fields.get("fuente", "unknown")
            threshold = self.config.loader.metadata_fields.get("seccion_threshold_page", 20)
            doc.metadata["seccion"] = "inicio" if doc.metadata.get("page", 0) < threshold else "cuerpo"

        # Chunking
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.child_chunk_size,
            chunk_overlap=self.config.chunking.child_chunk_overlap,
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.parent_chunk_size,
            chunk_overlap=self.config.chunking.parent_chunk_overlap,
        )

        # Build or load vector store
        if self.config.chroma_path.exists() and any(self.config.chroma_path.iterdir()):
            self.logger.log.info("loading_existing_vectorstore")
        else:
            self.logger.log.info("creating_vectorstore")
            store = InMemoryStore()
            parent_retriever = ParentDocumentRetriever(
                vectorstore=self._vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            parent_retriever.add_documents(documentos)
            self.logger.log.info("vectorstore_created", pages=len(documentos))

        # Hybrid retriever
        chunks = child_splitter.split_documents(documentos)

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = self.config.retriever.bm25_k

        vector_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.config.retriever.vector_k},
        )

        self._hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[
                self.config.retriever.bm25_weight,
                self.config.retriever.vector_weight,
            ],
        )

        self.logger.log.info(
            "retriever_ready",
            strategy=self.config.retriever.strategy,
            bm25_k=self.config.retriever.bm25_k,
            vector_k=self.config.retriever.vector_k,
        )

    def _build_chain(self) -> None:
        """Build the LangChain LCEL chain."""
        prompt = ChatPromptTemplate.from_template(self.config.prompt.template)

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(
                f"[Página {d.metadata.get('page', '?')}] {d.page_content}"
                for d in docs
            )

        self._chain = (
            {
                "context": RunnablePassthrough() | self._retrieve_and_rerank | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

    def _retrieve_and_rerank(self, question: str) -> list[Document]:
        """Retrieve documents with optional query expansion and reranking."""
        query_id = str(uuid.uuid4())[:8]
        self.metrics.total_queries += 1

        # Query expansion
        with track_latency(self.metrics.query_expansion_latency):
            if self._query_expander:
                queries = self._query_expander.expand(question, query_id=query_id)
            else:
                queries = [question]

        # Retrieve for all query variants
        with track_latency(self.metrics.retrieval_latency):
            if len(queries) == 1:
                docs = self._hybrid_retriever.invoke(queries[0])
            else:
                all_results = [
                    self._hybrid_retriever.invoke(q) for q in queries
                ]
                docs = merge_retrieval_results(
                    all_results, max_docs=self.config.retriever.vector_k * 2,
                )

        self.logger.retrieval_step(
            query_id=query_id,
            num_docs=len(docs),
            strategy=self.config.query_expansion.strategy if self._query_expander else "direct",
            ms=0,
        )

        # Rerank
        if self._reranker:
            with track_latency(self.metrics.rerank_latency):
                docs = self._reranker.rerank(question, docs, query_id=query_id)

        return docs

    def query(self, question: str) -> tuple[str, list[Document]]:
        """Execute a single query.

        Returns:
            Tuple of (answer, retrieved_documents).
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        query_id = str(uuid.uuid4())[:8]
        self.logger.query_start(question, query_id)

        with track_latency(self.metrics.total_latency):
            try:
                # Retrieve docs separately for return value
                docs = self._retrieve_and_rerank(question)

                # Generate answer
                prompt = ChatPromptTemplate.from_template(self.config.prompt.template)
                context = "\n\n".join(
                    f"[Página {d.metadata.get('page', '?')}] {d.page_content}"
                    for d in docs
                )
                chain = prompt | self._llm | StrOutputParser()

                with track_latency(self.metrics.llm_latency):
                    answer = chain.invoke({"context": context, "question": question})

                self.logger.query_end(query_id, answer, 0)
                return answer, docs

            except Exception as e:
                self.metrics.error_count += 1
                self.logger.query_error(query_id, str(e))
                raise

    async def aquery(self, question: str) -> tuple[str, list[Document]]:
        """Async version of query."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        query_id = str(uuid.uuid4())[:8]
        self.logger.query_start(question, query_id)

        with track_latency(self.metrics.total_latency):
            try:
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(
                    None, self._retrieve_and_rerank, question,
                )

                prompt = ChatPromptTemplate.from_template(self.config.prompt.template)
                context = "\n\n".join(
                    f"[Página {d.metadata.get('page', '?')}] {d.page_content}"
                    for d in docs
                )
                chain = prompt | self._llm | StrOutputParser()

                with track_latency(self.metrics.llm_latency):
                    answer = await chain.ainvoke({"context": context, "question": question})

                self.logger.query_end(query_id, answer, 0)
                return answer, docs

            except Exception as e:
                self.metrics.error_count += 1
                self.logger.query_error(query_id, str(e))
                raise

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current pipeline metrics."""
        return self.metrics.to_dict()

    def export_metrics(self) -> None:
        """Export metrics to configured file."""
        path = Path(self.config.monitoring.metrics_export_file)
        self.metrics.export(self.config.project_root / path)


# ── CLI Interactive Mode ──────────────────────────────────────

def run_interactive(pipeline: AdvancedRAGPipeline) -> None:
    """Run the pipeline in interactive CLI mode."""
    print("\n" + "=" * 60)
    print("  Advanced RAG Pipeline — Interactive Mode")
    print("=" * 60)
    print("  Type your question or 'salir' to exit.")
    print("  Type '/metrics' to see pipeline metrics.")
    print("=" * 60 + "\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            break
        if pregunta == "/metrics":
            import json
            print(json.dumps(pipeline.get_metrics_summary(), indent=2))
            continue

        try:
            answer, docs = pipeline.query(pregunta)
            print(f"\nAsistente: {answer}\n")
            print(f"  [{len(docs)} documentos recuperados]")
            for i, doc in enumerate(docs[:3], 1):
                page = doc.metadata.get("page", "?")
                score = doc.metadata.get("rerank_score", "")
                score_str = f" (score: {score:.3f})" if score else ""
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"    {i}. [p.{page}]{score_str} {preview}...")
            print()
        except Exception as e:
            print(f"\n  Error: {e}\n")

    # Export metrics on exit
    pipeline.export_metrics()
    print("\nMetrics exported. Goodbye!")
