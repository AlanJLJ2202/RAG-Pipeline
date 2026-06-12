"""Entry point for Advanced RAG Pipeline.

Usage:
    python -m Advanced                    # Interactive CLI
    python -m Advanced --eval             # Run evaluation
    python -m Advanced --query "question" # Single query
    python -m Advanced --metrics          # Show last metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from Advanced.config import load_config
from Advanced.evaluation import EvaluationSuite
from Advanced.rag_pipeline import AdvancedRAGPipeline, run_interactive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and print the answer.",
    )
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run the evaluation suite.",
    )
    parser.add_argument(
        "--metrics", "-m",
        action="store_true",
        help="Show metrics from the last run.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml (default: Advanced/config.yaml).",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker.",
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable query expansion.",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # CLI overrides
    if args.no_reranker:
        config.reranker.enabled = False
    if args.no_expansion:
        config.query_expansion.enabled = False

    # Show metrics mode
    if args.metrics:
        metrics_path = config.project_root / config.monitoring.metrics_export_file
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        else:
            print(f"No metrics found at {metrics_path}")
        return

    # Initialize pipeline
    pipeline = AdvancedRAGPipeline(config)
    pipeline.initialize()

    if args.eval:
        # Evaluation mode
        suite = EvaluationSuite(config)

        def query_fn(question: str):
            answer, docs = pipeline.query(question)
            metrics = pipeline.metrics
            latency = metrics.total_latency.measurements[-1] if metrics.total_latency.measurements else 0
            return answer, docs, latency

        summary = suite.run(query_fn)
        print(summary)
    elif args.query:
        # Single query mode
        answer, docs = pipeline.query(args.query)
        print(f"\nRespuesta: {answer}\n")
        print(f"Documentos recuperados: {len(docs)}")
        for i, doc in enumerate(docs[:5], 1):
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("rerank_score", "")
            score_str = f" (rerank: {score:.3f})" if score else ""
            print(f"  {i}. [p.{page}]{score_str} {doc.page_content[:100]}...")
    else:
        # Interactive mode
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
