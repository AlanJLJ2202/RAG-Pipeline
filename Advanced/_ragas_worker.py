"""Isolated RAGAS worker.

Runs RAGAS evaluation in its own process so that a native crash
(segfault from torch/pyarrow/datasets interop) cannot take down the
main evaluation suite. Invoked as:

    python -m Advanced._ragas_worker <input.json> <output.json>

Input JSON: {"questions": [...], "answers": [...],
             "contexts": [[...]], "ground_truths": [...]}
Output JSON: {"context_recall": float, ...} or {"error": str}

The parent process treats a missing/invalid output file (e.g. the
worker segfaulted before writing) as a RAGAS failure and keeps the
classic IR metrics it already computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: _ragas_worker.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    def write(result: dict) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    try:
        with open(in_path, encoding="utf-8") as f:
            payload = json.load(f)

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall

        dataset = Dataset.from_dict({
            "question": payload["questions"],
            "answer": payload["answers"],
            "contexts": payload["contexts"],
            "ground_truth": payload["ground_truths"],
        })

        result = evaluate(
            dataset,
            metrics=[context_recall, context_precision, answer_relevancy],
        )

        write({
            "context_recall": float(result["context_recall"]),
            "context_precision": float(result["context_precision"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        })
    except ImportError:
        write({"error": "ragas not installed"})
    except Exception as e:  # noqa: BLE001 - report any failure to parent
        write({"error": str(e)})


if __name__ == "__main__":
    main()
