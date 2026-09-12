"""CLI entry point: run the multi-hop RAG audit over a fixed HotpotQA sample
and write projects/02_hotpotqa_multihop_rag/results.json.

Usage (from repo root):
    uv run python projects/02_hotpotqa_multihop_rag/run_experiment.py --n 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (  # noqa: E402
    answer_with_citations,
    build_chat_model,
    list_ollama_models,
    load_hotpotqa_sample,
    pick_chat_model,
    score_example,
    two_hop_retrieve,
)

RESULTS_PATH = Path(__file__).parent / "results.json"


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA multi-hop RAG audit")
    parser.add_argument("--n", type=int, default=50, help="number of validation examples")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for the sample")
    parser.add_argument("--hop1-k", type=int, default=4)
    parser.add_argument("--hop2-k", type=int, default=4)
    args = parser.parse_args()

    t_start = time.time()

    with httpx.Client() as client:
        available = list_ollama_models(client)
        chat_model_name = pick_chat_model(available)
        print(f"[setup] available ollama models: {available}")
        print(f"[setup] using chat model: {chat_model_name}")

        print(f"[data] loading HotpotQA distractor/validation, n={args.n}, seed={args.seed} ...")
        records = load_hotpotqa_sample(args.n, args.seed)
        print(f"[data] loaded {len(records)} examples")

        structured_llm = build_chat_model(chat_model_name)

        per_example = []
        for i, record in enumerate(records):
            t0 = time.time()
            fed, hop1_trace, hop2_trace = two_hop_retrieve(
                record.question, record.sentences, client, args.hop1_k, args.hop2_k
            )
            try:
                cited = answer_with_citations(structured_llm, record.question, fed)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] example {i} ({record.qid}) generation failed: {exc}")
                continue
            result = score_example(record, fed, cited)
            result.hop1_query = hop1_trace.query
            result.hop2_query = hop2_trace.query
            per_example.append(result)
            dt = time.time() - t0
            print(
                f"[{i + 1}/{len(records)}] qid={record.qid} em={result.answer_em} "
                f"claim_vs_gold_f1={result.claim_vs_gold_f1:.2f} "
                f"claim_vs_fed_p={result.claim_vs_fed_precision:.2f} "
                f"fabricated={len(result.fabricated_citations)} ({dt:.1f}s)"
            )

    n_ok = len(per_example)
    summary = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "sample_size_requested": args.n,
        "sample_size_completed": n_ok,
        "seed": args.seed,
        "embed_model": "nomic-embed-text",
        "chat_model": chat_model_name,
        "hop1_k": args.hop1_k,
        "hop2_k": args.hop2_k,
        "retrieval_strategy": (
            "2-hop dense retrieval: hop1 embeds the raw question and takes top-k1 "
            "sentences by cosine similarity; hop2 embeds question+hop1-sentence-text "
            "(to carry forward whatever bridge entity/fact hop1 surfaced) and takes "
            "top-k2 additional sentences not already selected. hop1+hop2 sentences are "
            "the exact set fed into the answer-generation prompt."
        ),
        "metrics": {
            "answer_exact_match": mean([1.0 if r.answer_em else 0.0 for r in per_example]),
            "answer_f1": mean([r.answer_f1 for r in per_example]),
            "retrieval_recall_of_gold_facts": mean([r.retrieval_recall_of_gold for r in per_example]),
            "claim_vs_gold_precision": mean([r.claim_vs_gold_precision for r in per_example]),
            "claim_vs_gold_recall": mean([r.claim_vs_gold_recall for r in per_example]),
            "claim_vs_gold_f1": mean([r.claim_vs_gold_f1 for r in per_example]),
            "claim_vs_fed_precision": mean([r.claim_vs_fed_precision for r in per_example]),
            "claim_vs_fed_recall": mean([r.claim_vs_fed_recall for r in per_example]),
            "pct_examples_with_fabricated_citation": mean(
                [1.0 if r.fabricated_citations else 0.0 for r in per_example]
            ),
            "avg_fabricated_citations_per_example": mean(
                [float(len(r.fabricated_citations)) for r in per_example]
            ),
        },
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "examples": [
            {
                "qid": r.qid,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "model_answer": r.model_answer,
                "answer_em": r.answer_em,
                "answer_f1": round(r.answer_f1, 3),
                "gold_facts": r.gold_facts,
                "fed_facts": r.fed_facts,
                "claimed_facts": r.claimed_facts,
                "fabricated_citations": r.fabricated_citations,
                "retrieval_recall_of_gold": round(r.retrieval_recall_of_gold, 3),
                "claim_vs_gold_f1": round(r.claim_vs_gold_f1, 3),
                "claim_vs_fed_precision": round(r.claim_vs_fed_precision, 3),
            }
            for r in per_example
        ],
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {RESULTS_PATH} ({n_ok}/{args.n} examples completed)")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
