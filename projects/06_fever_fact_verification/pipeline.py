"""End-to-end FEVER fact-verification pipeline: sample -> retrieve -> verdict -> score.

Run via run_eval.py. This module holds the orchestration logic so it can
also be imported and unit-tested piecewise (see tests/test_06_fever.py).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """See data.py's `_sibling` docstring for why this isn't a bare `import config`."""
    key = f"fever06_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


config = _sibling("config")
data = _sibling("data")
retrieval = _sibling("retrieval")
verdict = _sibling("verdict")

FEVER_LABELS = config.FEVER_LABELS


def gold_titles_for_claim(claim: dict[str, Any]) -> set[str]:
    titles: set[str] = set()
    for ev_set in claim["evidence_sets"]:
        for title, _sent_id in ev_set:
            titles.add(title)
    return titles


def evidence_hit(retrieved_titles: set[str], gold_titles: set[str]) -> bool:
    """Title-level evidence recall: did retrieval surface >=1 gold page?

    Exact gold sentence-ID matching is not meaningful here because
    evidence sentences come from *current* live Wikipedia, not FEVER's
    frozen 2017 snapshot (see data.py docstring) -- so we score whether
    the retriever found the right *page*, which current sentence text can
    still validate or contradict.
    """
    if not gold_titles:
        return False
    return bool(retrieved_titles & gold_titles)


def build_confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    labels = [*FEVER_LABELS, "PARSE_ERROR"]
    matrix = {gold: dict.fromkeys(labels, 0) for gold in FEVER_LABELS}
    for gold, pred in pairs:
        if pred not in matrix.get(gold, {}):
            matrix.setdefault(gold, dict.fromkeys(labels, 0))
        matrix[gold][pred] = matrix[gold].get(pred, 0) + 1
    return matrix


def run_pipeline(
    n_per_label: int = config.N_CLAIMS_PER_LABEL,
    seed: int = config.SEED,
    top_k: int = config.TOP_K_EVIDENCE,
    chat_model: str = config.CHAT_MODEL,
    embed_model: str = config.EMBED_MODEL,
    progress_cb=print,
) -> dict[str, Any]:
    t0 = time.time()

    progress_cb("== Step 1: sample claims from FEVER validation (labelled_dev) ==")
    claims = data.sample_claims(n_per_label=n_per_label, seed=seed)
    progress_cb(f"Sampled {len(claims)} claims.")
    label_counts = {lbl: sum(1 for c in claims if c["label"] == lbl) for lbl in FEVER_LABELS}
    progress_cb(f"Label breakdown: {label_counts}")

    progress_cb("== Step 2: collect target Wikipedia titles from gold evidence ==")
    target_titles = data.target_titles_for(claims)
    progress_cb(f"{len(target_titles)} unique gold-evidence pages referenced by this sample.")

    progress_cb("== Step 3: fetch those Wikipedia pages (live MediaWiki API) ==")
    wiki_pages = data.fetch_wiki_pages(target_titles, progress_cb=progress_cb)

    progress_cb("== Step 4: build the shared sentence retrieval pool ==")
    pool = retrieval.build_pool(wiki_pages)
    progress_cb(f"Pool has {len(pool)} sentences across {len(wiki_pages)} pages.")

    progress_cb("== Step 5: embed pool sentences + claims (nomic-embed-text via Ollama) ==")
    with httpx.Client(base_url=config.OLLAMA_URL, timeout=120) as embed_client:
        pool_vecs = retrieval.embed_texts(
            [p.text for p in pool],
            model=embed_model,
            client=embed_client,
            progress_cb=progress_cb,
        )
        claim_vecs = retrieval.embed_texts(
            [c["claim"] for c in claims],
            model=embed_model,
            client=embed_client,
            progress_cb=progress_cb,
        )

    progress_cb(f"== Step 6: retrieve top-{top_k} evidence + get verdicts ({chat_model}) ==")
    per_claim_results = []
    gold_pred_pairs: list[tuple[str, str]] = []
    evidence_hits = 0
    evidence_eligible = 0

    with httpx.Client(base_url=config.OLLAMA_URL, timeout=180) as chat_client:
        for i, claim in enumerate(claims):
            top = retrieval.retrieve_top_k(claim_vecs[i], pool, pool_vecs, k=top_k)
            retrieved = [(sent.title, sent.text) for sent, _score in top]
            retrieved_titles = {sent.title for sent, _score in top}

            v = verdict.get_verdict(
                claim["claim"], retrieved, model=chat_model, client=chat_client
            )

            gold_titles = gold_titles_for_claim(claim)
            hit = evidence_hit(retrieved_titles, gold_titles)
            if gold_titles:
                evidence_eligible += 1
                if hit:
                    evidence_hits += 1

            gold_pred_pairs.append((claim["label"], v["label"]))
            per_claim_results.append(
                {
                    "id": claim["id"],
                    "claim": claim["claim"],
                    "gold_label": claim["label"],
                    "predicted_label": v["label"],
                    "correct": claim["label"] == v["label"],
                    "gold_titles": sorted(gold_titles),
                    "retrieved": [
                        {
                            "title": sent.title,
                            "sent_id": sent.sent_id,
                            "text": sent.text,
                            "score": round(score, 4),
                        }
                        for sent, score in top
                    ],
                    "evidence_title_hit": hit,
                    "raw_model_response": v["raw_response"],
                }
            )
            progress_cb(
                f"[{i + 1}/{len(claims)}] gold={claim['label']:<16} pred={v['label']:<16} "
                f"{'OK' if claim['label'] == v['label'] else 'X'}"
            )

    correct = sum(1 for r in per_claim_results if r["correct"])
    accuracy = correct / len(claims) if claims else 0.0
    confusion = build_confusion_matrix(gold_pred_pairs)
    evidence_recall = evidence_hits / evidence_eligible if evidence_eligible else None

    per_label_accuracy = {}
    for lbl in FEVER_LABELS:
        rows = [r for r in per_claim_results if r["gold_label"] == lbl]
        per_label_accuracy[lbl] = (sum(r["correct"] for r in rows) / len(rows)) if rows else None

    elapsed = time.time() - t0
    dataset_note = (
        "the parquet 'validation' split's row count/schema matches FEVER's official "
        "labelled_dev (19,998 claims, flattened one-row-per-evidence-sentence)"
    )
    sampling_method = (
        "label-stratified random sample, fixed seed, claims with evidence preferred "
        "for SUPPORTS/REFUTES"
    )
    wiki_note = (
        "FEVER's own wiki-pages.zip snapshot was tried first (fsspec random-access zip "
        "over HTTP range requests) but measured sandbox bandwidth to fever.ai (~150KB/s) "
        "made scanning it infeasible; see README."
    )
    evidence_recall_note = (
        "fraction of claims (with gold evidence) for which at least one retrieved "
        "sentence's Wikipedia title matched a gold evidence title; exact gold "
        "sentence-ID matching is not meaningful since evidence text comes from current "
        "Wikipedia, not FEVER's 2017 snapshot"
    )
    results = {
        "dataset": {
            "source": (
                "huggingface fever/fever (revision refs/convert/parquet), "
                "config default, split validation"
            ),
            "note": dataset_note,
            "sampling": {
                "method": sampling_method,
                "seed": seed,
                "n_per_label_requested": n_per_label,
                "n_claims_sampled": len(claims),
                "label_counts": label_counts,
            },
        },
        "wiki_pages": {
            "source": (
                "live Wikipedia (en.wikipedia.org MediaWiki API, "
                "action=query&prop=extracts&exintro=1)"
            ),
            "note": wiki_note,
            "target_titles": len(target_titles),
            "pages_fetched": len(wiki_pages),
            "pool_sentences": len(pool),
        },
        "models": {"chat_model": chat_model, "embed_model": embed_model},
        "retrieval": {"top_k": top_k},
        "scoring": {
            "n_claims": len(claims),
            "accuracy": accuracy,
            "correct": correct,
            "per_label_accuracy": per_label_accuracy,
            "confusion_matrix": confusion,
            "evidence_title_recall": evidence_recall,
            "evidence_title_recall_eligible_claims": evidence_eligible,
            "evidence_title_recall_note": evidence_recall_note,
        },
        "elapsed_seconds": round(elapsed, 1),
        "per_claim": per_claim_results,
    }
    return results
