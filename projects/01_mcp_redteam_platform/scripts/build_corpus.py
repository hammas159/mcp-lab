"""Fetch a real HotpotQA slice and lay it out as the search_docs corpus.

Run once with the repo-root venv (it already has `datasets` installed):
    uv run python projects/01_mcp_redteam_platform/scripts/build_corpus.py

Produces two SEPARATE roots on disk, which is the point of this script:
  data/corpus/<id>.txt        - one file per context paragraph, what
                                 search_docs is *supposed* to be scoped to.
  data/eval/qa/<qid>.json     - per-question {question, answer} fixtures used
                                 to SCORE the eval run. This is the realistic
                                 leak vector: each file's text contains the
                                 same question wording the agent is asked, so
                                 a semantic search for that question has high
                                 similarity to its own answer-labelled fixture
                                 if this directory is ever inside a retrieval
                                 tool's root.
  data/eval/answer_key.json   - the same mapping collapsed into one file, kept
                                 only as a convenience for scoring code.
  data/eval/questions.json    - the questions only (no answers) - what the
                                 agent is actually asked during the eval run.

Module 1 of the red-team audit (answer leakage) exists because it is easy to
misconfigure a retrieval tool's root directory to be the *parent* of
data/corpus/ and data/eval/ instead of data/corpus/ alone.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent.parent
CORPUS_DIR = HERE / "data" / "corpus"
EVAL_DIR = HERE / "data" / "eval"
SAMPLE_SIZE = 60
SEED = 42


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    qa_dir = EVAL_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(SEED)
    indices = rng.sample(range(len(ds)), SAMPLE_SIZE)

    answer_key: dict[str, str] = {}
    questions: list[dict] = []
    doc_count = 0
    seen_titles: set[str] = set()

    for idx in indices:
        row = ds[idx]
        qid = row["id"]
        answer_key[qid] = row["answer"]
        questions.append({"id": qid, "question": row["question"]})
        (qa_dir / f"{qid}.json").write_text(
            json.dumps({"id": qid, "question": row["question"], "answer": row["answer"]}, indent=2),
            encoding="utf-8",
        )

        titles = row["context"]["title"]
        sentences_lists = row["context"]["sentences"]
        for title, sentences in zip(titles, sentences_lists, strict=True):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            doc_count += 1
            safe_name = "".join(c if c.isalnum() else "_" for c in title)[:80]
            path = CORPUS_DIR / f"{doc_count:04d}_{safe_name}.txt"
            path.write_text(f"{title}\n\n" + " ".join(sentences), encoding="utf-8")

    (EVAL_DIR / "answer_key.json").write_text(
        json.dumps(answer_key, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "questions.json").write_text(
        json.dumps(questions, indent=2), encoding="utf-8"
    )

    print(f"Sampled {SAMPLE_SIZE} HotpotQA validation questions (seed={SEED}).")
    print(f"Wrote {doc_count} unique context documents to {CORPUS_DIR}")
    print(f"Wrote answer key ({len(answer_key)} entries) to {EVAL_DIR / 'answer_key.json'}")


if __name__ == "__main__":
    main()
