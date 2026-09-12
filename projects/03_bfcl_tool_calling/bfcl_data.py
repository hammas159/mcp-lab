"""Fetch and load real Berkeley Function-Calling Leaderboard (BFCL) v4 test cases.

Source: the public, Apache-licensed ``ShishirPatil/gorilla`` GitHub repo, raw JSON
files under ``berkeley-function-call-leaderboard/bfcl_eval/data/``. No API key is
needed -- these are static JSON files fetched over HTTPS (data, not an LLM call).

Categories used in this project are the non-executable / AST-scored categories
that need no live external API backend:

  - ``simple_python``     -- one function offered, one call expected
  - ``multiple``           -- several functions offered, one call expected
  - ``parallel``            -- one function offered, several calls expected
  - ``parallel_multiple``  -- several functions offered, several calls expected

Executable categories (which require calling a real backend API) and multi-turn
categories are intentionally skipped -- see the project README for why.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple"]

DATA_DIR = Path(__file__).parent / "data"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def fetch_category(category: str) -> tuple[Path, Path]:
    """Download the question file and possible_answer (ground truth) file for a category.

    Cached under ``data/`` so repeated runs don't re-hit the network.
    """
    q_path = DATA_DIR / f"BFCL_v4_{category}.json"
    a_path = DATA_DIR / f"BFCL_v4_{category}_answer.json"
    _download(f"{RAW_BASE}/BFCL_v4_{category}.json", q_path)
    _download(f"{RAW_BASE}/possible_answer/BFCL_v4_{category}.json", a_path)
    return q_path, a_path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_category(category: str) -> list[dict]:
    """Return a list of merged test cases for a category.

    Each item: {"id", "question" (flattened list of chat turns), "function" (list
    of BFCL tool schemas), "ground_truth" (list of {fn_name: {param: [accepted values]}})}.
    """
    q_path, a_path = fetch_category(category)
    questions = {row["id"]: row for row in _load_jsonl(q_path)}
    answers = {row["id"]: row for row in _load_jsonl(a_path)}

    merged = []
    for case_id, q in questions.items():
        a = answers.get(case_id)
        if a is None:
            continue
        # question is [[{role, content}, ...]] -- flatten all inner turn-lists in order.
        turns: list[dict] = []
        for turn_group in q["question"]:
            turns.extend(turn_group)
        merged.append(
            {
                "id": case_id,
                "category": category,
                "question": turns,
                "function": q["function"],
                "ground_truth": a["ground_truth"],
            }
        )
    # Stable order by the numeric suffix of the id (e.g. simple_python_12).
    def _sort_key(item: dict) -> int:
        try:
            return int(item["id"].rsplit("_", 1)[-1])
        except ValueError:
            return 0

    merged.sort(key=_sort_key)
    return merged
