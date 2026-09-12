"""Fetch real instances from princeton-nlp/SWE-bench_Lite (HF `datasets`) and
cache the ones we actually work with to a local JSON file so re-runs don't
re-hit the Hugging Face Hub.

Usage:
    uv run python projects/04_swebench_coding_agent/dataset.py --survey
    uv run python projects/04_swebench_coding_agent/dataset.py --fetch pallets__flask-4045 psf__requests-3362
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_FILE = DATA_DIR / "instances.json"

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
SPLIT = "test"


def _load_hf_dataset():
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, split=SPLIT)


def survey() -> None:
    """Print a per-repo instance count, to pick lightweight repos by eye."""
    ds = _load_hf_dataset()
    counts = Counter(r["repo"] for r in ds)
    print(f"{len(ds)} instances total in {DATASET_NAME}:{SPLIT}\n")
    for repo, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{n:4d}  {repo}")


def fetch(instance_ids: list[str]) -> list[dict]:
    """Pull specific instances by instance_id and cache them locally."""
    ds = _load_hf_dataset()
    wanted = set(instance_ids)
    found = [dict(r) for r in ds if r["instance_id"] in wanted]

    missing = wanted - {r["instance_id"] for r in found}
    if missing:
        raise SystemExit(f"instance_id(s) not found in dataset: {sorted(missing)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_cached()
    by_id = {r["instance_id"]: r for r in existing}
    for r in found:
        by_id[r["instance_id"]] = r
    CACHE_FILE.write_text(json.dumps(list(by_id.values()), indent=2), encoding="utf-8")
    print(f"cached {len(found)} instance(s) -> {CACHE_FILE}")
    return found


def load_cached() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def get_instance(instance_id: str) -> dict:
    for r in load_cached():
        if r["instance_id"] == instance_id:
            return r
    raise KeyError(
        f"{instance_id} not in local cache ({CACHE_FILE}); run --fetch first"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true", help="print per-repo instance counts")
    ap.add_argument("--fetch", nargs="+", metavar="INSTANCE_ID", help="fetch+cache these instances")
    args = ap.parse_args()

    if args.survey:
        survey()
    elif args.fetch:
        fetch(args.fetch)
    else:
        ap.print_help()
