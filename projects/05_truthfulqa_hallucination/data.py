"""Dataset loading and sampling for the real TruthfulQA benchmark (multiple_choice config).

Source: Hugging Face dataset `truthfulqa/truthful_qa` (the modern namespaced id for the
dataset formerly loadable as `truthful_qa` — the old bare name fails to resolve with
current `datasets`/`huggingface_hub` versions, see README "Problems hit" section).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

DATASET_PATH = "truthfulqa/truthful_qa"
DATASET_CONFIG = "multiple_choice"
DATASET_SPLIT = "validation"  # the only split TruthfulQA ships
DEFAULT_SEED = 42


@dataclass
class MCQuestion:
    qid: int  # row index into the full 817-row validation split
    question: str
    mc1_choices: list[str]
    mc1_labels: list[int]
    mc2_choices: list[str]
    mc2_labels: list[int]


def load_truthfulqa_mc():
    """Load the real truthful_qa `multiple_choice` validation split (817 rows) from HF."""
    return load_dataset(DATASET_PATH, DATASET_CONFIG)[DATASET_SPLIT]


def sample_questions(n: int, seed: int = DEFAULT_SEED) -> list[MCQuestion]:
    """Deterministically sample `n` questions (by row index) from the 817-row split.

    Uses `random.Random(seed).sample` over the row indices so the exact sample is
    reproducible given the same seed, and is documented (question_ids) in results.json.
    """
    ds = load_truthfulqa_mc()
    total = len(ds)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(n, total)))
    out = []
    for idx in indices:
        row = ds[idx]
        out.append(
            MCQuestion(
                qid=idx,
                question=row["question"],
                mc1_choices=list(row["mc1_targets"]["choices"]),
                mc1_labels=list(row["mc1_targets"]["labels"]),
                mc2_choices=list(row["mc2_targets"]["choices"]),
                mc2_labels=list(row["mc2_targets"]["labels"]),
            )
        )
    return out
