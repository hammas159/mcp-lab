"""Shared helpers for the 5 audit modules."""

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def save_results(module: str, data: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{module}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
