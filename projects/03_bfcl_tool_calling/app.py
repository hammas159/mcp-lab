"""Minimal FastAPI + Jinja2 viewer for the BFCL tool-calling results.json.

Run from repo root::

    uv run uvicorn projects.03_bfcl_tool_calling.app:app --reload

(Note: because the folder name starts with a digit, that dotted uvicorn target
won't import via a plain `import` statement either -- run it standalone instead::

    uv run python projects/03_bfcl_tool_calling/app.py

which starts uvicorn programmatically on http://127.0.0.1:8003.)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "results.json"

app = FastAPI(title="BFCL Tool-Calling Accuracy")
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))


def _load_results() -> dict:
    if not RESULTS_PATH.exists():
        return {"meta": {}, "per_model": {}}
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@app.get("/")
def index(request: Request):
    data = _load_results()
    per_model = data.get("per_model", {})
    categories = sorted({cat for res in per_model.values() for cat in res.get("by_category", {})})
    rows = []
    for model, res in sorted(
        per_model.items(), key=lambda kv: kv[1].get("overall_accuracy") or 0, reverse=True
    ):
        row = {
            "model": model,
            "overall_accuracy": res.get("overall_accuracy"),
            "overall_n": res.get("overall_n"),
            "by_category": res.get("by_category", {}),
        }
        rows.append(row)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "meta": data.get("meta", {}),
            "categories": categories,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8003)
