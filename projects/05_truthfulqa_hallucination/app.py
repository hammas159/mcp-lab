"""Minimal FastAPI + Jinja2 page showing the TruthfulQA hallucination-rate comparison table.

Reads projects/05_truthfulqa_hallucination/results.json (produced by run_eval.py) and
renders it as an HTML table: one row per model, columns for MC1 accuracy / hallucination
rate and MC2 score, split by prompting strategy (zero_shot vs. cot).

Run directly (the numeric folder prefix means it can't be imported as
`projects.05_truthfulqa_hallucination.app`, so `uvicorn app:module` dotted paths don't
work either — run the file itself):

    uv run python projects/05_truthfulqa_hallucination/app.py

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(title="TruthfulQA Hallucination Rate — Local Ollama Fleet")


def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def build_rows(summary: dict) -> list[dict]:
    """Flatten results.json into one row per (model, strategy) for the template."""
    rows = []
    models = summary["models"]
    strategies = summary["strategies"]
    results = summary["results"]
    for model in models:
        for strategy in strategies:
            mc1 = results.get("mc1", {}).get(model, {}).get(strategy)
            mc2 = results.get("mc2", {}).get(model, {}).get(strategy)
            acc = mc1["accuracy"] if mc1 else None
            mc2_score = mc2["mc2_score"] if mc2 else None
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "mc1_accuracy": acc,
                    "mc1_hallucination_rate": (1 - acc) if acc is not None else None,
                    "mc1_n": mc1["n"] if mc1 else None,
                    "mc2_score": mc2_score,
                    "mc2_n": mc2["n"] if mc2 else None,
                }
            )
    return rows


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    summary = load_results()
    if summary is None:
        return TEMPLATES.TemplateResponse(request, "index.html", {"summary": None, "rows": []})
    rows = build_rows(summary)
    return TEMPLATES.TemplateResponse(request, "index.html", {"summary": summary, "rows": rows})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
