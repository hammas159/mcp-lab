"""Minimal FastAPI + Jinja2 page showing example FEVER claim/evidence/verdict results.

Reads projects/06_fever_fact_verification/results.json (produced by run_eval.py) and
renders the summary scores (accuracy, confusion matrix, evidence recall) plus a table of
every evaluated claim: gold label, predicted label, retrieved evidence sentences, and the
raw model response.

Run directly (the numeric folder prefix means it can't be imported as
`projects.06_fever_fact_verification.app`, so `uvicorn app:module` dotted paths don't work
either -- run the file itself):

    uv run python projects/06_fever_fact_verification/app.py

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

app = FastAPI(title="FEVER Fact-Verification Agent")


def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    results = load_results()
    return TEMPLATES.TemplateResponse(request, "index.html", {"results": results})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
