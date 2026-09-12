"""The verdict step: a local Ollama chat model reads a claim + retrieved
evidence sentences and decides SUPPORTS / REFUTES / NOT ENOUGH INFO.

Uses FEVER's own canonical label strings ("SUPPORTS", "REFUTES",
"NOT ENOUGH INFO") so model output can be compared directly against the
dataset's gold `label` field. The task brief's wording ("SUPPORTED /
REFUTED / NOT ENOUGH INFO") is treated as a synonym and normalized.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

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

SYSTEM_PROMPT = (
    "You are a fact-verification assistant. You will be given a CLAIM and a list of "
    "EVIDENCE sentences retrieved from Wikipedia. Decide whether the evidence, taken "
    "together, SUPPORTS the claim, REFUTES the claim, or provides NOT ENOUGH INFO to "
    "decide. Only use the given evidence -- do not use outside knowledge. If the "
    "evidence does not mention the claim's subject at all, or does not clearly confirm "
    "or contradict it, answer NOT ENOUGH INFO.\n\n"
    "Respond with exactly one line in this format:\n"
    "VERDICT: <SUPPORTS|REFUTES|NOT ENOUGH INFO>\n"
    "Then a second line with a one-sentence REASON."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)", re.IGNORECASE)

_LABEL_ALIASES = {
    "SUPPORTS": "SUPPORTS",
    "SUPPORTED": "SUPPORTS",
    "SUPPORT": "SUPPORTS",
    "REFUTES": "REFUTES",
    "REFUTED": "REFUTES",
    "REFUTE": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    "NOTENOUGHINFO": "NOT ENOUGH INFO",
    "NEI": "NOT ENOUGH INFO",
    "INSUFFICIENT INFO": "NOT ENOUGH INFO",
    "INSUFFICIENT INFORMATION": "NOT ENOUGH INFO",
}


def normalize_label(raw: str) -> str | None:
    """Map arbitrary model text to one of FEVER's 3 canonical labels, or None."""
    if not raw:
        return None
    m = _VERDICT_RE.search(raw)
    candidate = m.group(1).upper() if m else raw.strip().upper()
    candidate = re.sub(r"\s+", " ", candidate).strip(" .!\"'")
    return _LABEL_ALIASES.get(candidate)


def build_user_prompt(claim: str, evidence: list[tuple[str, str]]) -> str:
    if evidence:
        ev_lines = "\n".join(f"- ({title}) {text}" for title, text in evidence)
    else:
        ev_lines = "(no evidence sentences were retrieved)"
    return f"CLAIM: {claim}\n\nEVIDENCE:\n{ev_lines}"


def get_verdict(
    claim: str,
    evidence: list[tuple[str, str]],
    model: str = config.CHAT_MODEL,
    client: httpx.Client | None = None,
) -> dict:
    """Call the local Ollama chat model and return {label, raw_response}.

    `label` is one of FEVER's canonical strings, or "PARSE_ERROR" if the
    model's output couldn't be normalized to one of them.
    """
    owns_client = client is None
    client = client or httpx.Client(base_url=config.OLLAMA_URL, timeout=120)
    try:
        resp = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(claim, evidence)},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        label = normalize_label(raw) or "PARSE_ERROR"
        return {"label": label, "raw_response": raw}
    finally:
        if owns_client:
            client.close()
