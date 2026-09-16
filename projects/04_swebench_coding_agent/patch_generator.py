"""Call a local Ollama coding model to produce a real unified-diff patch for
a real SWE-bench-Lite instance, using retrieved source context.

No API keys, no external LLM calls: Ollama only, local.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Overridable so the 3B baseline and a larger model can both be reproduced from the
# same code. The default stays on the model the committed baseline was measured with.
MODEL = os.environ.get("MCP_LAB_CODE_MODEL", "qwen2.5-coder:3b")

SYSTEM_PROMPT = """You are an expert software engineer fixing a real bug in an open-source \
Python project. You will be given the GitHub issue text and the full contents of the \
source file(s) most likely to need a change.

Respond with ONLY a unified diff (git patch format) that fixes the issue. Rules:
- Use real `--- a/<path>` / `+++ b/<path>` headers with the exact relative paths given.
- Use correct @@ hunk headers with accurate line numbers matching the shown file content.
- Make the smallest change that plausibly fixes the described bug.
- Do not include explanations, markdown fences, or any text outside the diff.
"""


@dataclass
class PatchAttempt:
    instance_id: str
    model: str
    prompt: str
    raw_response: str
    patch_text: str  # extracted diff, possibly empty if extraction failed


def build_prompt(problem_statement: str, files: list[tuple[str, str]]) -> str:
    parts = [f"GitHub issue:\n{problem_statement.strip()}\n"]
    for path, content in files:
        parts.append(f"\n--- current contents of {path} ---\n{content}\n")
    parts.append(
        "\nProduce a unified diff patch (git format) against the file(s) above "
        "that fixes the issue described. Output ONLY the diff."
    )
    return "\n".join(parts)


_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.DOTALL)


def extract_diff(raw: str) -> str:
    """Pull a unified diff out of a model response, tolerating markdown fences
    or leading/trailing chatter."""
    fenced = _DIFF_FENCE_RE.search(raw)
    text = fenced.group(1) if fenced else raw

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("--- ") or line.startswith("diff --git"):
            start = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:]).strip() + "\n"


def generate_patch(
    instance_id: str,
    problem_statement: str,
    files: list[tuple[str, str]],
    *,
    model: str = MODEL,
    timeout: float = 600.0,
) -> PatchAttempt:
    prompt = build_prompt(problem_statement, files)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            # Ollama defaults num_ctx to 4096, which the two retrieved source
            # files (up to ~12KB each) plus the issue text can blow past --
            # when that happens generation falls back to slow context-shifting
            # instead of erroring, which is what caused multi-minute timeouts.
            # 16384 comfortably fits our prompt + response on this model.
            "options": {"temperature": 0.1, "num_ctx": 16384},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    diff = extract_diff(raw)
    return PatchAttempt(
        instance_id=instance_id,
        model=model,
        prompt=full_prompt,
        raw_response=raw,
        patch_text=diff,
    )


def save_attempt(attempt: PatchAttempt, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{attempt.instance_id}.prompt.txt").write_text(attempt.prompt, encoding="utf-8")
    (out_dir / f"{attempt.instance_id}.raw_response.txt").write_text(
        attempt.raw_response, encoding="utf-8"
    )
    # newline="\n": keep the diff in real unified-diff form (LF line endings)
    # so it can be `git apply`-ed as-is -- Windows text mode would otherwise
    # rewrite it to CRLF and break patch application.
    (out_dir / f"{attempt.instance_id}.model_patch.diff").write_text(
        attempt.patch_text, encoding="utf-8", newline="\n"
    )
