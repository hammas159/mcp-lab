"""Very small "RAG for code edits" step: given a SWE-bench problem statement
and a real repo checkout, guess which source file(s) are relevant and return
their contents so the patch-generation prompt has real code to edit instead
of hallucinating file layout from scratch.

This is intentionally simple (keyword scoring, not embeddings) — the goal is
a real, inspectable retrieval step feeding a real local LLM, not a polished
search system.
"""

from __future__ import annotations

import re
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "this", "that", "these", "those",
    "and", "or", "but", "for", "with", "from", "into", "when", "should",
    "have", "has", "had", "not", "already", "added", "well", "since",
    "every", "significant", "can", "be", "in", "on", "of", "to", "as",
    "it", "its", "if", "so", "i", "my", "am", "are", "at", "by",
}


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    out = []
    for t in tokens:
        low = t.lower()
        if low in STOPWORDS:
            continue
        # Prefer identifier-ish tokens: snake_case, CamelCase, or long words.
        if "_" in t or (t[0].isupper() and not t.isupper()) or len(t) >= 5:
            out.append(t)
    return out


def find_relevant_files(
    repo_path: Path,
    problem_statement: str,
    *,
    max_files: int = 2,
    max_bytes_per_file: int = 12_000,
) -> list[tuple[str, str]]:
    """Grep the checkout for problem-statement keywords, rank .py files by
    hit count, and return [(relative_path, file_content), ...] for the top
    matches (source files only — tests/, docs/, examples/ excluded so we
    don't accidentally hand the model the answer key)."""
    keywords = _keywords(problem_statement)
    if not keywords:
        return []

    pattern = re.compile("|".join(re.escape(k) for k in set(keywords)))

    scores: dict[Path, int] = {}
    for py_file in repo_path.rglob("*.py"):
        rel = py_file.relative_to(repo_path)
        parts = rel.parts
        if any(p in ("tests", "test", "docs", "examples", ".tox") for p in parts):
            continue
        if ".git" in parts:
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = len(pattern.findall(text))
        if hits:
            scores[py_file] = hits

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:max_files]
    results = []
    for path, _hits in ranked:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_bytes_per_file:
            content = content[:max_bytes_per_file] + "\n# ... (truncated) ...\n"
        results.append((str(path.relative_to(repo_path)).replace("\\", "/"), content))
    return results
