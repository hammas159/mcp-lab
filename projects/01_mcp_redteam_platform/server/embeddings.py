"""Thin wrapper around Ollama's embedding endpoint, used by search_docs.

Every audit module spawns fresh MCPAgentSession/ToolSet instances per trial
(each is a real, separate MCP subprocess, deliberately — that's what makes
the measurement honest), which would otherwise mean re-embedding the same
~600-document corpus from scratch, via a real network call per document,
on every single trial. A small on-disk cache keyed by (model, file content
hash) avoids that: the corpus is static, so its embeddings are too.
"""

import hashlib
import json
import math
from pathlib import Path

import httpx

from config import EMBED_MODEL, OLLAMA_HOST, ROOT

_CACHE_PATH = ROOT / "data" / ".embeddings_cache.json"
_cache: dict[str, list[float]] | None = None


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        if _CACHE_PATH.exists():
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def embed(text: str) -> list[float]:
    cache = _load_cache()
    key = f"{EMBED_MODEL}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    if key in cache:
        return cache[key]

    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    vector = resp.json()["embedding"]
    cache[key] = vector
    _save_cache()
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
