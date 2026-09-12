"""Real embedding-based retrieval over the locally-collected Wikipedia sentence pool.

Sentences from every fetched page (across ALL sampled claims, not just one
claim's own gold page) form a single shared candidate pool. This matters:
it means retrieval for any one claim has to find the right sentences among
real distractors from unrelated pages, instead of trivially returning "the
one page we already know is relevant." Each claim is embedded and compared
by cosine similarity against every pooled sentence's embedding (both via
Ollama's `nomic-embed-text`), and the top-k sentences are returned as the
retrieved evidence handed to the verdict step.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

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


class PoolSentence(NamedTuple):
    title: str
    sent_id: int
    text: str


def build_pool(wiki_pages: dict[str, list[list]]) -> list[PoolSentence]:
    pool: list[PoolSentence] = []
    for title, sentences in wiki_pages.items():
        for sent_id, text in sentences:
            if text.strip():
                pool.append(PoolSentence(title, sent_id, text))
    return pool


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def _load_embed_cache() -> dict[str, list[float]]:
    if config.EMBED_CACHE.exists():
        return json.loads(config.EMBED_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_embed_cache(cache: dict[str, list[float]]) -> None:
    config.EMBED_CACHE.write_text(json.dumps(cache), encoding="utf-8")


def embed_texts(
    texts: list[str],
    model: str = config.EMBED_MODEL,
    client: httpx.Client | None = None,
    progress_cb=lambda *a: None,
) -> list[list[float]]:
    """Embed a list of texts via the local Ollama `/api/embeddings` endpoint.

    Disk-cached by (model, text) hash so re-runs (and crash-resumes) don't
    re-embed sentences already computed.
    """
    cache = _load_embed_cache()
    owns_client = client is None
    client = client or httpx.Client(base_url=config.OLLAMA_URL, timeout=60)
    try:
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        to_compute = []
        for i, t in enumerate(texts):
            key = _cache_key(model, t)
            if key in cache:
                out[i] = cache[key]
            else:
                to_compute.append((i, t, key))

        for n, (i, t, key) in enumerate(to_compute):
            resp = client.post("/api/embeddings", json={"model": model, "prompt": t})
            resp.raise_for_status()
            vec = resp.json()["embedding"]
            cache[key] = vec
            out[i] = vec
            if (n + 1) % 25 == 0 or n == len(to_compute) - 1:
                progress_cb(f"[embed] {n + 1}/{len(to_compute)} new embeddings computed")
                _save_embed_cache(cache)
        if to_compute:
            _save_embed_cache(cache)
        return out
    finally:
        if owns_client:
            client.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(
    claim_vec: list[float],
    pool: list[PoolSentence],
    pool_vecs: list[list[float]],
    k: int = config.TOP_K_EVIDENCE,
) -> list[tuple[PoolSentence, float]]:
    scored = [(pool[i], cosine_similarity(claim_vec, pool_vecs[i])) for i in range(len(pool))]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
