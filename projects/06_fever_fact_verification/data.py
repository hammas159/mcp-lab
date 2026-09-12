"""Dataset plumbing for the FEVER fact-verification agent.

Two data sources are used, both real (no synthetic data):

1. Claims + gold labels + gold evidence pointers: the Hugging Face dataset
   ``fever/fever`` (config ``default``, split ``validation`` == the official
   FEVER "labelled_dev" / "paper_dev" split of 19,998 claims). Note: the
   dataset's own loading script ("fever.py") is no longer supported by
   current `datasets` versions (scripts were removed in `datasets` 4+), so
   we load it from the auto-generated `refs/convert/parquet` branch instead,
   which exposes the same rows without needing a script.

   The parquet rows are *flattened*: one row per (claim, evidence sentence)
   pair, so a claim needing several sentences appears as several rows
   sharing the same `id`. We regroup rows by claim id to recover, per
   claim, the list of gold "evidence sets" (each a list of
   (wikipedia_title, sentence_id) pairs) exactly as FEVER defines them.

2. The Wikipedia evidence corpus: we need the actual text of the small
   set of Wikipedia pages our sampled claims' gold evidence references
   (collected via `target_titles_for`, never the full 5.4M-row corpus).

   We first tried FEVER's official `wiki-pages.zip` (~1.7GB, 109 shard
   files) from fever.ai, opened as a *remote random-access* zip via
   `fsspec`'s HTTP filesystem (confirmed working: the server supports
   HTTP range requests, so we could list the central directory and open
   individual shards without downloading the whole archive). This was
   abandoned after measuring real throughput to fever.ai in this sandbox
   at ~150 KB/s — decoding even a single 53MB shard took ~113s, and the
   target titles are not alphabetically clustered, so a full scan could
   take hours. See the project README ("Problems hit") for the numbers.

   Instead we fetch the *current* live Wikipedia plaintext for the exact
   gold-evidence titles via the MediaWiki API (`action=query&prop=
   extracts&exintro=1`), batching up to 50 titles per request (MediaWiki
   only allows batching >1 title per call when the request is restricted
   to each page's lead section — asking for the *whole* article as plain
   text silently drops to one page per request for anonymous callers, an
   API limitation we hit and measured directly). This is still real,
   non-synthetic data and still requires real sentence-level retrieval
   (each page yields several-to-dozens of real sentences, most irrelevant
   to any given claim) — the tradeoffs are (a) only lead-section text is
   available, so gold evidence cited from later sections of a page won't
   appear in our pool, and (b) article text can have drifted since
   FEVER's 2017 Wikipedia snapshot, so exact gold sentence-ID matching is
   not meaningful; we use title-level evidence recall instead (see
   pipeline.py). Both are documented in the project README.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """Load a sibling module from this project's own directory by explicit file
    path, cached under a project-unique sys.modules key ("fever06_<name>").

    This project's folder is numeric-prefixed (06_...), which is not a valid
    package name, so sibling modules can't be reached via a normal dotted
    import. A bare `import config` would work too (after adding this
    directory to sys.path) but several sibling *-lab projects in this same
    repo use equally generic module names (config.py, data.py, ...); if two
    such modules ever load in the same Python process (e.g. a combined
    pytest run across projects), a bare import could silently resolve to the
    wrong project's module via the shared sys.modules cache. Keying by a
    project-unique prefix avoids that entirely.
    """
    key = f"fever06_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


config = _sibling("config")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _load_validation_rows() -> list[dict[str, Any]]:
    """Load the flattened FEVER validation rows via the parquet-converted branch."""
    from datasets import load_dataset

    ds = load_dataset(
        "fever/fever",
        split="validation",
        revision="refs/convert/parquet",
    )
    return ds.to_list()


def group_claims(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Regroup flattened FEVER rows into one record per claim.

    Each record: {id, claim, label, evidence_sets: [[[title, sent_id], ...], ...]}
    `evidence_sets` mirrors FEVER's semantics: each inner list is one
    independently-sufficient set of (title, sentence_id) evidence pairs.
    """
    by_id: dict[int, dict[str, Any]] = {}
    groups: dict[int, dict[int, list[list]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        cid = row["id"]
        if cid not in by_id:
            by_id[cid] = {"id": cid, "claim": row["claim"], "label": row["label"]}
        if row["evidence_id"] != -1 and row["evidence_wiki_url"]:
            ann = row["evidence_annotation_id"]
            groups[cid][ann].append([row["evidence_wiki_url"], row["evidence_sentence_id"]])

    for cid, rec in by_id.items():
        rec["evidence_sets"] = list(groups.get(cid, {}).values())
    return by_id


def sample_claims(
    n_per_label: int = config.N_CLAIMS_PER_LABEL,
    seed: int = config.SEED,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Sample a fixed-seed, label-stratified set of claims from FEVER validation.

    Cached to disk (config.CLAIMS_CACHE) so repeated runs are instant and
    reproducible across restarts.
    """
    import random

    if config.CLAIMS_CACHE.exists() and not force_refresh:
        cached = json.loads(config.CLAIMS_CACHE.read_text(encoding="utf-8"))
        if cached.get("n_per_label") == n_per_label and cached.get("seed") == seed:
            return cached["claims"]

    rows = _load_validation_rows()
    by_id = group_claims(rows)

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in by_id.values():
        by_label[rec["label"]].append(rec)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label in config.FEVER_LABELS:
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        # Prefer SUPPORTS/REFUTES claims that actually carry evidence (should be ~all of them);
        # NEI claims typically carry none, which is expected and part of the real signal.
        pool.sort(key=lambda r: 0 if r["evidence_sets"] else 1)
        sampled.extend(pool[:n_per_label])

    sampled.sort(key=lambda r: r["id"])

    config.CLAIMS_CACHE.write_text(
        json.dumps({"n_per_label": n_per_label, "seed": seed, "claims": sampled}, indent=2),
        encoding="utf-8",
    )
    return sampled


def target_titles_for(claims: list[dict[str, Any]]) -> set[str]:
    titles: set[str] = set()
    for rec in claims:
        for ev_set in rec["evidence_sets"]:
            for title, _sent_id in ev_set:
                titles.add(title)
    return titles


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str) -> list[list]:
    """Lightweight regex sentence splitter -> [[sent_id, sentence_text], ...].

    Not a proper NLP tokenizer (no abbreviation handling); adequate for
    building a real retrieval candidate pool. Documented as a known
    simplification in the README.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    parts = parts[: config.MAX_SENTENCES_PER_PAGE]
    return [[i, p] for i, p in enumerate(parts)]


def fetch_wiki_pages(
    target_titles: set[str],
    progress_cb=print,
    batch_size: int = 50,
) -> dict[str, list[list]]:
    """Fetch current live-Wikipedia plaintext for `target_titles`.

    Returns {title: [[sent_id, sentence_text], ...]}, batching MediaWiki
    `action=query&prop=extracts` calls (up to `batch_size` titles per
    request) and checkpointing to config.WIKI_PAGES_CACHE after every
    batch so a crash resumes instead of re-fetching everything.

    See the module docstring for why this replaced streaming FEVER's own
    wiki-pages.zip (measured sandbox bandwidth made that infeasible).
    """
    cache: dict[str, Any] = {"found": {}}
    if config.WIKI_PAGES_CACHE.exists():
        cache = json.loads(config.WIKI_PAGES_CACHE.read_text(encoding="utf-8"))
    found: dict[str, list[list]] = cache.get("found", {})

    missing = sorted(target_titles - found.keys())
    if not missing:
        progress_cb(f"[wiki] all {len(target_titles)} target pages already cached.")
        return {t: found[t] for t in target_titles if t in found}

    progress_cb(f"[wiki] fetching {len(missing)}/{len(target_titles)} pages from live Wikipedia...")
    headers = {"User-Agent": "mcp-lab-fever-agent/0.1 (local research project)"}
    with httpx.Client(timeout=30, headers=headers) as client:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            # FEVER encodes literal parentheses in titles as -LRB-/-RRB- (e.g.
            # "Tom_Baker_-LRB-English_actor-RRB-" -> "Tom Baker (English actor)");
            # MediaWiki only auto-normalizes underscores, not that encoding, so we
            # must convert *before* sending the request, not just when reading it back.
            orig_by_mw_title = {
                orig.replace("-LRB-", "(").replace("-RRB-", ")").replace("_", " "): orig
                for orig in batch
            }
            resp = client.get(
                WIKIPEDIA_API,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,  # MediaWiki only returns >1 page's *full* extract per
                                    # request for anonymous callers ("exlimit lowered to
                                    # 1"); intro-only extracts don't have that limit, so
                                    # this is what makes batching many titles feasible.
                    "redirects": 1,
                    "format": "json",
                    "titles": "|".join(orig_by_mw_title.keys()),
                },
            )
            resp.raise_for_status()
            data = resp.json().get("query", {})
            pages_by_title = {p.get("title"): p for p in data.get("pages", {}).values()}
            normalized = {m["from"]: m["to"] for m in data.get("normalized", [])}
            redirects = {m["from"]: m["to"] for m in data.get("redirects", [])}

            for mw_title, orig in orig_by_mw_title.items():
                step1 = normalized.get(mw_title, mw_title)
                resolved = redirects.get(step1, step1)
                page = pages_by_title.get(resolved)
                if not page or "missing" in page:
                    continue
                found[orig] = _split_sentences(page.get("extract", ""))

            got = sum(1 for t in batch if t in found)
            batch_num = start // batch_size + 1
            progress_cb(f"[wiki] batch {batch_num}: {got}/{len(batch)} pages resolved")
            config.WIKI_PAGES_CACHE.write_text(json.dumps({"found": found}), encoding="utf-8")
            time.sleep(0.2)  # be polite to the public Wikipedia API

    still_missing = target_titles - found.keys()
    if still_missing:
        sample = sorted(still_missing)[:10]
        progress_cb(f"[wiki] {len(still_missing)} titles never resolved: {sample}")

    return {t: found[t] for t in target_titles if t in found}
