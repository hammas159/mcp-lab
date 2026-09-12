"""Tests for projects/06_fever_fact_verification.

The project folder is named with a numeric prefix (06_fever_fact_verification), which is
not a valid Python identifier, so its modules cannot be reached with a normal
`import projects.06_fever_fact_verification.foo` statement. We load each module directly
from its file path with importlib instead (matching tests/test_05_truthfulqa.py).

Tests that only need pure Python (label normalization, cosine similarity, sentence
splitting, evidence grouping/scoring math) always run. Tests that need the real FEVER
dataset from Hugging Face or live Wikipedia are skipped if they can't be fetched (e.g. no
network / not yet cached). Tests that need a live local Ollama server are marked
`@pytest.mark.live`, matching this repo's existing marker convention.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent / "projects" / "06_fever_fact_verification"


def _load(name: str):
    """Load projects/06.../<name>.py under the SAME project-unique sys.modules key
    ("fever06_<name>") that the project's own modules use for their internal
    sibling imports (see data.py's `_sibling` docstring). Loading in dependency
    order (config, then data/retrieval/verdict, then pipeline) means each
    module's own `_sibling("config")` call finds this exact object already
    cached and reuses it -- so e.g. `monkeypatch.setattr(config, ...)` here
    actually affects data.py's view of config too, not just this file's.
    """
    key = f"fever06_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, PROJECT_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


config = _load("config")
data = _load("data")
retrieval = _load("retrieval")
verdict = _load("verdict")
pipeline = _load("pipeline")


# ---------------------------------------------------------------------------
# verdict.py -- label normalization (pure Python, always runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("VERDICT: SUPPORTS\nREASON: because evidence says so.", "SUPPORTS"),
        ("VERDICT: REFUTES\nthe evidence contradicts it", "REFUTES"),
        ("VERDICT: NOT ENOUGH INFO\nunclear", "NOT ENOUGH INFO"),
        ("verdict: supports", "SUPPORTS"),
        ("SUPPORTED", "SUPPORTS"),
        ("Refuted.", "REFUTES"),
        ("NEI", "NOT ENOUGH INFO"),
    ],
)
def test_normalize_label_recognizes_variants(raw, expected):
    assert verdict.normalize_label(raw) == expected


def test_normalize_label_returns_none_for_gibberish():
    assert verdict.normalize_label("I cannot answer that question.") is None


def test_normalize_label_handles_empty_string():
    assert verdict.normalize_label("") is None


def test_build_user_prompt_includes_claim_and_evidence():
    prompt = verdict.build_user_prompt("The sky is blue.", [("Sky", "The sky appears blue.")])
    assert "The sky is blue." in prompt
    assert "Sky" in prompt
    assert "The sky appears blue." in prompt


def test_build_user_prompt_handles_no_evidence():
    prompt = verdict.build_user_prompt("Some claim.", [])
    assert "no evidence" in prompt.lower()


# ---------------------------------------------------------------------------
# retrieval.py -- cosine similarity + pool building (pure Python, always runs)
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert retrieval.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert retrieval.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert retrieval.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_zero_vector_without_crashing():
    assert retrieval.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_build_pool_flattens_pages_into_sentences():
    pages = {
        "Page_A": [[0, "First sentence."], [1, "Second sentence."]],
        "Page_B": [[0, "Only sentence."]],
    }
    pool = retrieval.build_pool(pages)
    assert len(pool) == 3
    titles = {p.title for p in pool}
    assert titles == {"Page_A", "Page_B"}


def test_build_pool_skips_blank_sentences():
    pages = {"Page_A": [[0, "Real sentence."], [1, "   "], [2, ""]]}
    pool = retrieval.build_pool(pages)
    assert len(pool) == 1


def test_retrieve_top_k_returns_most_similar_first():
    pool = [
        retrieval.PoolSentence("A", 0, "x"),
        retrieval.PoolSentence("B", 0, "y"),
        retrieval.PoolSentence("C", 0, "z"),
    ]
    pool_vecs = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    claim_vec = [1.0, 0.0]
    top = retrieval.retrieve_top_k(claim_vec, pool, pool_vecs, k=2)
    assert [sent.title for sent, _score in top] == ["A", "C"]


# ---------------------------------------------------------------------------
# data.py -- pure parsing helpers (always run)
# ---------------------------------------------------------------------------


def test_split_sentences_splits_on_terminal_punctuation():
    sents = data._split_sentences("First sentence. Second sentence! Third one?")
    texts = [t for _i, t in sents]
    assert texts == ["First sentence.", "Second sentence!", "Third one?"]


def test_split_sentences_handles_empty_text():
    assert data._split_sentences("") == []
    assert data._split_sentences(None) == []


def test_split_sentences_respects_max_sentences_cap(monkeypatch):
    monkeypatch.setattr(config, "MAX_SENTENCES_PER_PAGE", 2)
    text = "One. Two. Three. Four."
    sents = data._split_sentences(text)
    assert len(sents) == 2


def test_group_claims_reconstructs_evidence_sets():
    rows = [
        {"id": 1, "label": "SUPPORTS", "claim": "c1", "evidence_annotation_id": 10,
         "evidence_id": 100, "evidence_wiki_url": "Page_A", "evidence_sentence_id": 0},
        {"id": 1, "label": "SUPPORTS", "claim": "c1", "evidence_annotation_id": 10,
         "evidence_id": 101, "evidence_wiki_url": "Page_A", "evidence_sentence_id": 1},
        {"id": 2, "label": "NOT ENOUGH INFO", "claim": "c2", "evidence_annotation_id": 20,
         "evidence_id": -1, "evidence_wiki_url": "", "evidence_sentence_id": -1},
    ]
    grouped = data.group_claims(rows)
    assert grouped[1]["evidence_sets"] == [[["Page_A", 0], ["Page_A", 1]]]
    assert grouped[2]["evidence_sets"] == []
    assert grouped[2]["label"] == "NOT ENOUGH INFO"


def test_target_titles_for_collects_unique_titles_across_claims():
    claims = [
        {"evidence_sets": [[["Page_A", 0]], [["Page_B", 1]]]},
        {"evidence_sets": [[["Page_A", 2]]]},
        {"evidence_sets": []},
    ]
    assert data.target_titles_for(claims) == {"Page_A", "Page_B"}


# ---------------------------------------------------------------------------
# pipeline.py -- scoring math (pure Python, always runs)
# ---------------------------------------------------------------------------


def test_evidence_hit_true_when_titles_overlap():
    assert pipeline.evidence_hit({"Page_A", "Page_C"}, {"Page_A", "Page_B"}) is True


def test_evidence_hit_false_when_no_overlap():
    assert pipeline.evidence_hit({"Page_C"}, {"Page_A", "Page_B"}) is False


def test_evidence_hit_false_when_no_gold_titles():
    assert pipeline.evidence_hit({"Page_A"}, set()) is False


def test_build_confusion_matrix_counts_correctly():
    pairs = [
        ("SUPPORTS", "SUPPORTS"),
        ("SUPPORTS", "REFUTES"),
        ("REFUTES", "REFUTES"),
        ("NOT ENOUGH INFO", "PARSE_ERROR"),
    ]
    matrix = pipeline.build_confusion_matrix(pairs)
    assert matrix["SUPPORTS"]["SUPPORTS"] == 1
    assert matrix["SUPPORTS"]["REFUTES"] == 1
    assert matrix["REFUTES"]["REFUTES"] == 1
    assert matrix["NOT ENOUGH INFO"]["PARSE_ERROR"] == 1
    assert matrix["REFUTES"]["SUPPORTS"] == 0


def test_gold_titles_for_claim_flattens_evidence_sets():
    claim = {"evidence_sets": [[["Page_A", 0], ["Page_B", 1]], [["Page_C", 0]]]}
    assert pipeline.gold_titles_for_claim(claim) == {"Page_A", "Page_B", "Page_C"}


# ---------------------------------------------------------------------------
# data.py -- real FEVER dataset from Hugging Face (skipped if unreachable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def claim_sample():
    try:
        return data.sample_claims(n_per_label=2, seed=123, force_refresh=True)
    except Exception as exc:  # network / HF hub unavailable
        pytest.skip(f"fever/fever dataset unavailable: {exc}")


def test_sample_claims_returns_real_fever_labels(claim_sample):
    assert len(claim_sample) > 0
    for c in claim_sample:
        assert c["label"] in config.FEVER_LABELS
        assert isinstance(c["claim"], str) and c["claim"]


def test_sample_claims_is_deterministic_given_seed():
    try:
        a = data.sample_claims(n_per_label=2, seed=99, force_refresh=True)
        b = data.sample_claims(n_per_label=2, seed=99, force_refresh=False)
    except Exception as exc:
        pytest.skip(f"fever/fever dataset unavailable: {exc}")
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_supports_and_refutes_claims_carry_gold_evidence(claim_sample):
    for c in claim_sample:
        if c["label"] in ("SUPPORTS", "REFUTES"):
            assert len(c["evidence_sets"]) > 0, f"claim {c['id']} has no gold evidence"


# ---------------------------------------------------------------------------
# Live Ollama server required
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_embed_texts_returns_real_vectors():
    vecs = retrieval.embed_texts(["The sky is blue."], model=config.EMBED_MODEL)
    assert len(vecs) == 1
    assert len(vecs[0]) > 0
    assert isinstance(vecs[0][0], float)


@pytest.mark.live
def test_get_verdict_returns_a_canonical_label():
    result = verdict.get_verdict(
        "Simon Pegg is an actor.",
        [("Simon_Pegg", "Simon John Pegg is an English actor, comedian and screenwriter.")],
        model=config.CHAT_MODEL,
    )
    assert result["label"] in {*config.FEVER_LABELS, "PARSE_ERROR"}
    assert isinstance(result["raw_response"], str) and result["raw_response"]


@pytest.mark.live
def test_tiny_end_to_end_pipeline_runs_and_scores():
    results = pipeline.run_pipeline(n_per_label=1, seed=7, top_k=3, progress_cb=lambda *_a: None)
    assert results["scoring"]["n_claims"] >= 1
    assert 0.0 <= results["scoring"]["accuracy"] <= 1.0
    for row in results["per_claim"]:
        assert row["predicted_label"] in {*config.FEVER_LABELS, "PARSE_ERROR"}
