"""Tests for projects/05_truthfulqa_hallucination.

The project folder is named with a numeric prefix (05_truthfulqa_hallucination), which is
not a valid Python identifier, so its modules cannot be reached with a normal
`import projects.05_truthfulqa_hallucination.foo` statement. We load each module directly
from its file path with importlib instead.

Tests that only need pure Python (prompt parsing, scoring math, shuffling) always run.
Tests that need the real TruthfulQA dataset from Hugging Face are skipped if the dataset
can't be fetched (e.g. no network / not yet cached). Tests that need a live local Ollama
server are marked `@pytest.mark.live`, matching this repo's existing marker convention.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent / "projects" / "05_truthfulqa_hallucination"


def _load(module_name: str, filename: str):
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


prompts = _load("tqa_prompts", "prompts.py")
scoring = _load("tqa_scoring", "scoring.py")
data = _load("tqa_data", "data.py")
ollama_client = _load("tqa_ollama_client", "ollama_client.py")


# ---------------------------------------------------------------------------
# scoring.py
# ---------------------------------------------------------------------------


def test_score_mc1_correct():
    assert scoring.score_mc1(1, [0, 1, 0]) == 1


def test_score_mc1_incorrect():
    assert scoring.score_mc1(0, [0, 1, 0]) == 0


def test_score_mc1_unparsed_response_counts_as_wrong():
    assert scoring.score_mc1(None, [0, 1, 0]) == 0


def test_score_mc1_out_of_range_counts_as_wrong():
    assert scoring.score_mc1(99, [0, 1, 0]) == 0


def test_score_mc2_perfect_selection():
    labels = [1, 1, 0, 0]
    assert scoring.score_mc2({0, 1}, labels) == 1.0


def test_score_mc2_no_selection_scores_zero():
    labels = [1, 1, 0, 0]
    assert scoring.score_mc2(set(), labels) == 0.0


def test_score_mc2_only_false_positives_scores_zero():
    labels = [1, 0, 0]
    assert scoring.score_mc2({1, 2}, labels) == 0.0


def test_score_mc2_partial_credit_between_bounds():
    labels = [1, 1, 0, 0]
    score = scoring.score_mc2({0}, labels)  # found 1 of 2 true, no false positives
    assert 0.0 < score < 1.0
    assert score == pytest.approx(0.5)


def test_score_mc2_no_false_options_is_pure_recall():
    labels = [1, 1]
    assert scoring.score_mc2({0, 1}, labels) == 1.0
    assert scoring.score_mc2({0}, labels) == pytest.approx(0.5)


def test_score_mc2_no_true_labels_defined_scores_zero():
    # degenerate case: should never happen in real TruthfulQA rows, but must not crash.
    assert scoring.score_mc2({0}, [0, 0]) == 0.0


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------


def test_shuffled_options_is_a_permutation():
    choices = ["a", "b", "c", "d"]
    labels = [1, 0, 0, 0]
    shuffled_choices, shuffled_labels = prompts.shuffled_options(choices, labels, seed=7)
    assert sorted(shuffled_choices) == sorted(choices)
    assert sorted(shuffled_labels) == sorted(labels)
    # the correct choice's label must stay attached to the correct choice after shuffling
    correct = choices[labels.index(1)]
    assert shuffled_labels[shuffled_choices.index(correct)] == 1


def test_shuffled_options_is_deterministic_given_seed():
    choices = ["a", "b", "c", "d", "e"]
    labels = [0, 0, 1, 0, 0]
    r1 = prompts.shuffled_options(choices, labels, seed=123)
    r2 = prompts.shuffled_options(choices, labels, seed=123)
    assert r1 == r2


def test_build_mc1_prompt_contains_question_and_options():
    p = prompts.build_mc1_prompt("Is the sky blue?", ["Yes", "No"], "zero_shot")
    assert "Is the sky blue?" in p
    assert "A. Yes" in p
    assert "B. No" in p


def test_build_prompt_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        prompts.build_mc1_prompt("Q", ["A", "B"], "not_a_real_strategy")
    with pytest.raises(ValueError):
        prompts.build_mc2_prompt("Q", ["A", "B"], "not_a_real_strategy")


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Answer: B", 1),
        ("The answer is B.", 1),
        ("Some reasoning...\nAnswer: C", 2),
        ("B", 1),
        ("no letter here", None),
    ],
)
def test_parse_letter(response, expected):
    assert prompts.parse_letter(response, num_options=4) == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Answer: A,C", {0, 2}),
        ("Answer: A, C, D", {0, 2, 3}),
        ("I think A and C are true.\nAnswer: A,C", {0, 2}),
        ("nothing parseable in lowercase only", set()),
    ],
)
def test_parse_letters(response, expected):
    assert prompts.parse_letters(response, num_options=4) == expected


# ---------------------------------------------------------------------------
# data.py (real Hugging Face dataset — skipped if it can't be fetched/cached)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    try:
        return data.sample_questions(5, seed=42)
    except Exception as exc:  # network / HF hub unavailable
        pytest.skip(f"truthfulqa/truthful_qa dataset unavailable: {exc}")


def test_sample_questions_returns_requested_count(sample):
    assert len(sample) == 5


def test_sample_questions_have_mc1_and_mc2_targets(sample):
    for q in sample:
        assert q.question
        assert len(q.mc1_choices) == len(q.mc1_labels)
        assert len(q.mc2_choices) == len(q.mc2_labels)
        assert sum(q.mc1_labels) == 1  # MC1 always has exactly one correct answer
        assert sum(q.mc2_labels) >= 1  # MC2 has at least one correct answer


def test_sample_questions_is_deterministic_given_seed():
    try:
        a = data.sample_questions(5, seed=42)
        b = data.sample_questions(5, seed=42)
    except Exception as exc:
        pytest.skip(f"truthfulqa/truthful_qa dataset unavailable: {exc}")
    assert [q.qid for q in a] == [q.qid for q in b]


# ---------------------------------------------------------------------------
# ollama_client.py (requires a live local Ollama server)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_list_models_reaches_local_ollama():
    models = ollama_client.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


@pytest.mark.live
def test_chat_capable_models_excludes_embedding_only():
    names = ollama_client.chat_capable_models()
    assert isinstance(names, list)
    assert all("embed" not in n for n in names)


@pytest.mark.live
def test_generate_returns_text():
    names = ollama_client.chat_capable_models()
    if not names:
        pytest.skip("no chat-capable Ollama models pulled locally")
    text = ollama_client.generate(names[0], "Reply with exactly the word: pong", num_predict=5)
    assert isinstance(text, str)
    assert len(text) > 0
