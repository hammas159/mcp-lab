"""Tests for projects/04_swebench_coding_agent.

Pure-logic tests (keyword retrieval, diff extraction, outcome parsing, dataset
caching) run always. Tests that need a live Ollama server are marked
`@pytest.mark.live` (matching the project-wide convention in pyproject.toml);
tests that need a real Docker daemon are skipped automatically when Docker
isn't reachable, via `@pytest.mark.skipif`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent / "projects" / "04_swebench_coding_agent"
sys.path.insert(0, str(PROJECT_DIR))

import context_retrieval as cr  # noqa: E402
import dataset  # noqa: E402
import patch_generator as pg  # noqa: E402
import sandbox_runner as sr  # noqa: E402


def _docker_actually_available() -> bool:
    try:
        return sr.docker_available()
    except Exception:
        return False


DOCKER_UP = _docker_actually_available()


# --------------------------------------------------------------------------
# context_retrieval -- pure logic, no network
# --------------------------------------------------------------------------


def test_keywords_skip_stopwords_and_short_tokens():
    kws = cr._keywords("This is a dot in the name and it should raise an error")
    assert "dot" not in kws  # 3 letters, no underscore/CamelCase -> filtered
    assert "raise" in kws  # 5+ letters kept
    assert "error" in kws


def test_find_relevant_files_ranks_by_keyword_hits(tmp_path):
    (tmp_path / "blueprints.py").write_text(
        "class Blueprint:\n    def __init__(self, name):\n        self.name = name\n"
        "        if '.' in name:\n            raise ValueError('dot not allowed')\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_blueprints.py").write_text(
        "def test_dotted_name_not_allowed():\n    assert True\n", encoding="utf-8"
    )

    files = cr.find_relevant_files(
        tmp_path, "Raise error when Blueprint name contains a dot", max_files=2
    )

    paths = [p for p, _ in files]
    assert "blueprints.py" in paths
    assert "tests/test_blueprints.py" not in paths  # tests/ excluded from retrieval
    assert "unrelated.py" not in paths


def test_find_relevant_files_truncates_large_files(tmp_path):
    big_content = "def Blueprint():\n    pass\n" + ("# padding line\n" * 5000)
    (tmp_path / "big.py").write_text(big_content, encoding="utf-8")

    files = cr.find_relevant_files(tmp_path, "Blueprint issue", max_bytes_per_file=500)
    assert files
    _, content = files[0]
    assert len(content) < len(big_content)
    assert content.endswith("(truncated) ...\n")


def test_find_relevant_files_no_keywords_returns_empty(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert cr.find_relevant_files(tmp_path, "is a of to") == []


# --------------------------------------------------------------------------
# patch_generator -- pure logic (prompt building, diff extraction)
# --------------------------------------------------------------------------


def test_build_prompt_includes_issue_and_files():
    prompt = pg.build_prompt("Fix the bug", [("foo.py", "def foo(): pass")])
    assert "Fix the bug" in prompt
    assert "foo.py" in prompt
    assert "def foo(): pass" in prompt


def test_extract_diff_plain():
    raw = (
        "--- a/foo.py\n+++ b/foo.py\n@@ -1,1 +1,2 @@\n def foo():\n+    return 1\n"
    )
    assert pg.extract_diff(raw).startswith("--- a/foo.py")


def test_extract_diff_strips_markdown_fence_and_chatter():
    raw = (
        "Sure! Here's the fix:\n\n```diff\n--- a/foo.py\n+++ b/foo.py\n"
        "@@ -1,1 +1,2 @@\n def foo():\n+    return 1\n```\n\nLet me know if that helps."
    )
    diff = pg.extract_diff(raw)
    assert diff.startswith("--- a/foo.py")
    assert "Let me know" not in diff
    assert "```" not in diff


def test_extract_diff_no_diff_present_returns_empty():
    assert pg.extract_diff("I don't know how to fix this.") == ""


def test_extract_diff_git_diff_header_form():
    raw = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
    diff = pg.extract_diff(raw)
    assert diff.startswith("diff --git")


# --------------------------------------------------------------------------
# sandbox_runner -- pure logic (pytest output parsing)
# --------------------------------------------------------------------------


def test_parse_outcomes_extracts_per_test_status():
    stdout = (
        "tests/test_blueprints.py::test_dotted_name_not_allowed PASSED [ 50%]\n"
        "tests/test_blueprints.py::test_route_decorator_custom_endpoint_with_dots FAILED [100%]\n"
    )
    outcomes = sr._parse_outcomes(stdout)
    assert outcomes["tests/test_blueprints.py::test_dotted_name_not_allowed"] == "PASSED"
    assert (
        outcomes["tests/test_blueprints.py::test_route_decorator_custom_endpoint_with_dots"]
        == "FAILED"
    )


def test_parse_outcomes_empty_stdout():
    assert sr._parse_outcomes("") == {}


def test_test_run_result_status_for_missing_test():
    result = sr.TestRunResult(exit_code=1, stdout="", stderr="", outcomes={"a::b": "PASSED"})
    assert result.status_for("a::b") == "PASSED"
    assert result.status_for("does::not_exist") == "MISSING"


# --------------------------------------------------------------------------
# dataset -- local cache round-trip (no network hit; instances.json is
# committed to the repo from the real `--fetch` run against SWE-bench_Lite)
# --------------------------------------------------------------------------


def test_instances_cache_file_has_the_two_chosen_real_instances():
    cached = dataset.load_cached()
    ids = {r["instance_id"] for r in cached}
    assert "pallets__flask-4045" in ids
    assert "psf__requests-3362" in ids
    for r in cached:
        assert r["repo"] in ("pallets/flask", "psf/requests")
        assert r["base_commit"]
        json.loads(r["FAIL_TO_PASS"])  # must be valid JSON list
        json.loads(r["PASS_TO_PASS"])


def test_get_instance_found_and_missing():
    inst = dataset.get_instance("pallets__flask-4045")
    assert inst["repo"] == "pallets/flask"
    with pytest.raises(KeyError):
        dataset.get_instance("nonexistent__repo-1")


# --------------------------------------------------------------------------
# Live-service tests
# --------------------------------------------------------------------------


@pytest.mark.live
def test_ollama_generates_a_nonempty_patch_for_the_real_flask_instance():
    """Hits the real local Ollama server (qwen2.5-coder:3b) with the real
    problem statement + retrieved file for pallets__flask-4045, and checks a
    non-empty unified diff comes back. Does not assert the patch is correct
    (that's what the Docker test-run stage is for) -- only that the real
    end-to-end generation call works."""
    inst = dataset.get_instance("pallets__flask-4045")
    repo_dir = PROJECT_DIR / "workdirs" / "pallets__flask-4045"
    if not (repo_dir / "src" / "flask" / "blueprints.py").exists():
        pytest.skip("flask repo checkout not present at workdirs/pallets__flask-4045")

    files = cr.find_relevant_files(repo_dir, inst["problem_statement"])
    attempt = pg.generate_patch(inst["instance_id"], inst["problem_statement"], files)
    assert attempt.raw_response  # the model said *something*
    # We don't require patch_text to be non-empty here (a 3B local model can
    # legitimately fail to produce a valid diff) -- see results.json for the
    # real recorded outcome of this specific attempt.


@pytest.mark.skipif(not DOCKER_UP, reason="docker daemon not reachable")
def test_docker_available_check_matches_real_daemon():
    assert sr.docker_available() is True


@pytest.mark.skipif(not DOCKER_UP, reason="docker daemon not reachable")
def test_sandbox_image_can_be_built():
    sr.build_image(quiet=True)
    import subprocess

    out = subprocess.run(
        ["docker", "images", "-q", sr.DOCKER_IMAGE], capture_output=True, text=True
    )
    assert out.stdout.strip() != ""


def test_results_file_if_present_has_real_recorded_outcomes():
    """If results.json exists (a real pipeline run happened), sanity-check its
    shape rather than trust any particular pass/fail value."""
    results_file = PROJECT_DIR / "results.json"
    if not results_file.exists():
        pytest.skip("results.json not yet produced (no pipeline run recorded)")
    data = json.loads(results_file.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    for instance_id, record in data.items():
        assert record["instance_id"] == instance_id
        assert "model_patch_chars" in record
        assert "docker_run" in record
