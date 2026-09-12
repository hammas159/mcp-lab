"""End-to-end driver: real SWE-bench-Lite instance -> local Ollama patch
attempt -> real Docker sandbox test run -> results.json.

Usage:
    uv run python projects/04_swebench_coding_agent/run_pipeline.py --instance pallets__flask-4045
    uv run python projects/04_swebench_coding_agent/run_pipeline.py --instance psf__requests-3362
    uv run python projects/04_swebench_coding_agent/run_pipeline.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import context_retrieval as cr  # noqa: E402
import dataset  # noqa: E402
import patch_generator as pg  # noqa: E402
import repo_checkout as rc  # noqa: E402
import sandbox_runner as sr  # noqa: E402

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_FILE = HERE / "results.json"


def load_all_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {}


def save_all_results(d: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def run_one(
    instance_id: str, *, use_docker: bool, install_extra: str = "", test_cwd: str = "."
) -> dict:
    inst = dataset.get_instance(instance_id)
    print(f"\n=== {instance_id} ({inst['repo']}) ===")

    print("[1/4] checking out repo at base_commit ...")
    repo_path = rc.checkout(instance_id, inst["repo"], inst["base_commit"])

    print("[2/4] retrieving relevant source file(s) ...")
    files = cr.find_relevant_files(repo_path, inst["problem_statement"])
    for path, content in files:
        print(f"       -> {path} ({len(content)} chars)")
    if not files:
        print("       -> no files matched; patch generation will likely fail")

    print("[3/4] generating patch with local Ollama model ...")
    attempt = pg.generate_patch(instance_id, inst["problem_statement"], files)
    pg.save_attempt(attempt, RESULTS_DIR)
    print(f"       -> extracted diff: {len(attempt.patch_text)} chars")

    record: dict = {
        "instance_id": instance_id,
        "repo": inst["repo"],
        "base_commit": inst["base_commit"],
        "model": attempt.model,
        "retrieved_files": [p for p, _ in files],
        "model_patch_chars": len(attempt.patch_text),
        "model_patch_empty": len(attempt.patch_text.strip()) == 0,
    }

    if not use_docker:
        record["docker_run"] = "skipped (docker unavailable or not requested)"
        return record

    print("[4/4] running real Docker sandbox test (test_patch + model patch, "
          "FAIL_TO_PASS/PASS_TO_PASS) ...")
    fail_to_pass = json.loads(inst["FAIL_TO_PASS"])
    pass_to_pass = json.loads(inst["PASS_TO_PASS"])
    sandbox_result = sr.run_instance(
        instance_id,
        test_patch=inst["test_patch"],
        candidate_patch=attempt.patch_text,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        install_extra=install_extra,
        test_cwd=test_cwd,
    )
    record["docker_run"] = sandbox_result
    print(f"       -> outcome: {sandbox_result.get('outcome')}")
    return record


INSTANCE_INSTALL_EXTRA = {
    # requests' own test deps aren't declared as extras; pull them in explicitly
    # (its tests/conftest.py wraps the `httpbin`/`httpbin_secure` fixtures that
    # this plugin provides).
    "psf__requests-3362": "pytest-httpbin",
}

# psf/requests' FAIL_TO_PASS/PASS_TO_PASS node-ids (e.g. "test_requests.py::...")
# are relative to the tests/ directory, not the repo root -- run pytest from there.
INSTANCE_TEST_CWD = {
    "psf__requests-3362": "tests",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", help="single instance_id to run")
    ap.add_argument("--all", action="store_true", help="run every cached instance")
    ap.add_argument("--no-docker", action="store_true", help="skip the docker test-run stage")
    args = ap.parse_args()

    if args.all:
        ids = [r["instance_id"] for r in dataset.load_cached()]
    elif args.instance:
        ids = [args.instance]
    else:
        ap.error("pass --instance <id> or --all")
        return

    use_docker = not args.no_docker
    if use_docker and not sr.docker_available():
        print("WARNING: docker not reachable; running patch-generation only, "
              "recording docker_run as unavailable.")
        use_docker = False
    elif use_docker:
        print("building/verifying docker sandbox image ...")
        sr.build_image(quiet=True)

    all_results = load_all_results()
    for instance_id in ids:
        extra = INSTANCE_INSTALL_EXTRA.get(instance_id, "")
        cwd = INSTANCE_TEST_CWD.get(instance_id, ".")
        rec = run_one(instance_id, use_docker=use_docker, install_extra=extra, test_cwd=cwd)
        all_results[instance_id] = rec
        save_all_results(all_results)

    print(f"\nresults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
