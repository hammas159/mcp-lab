"""Does the model's patch apply to the real repository?

    python apply_check.py

This answers the question the Docker sandbox answers *second*. The sandbox
first installs the project's dependencies, which needs the network; `git apply`
needs nothing but the checked-out repo, so this still works when the sandbox
cannot be built.

It separates the two failure modes the project cares about:
  * "corrupt patch" / "fragment without header" -> the diff is malformed
  * "patch does not apply"                      -> the diff parses, and its
                                                   line numbers are wrong
"""

from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
PATCH_DIRS = [HERE / "results_14b_patches", HERE / "results"]

MALFORMED = ("corrupt patch", "without header", "unrecognized input")


def check(instance_id: str, patch: pathlib.Path) -> dict:
    workdir = HERE / "workdirs" / instance_id
    if not workdir.exists():
        return {"instance_id": instance_id, "error": "no workdir checked out"}
    r = subprocess.run(
        ["git", "-C", str(workdir), "apply", "--check", str(patch)],
        capture_output=True,
        text=True,
    )
    err = r.stderr.strip()
    malformed = any(m in err for m in MALFORMED)
    return {
        "instance_id": instance_id,
        "patch_chars": len(patch.read_text(encoding="utf-8")),
        "applies": r.returncode == 0,
        "diff_is_malformed": malformed,
        "failure_mode": (
            "applies"
            if r.returncode == 0
            else "malformed_diff"
            if malformed
            else "wrong_line_numbers"
        ),
        "git_apply_stderr": err,
    }


def main() -> None:
    results = {}
    for d in PATCH_DIRS:
        if not d.exists():
            continue
        for patch in sorted(d.glob("*.model_patch.diff")):
            instance_id = patch.name.replace(".model_patch.diff", "")
            if instance_id in results:
                continue
            results[instance_id] = check(instance_id, patch)

    out = HERE / "results_apply_check.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{'instance':28} {'applies':>8}  failure mode")
    print("-" * 72)
    for instance_id, r in results.items():
        if "error" in r:
            print(f"{instance_id:28} {'-':>8}  {r['error']}")
            continue
        print(f"{instance_id:28} {str(r['applies']):>8}  {r['failure_mode']}")
        if r["git_apply_stderr"]:
            print(f"{'':28}           {r['git_apply_stderr'].splitlines()[0][:70]}")
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
