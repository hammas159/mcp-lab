"""Module 3 — path traversal.

Pure tool-level test, no LLM calls needed: read_file's sandbox is supposed to
confine every request to data/sandbox/. We call it directly with a handful of
traversal payloads against both configurations and check whether the file
seeded at data/outside_sandbox_secret.txt (one level above the sandbox root)
is reachable.

Run: uv run python -m redteam.m3_path_traversal
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.tools import ToolSet  # noqa: E402

SECRET_MARKER = "path-traversal hole"

PAYLOADS = [
    "../outside_sandbox_secret.txt",
    "..\\outside_sandbox_secret.txt",
    "subdir/../../outside_sandbox_secret.txt",
    "./../outside_sandbox_secret.txt",
    "welcome.txt",  # control case: must still work in BOTH modes
]


def main() -> None:
    results = {}
    for hardened in (False, True):
        tools = ToolSet(hardened=hardened)
        key = "hardened" if hardened else "vulnerable"
        attempts = []
        for payload in PAYLOADS:
            try:
                content = tools.read_file(payload)
                escaped = SECRET_MARKER in content
                attempts.append(
                    {"payload": payload, "blocked": False, "escaped_sandbox": escaped}
                )
            except (PermissionError, FileNotFoundError, OSError) as exc:
                attempts.append(
                    {"payload": payload, "blocked": True, "error": str(exc)}
                )
        escapes = sum(a.get("escaped_sandbox", False) for a in attempts)
        results[key] = {
            "n_payloads": len(PAYLOADS),
            "escapes": escapes,
            "attempts": attempts,
        }

    print(json.dumps({k: {"escapes": v["escapes"]} for k, v in results.items()}, indent=2))

    from redteam.common import save_results

    save_results("m3_path_traversal", results)


if __name__ == "__main__":
    main()
