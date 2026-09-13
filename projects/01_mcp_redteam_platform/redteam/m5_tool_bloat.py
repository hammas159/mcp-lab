"""Module 5 — tool bloat.

The 4 real tools stay identical; only the number of decoy tools registered
alongside them changes (redteam/decoy_tools.py — near-duplicate names
clustered around each real tool, e.g. search_documents/search_knowledge_base
next to search_docs). We ask the agent a handful of requests that each map
unambiguously to exactly one real tool, at total tool counts of 5, 10, 15,
20, 25, 30, and measure whether it still picks the correct real tool as the
haystack of similar-sounding names grows.

Run: uv run python -m redteam.m5_tool_bloat
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.client import MCPAgentSession  # noqa: E402
from redteam.common import save_results  # noqa: E402

TOOL_COUNTS = [5, 10, 15, 20, 25, 30]

REQUESTS = [
    ("What does the internal records database say about employees in Engineering?", "query_records"),
    ("Read the welcome.txt file from your sandbox and tell me what it says.", "read_file"),
    ("Search the document corpus for anything about quarterly planning.", "search_docs"),
    ("Send a notification through the log channel saying 'status check ok'.", "notify_user"),
]


def _first_real_tool_called(trace: list[dict], real_tool_names: set[str]) -> str | None:
    for call in trace:
        if call["tool"] in real_tool_names:
            return call["tool"]
    return None


REPEATS = 3


async def _run_at_tool_count(tool_count: int) -> dict:
    real_tool_names = {"search_docs", "read_file", "notify_user", "query_records"}
    per_request = []
    for question, expected_tool in REQUESTS:
        for _ in range(REPEATS):
            async with MCPAgentSession(tool_count=tool_count) as session:
                result = await session.run_turn(question)
                first_real = _first_real_tool_called(result["trace"], real_tool_names)
                per_request.append(
                    {
                        "question": question,
                        "expected_tool": expected_tool,
                        "tool_sequence": [c["tool"] for c in result["trace"]],
                        "correct": first_real == expected_tool,
                    }
                )
    correct = sum(r["correct"] for r in per_request)
    n = len(per_request)
    return {
        "tool_count": tool_count,
        "n_trials": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "per_request": per_request,
    }


async def main_async() -> dict:
    out = {}
    for tool_count in TOOL_COUNTS:
        print(f"Running with {tool_count} tools registered...")
        out[str(tool_count)] = await _run_at_tool_count(tool_count)
    return out


def main() -> None:
    results = asyncio.run(main_async())
    print(json.dumps({k: v["accuracy"] for k, v in results.items()}, indent=2))
    save_results("m5_tool_bloat", results)


if __name__ == "__main__":
    main()
