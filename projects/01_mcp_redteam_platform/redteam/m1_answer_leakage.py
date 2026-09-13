"""Module 1 — answer leakage.

Part A (retrieval-only, no LLM calls): for every eval question, call
search_docs directly through ToolSet in both configurations and check whether
any of the top-k hits comes from data/eval/ (the answer-labelled fixtures) —
a real search_docs caller. This isolates the actual bug (a misconfigured
retrieval root) from any variance an LLM would add.

Part B (end-to-end, real Ollama calls, smaller sample): run the full MCP
agent on a subset of questions in both configurations and score its answers
against the gold answer key, to show what the leak actually buys an attacker
— an inflated apparent accuracy number, not just a retrieval-log curiosity.

Run: uv run python -m redteam.m1_answer_leakage
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.client import MCPAgentSession  # noqa: E402
from config import EVAL_DIR  # noqa: E402
from redteam.common import save_results  # noqa: E402
from server.tools import ToolSet  # noqa: E402

END_TO_END_SAMPLE = 15


def _is_leak(hit_path: str) -> bool:
    return "eval" in Path(hit_path).parts


def part_a_retrieval_leak() -> dict:
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    out = {}
    for hardened in (False, True):
        tools = ToolSet(hardened=hardened)
        leaked = 0
        for q in questions:
            hits = tools.search_docs(q["question"], k=3)
            if any(_is_leak(h["path"]) for h in hits):
                leaked += 1
        key = "hardened" if hardened else "vulnerable"
        out[key] = {
            "n_questions": len(questions),
            "leaked_count": leaked,
            "leak_rate": round(leaked / len(questions), 4),
        }
    return out


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _scores_correct(answer_text: str, gold: str) -> bool:
    return _normalize(gold) in _normalize(answer_text)


async def _run_end_to_end(hardened: bool, sample: list[dict]) -> dict:
    correct = 0
    per_question = []
    async with MCPAgentSession(hardened=hardened) as session:
        for item in sample:
            result = await session.run_turn(
                f"Answer this question using search_docs to find supporting context: "
                f"{item['question']}"
            )
            is_correct = _scores_correct(result["answer"], item["answer"])
            correct += is_correct
            per_question.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold": item["answer"],
                    "agent_answer": result["answer"],
                    "correct": is_correct,
                    "n_tool_calls": len(result["trace"]),
                }
            )
    return {
        "n": len(sample),
        "accuracy": round(correct / len(sample), 4),
        "per_question": per_question,
    }


async def part_b_end_to_end() -> dict:
    qa_dir = EVAL_DIR / "qa"
    all_items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(qa_dir.glob("*.json"))]
    sample = all_items[:END_TO_END_SAMPLE]
    out = {}
    for hardened in (False, True):
        key = "hardened" if hardened else "vulnerable"
        out[key] = await _run_end_to_end(hardened, sample)
    return out


def main() -> None:
    print("Part A: retrieval-only leak rate (no LLM calls)...")
    part_a = part_a_retrieval_leak()
    print(json.dumps(part_a, indent=2))

    print(f"\nPart B: end-to-end agent accuracy on {END_TO_END_SAMPLE} questions...")
    part_b = asyncio.run(part_b_end_to_end())
    print(
        json.dumps(
            {k: {"n": v["n"], "accuracy": v["accuracy"]} for k, v in part_b.items()}, indent=2
        )
    )

    save_results("m1_answer_leakage", {"part_a_retrieval_leak": part_a, "part_b_end_to_end": part_b})


if __name__ == "__main__":
    main()
