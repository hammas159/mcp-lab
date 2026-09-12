"""Call a local Ollama model on one BFCL test case using native tool-calling.

Uses ``langchain_ollama.ChatOllama.bind_tools`` (per the project brief), which
sends the OpenAI-style ``tools`` list straight through to Ollama's ``/api/chat``
``tools`` parameter and parses any ``tool_calls`` Ollama returns back into
``AIMessage.tool_calls``.

Some local models (see README "Problems hit while building this") don't reliably
emit the native ``tool_calls`` field and instead print a JSON-looking function
call as plain text content. We add a best-effort fallback parser for that case
so those models aren't scored as "produced literally nothing" when they clearly
attempted the task -- but we record whether the fallback was used, so the
results are honest about which models used real native tool-calling.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from schema_convert import build_tools

_ROLE_MAP = {
    "user": HumanMessage,
    "system": SystemMessage,
    "assistant": AIMessage,
}


def to_messages(turns: list[dict]) -> list:
    messages = []
    for turn in turns:
        cls = _ROLE_MAP.get(turn.get("role", "user"), HumanMessage)
        messages.append(cls(content=turn.get("content", "")))
    return messages


def _try_parse_content_as_calls(content: str) -> list[tuple[str, dict]] | None:
    """Best-effort extraction of function-call-shaped JSON from free-text content."""
    text = content.strip()
    if not text:
        return None
    # Strip common code-fence wrapping.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        # Fall back to grabbing the outermost {...} or [...] span.
        start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not start_candidates:
            return None
        start = min(start_candidates)
        end_candidates = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
        if not end_candidates:
            return None
        end = max(end_candidates) + 1
        try:
            candidates.append(json.loads(text[start:end]))
        except json.JSONDecodeError:
            return None

    parsed = candidates[0]
    items = parsed if isinstance(parsed, list) else [parsed]
    calls: list[tuple[str, dict]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or item.get("function")
        args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
        if not name or not isinstance(args, dict):
            return None
        calls.append((name, args))
    return calls or None


def call_model_on_case(
    chat: ChatOllama, case: dict, timeout_s: float = 90.0
) -> dict[str, Any]:
    """Invoke one model on one BFCL case and return raw + normalized results."""
    tools = build_tools(case)
    bound = chat.bind_tools(tools)
    messages = to_messages(case["question"])

    t0 = time.time()
    try:
        resp = bound.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - want to record any failure, not crash the run
        return {
            "predicted_calls": [],
            "latency_s": time.time() - t0,
            "raw_content": None,
            "used_fallback_parse": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    latency = time.time() - t0

    calls: list[tuple[str, dict]] = []
    for tc in resp.tool_calls or []:
        calls.append((tc.get("name"), tc.get("args") or {}))

    raw_content = resp.content if isinstance(resp.content, str) else str(resp.content)
    used_fallback = False
    if not calls and raw_content:
        fallback = _try_parse_content_as_calls(raw_content)
        if fallback:
            calls = fallback
            used_fallback = True

    return {
        "predicted_calls": calls,
        "latency_s": latency,
        "raw_content": raw_content,
        "used_fallback_parse": used_fallback,
        "error": None,
    }
