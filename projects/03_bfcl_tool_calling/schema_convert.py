"""Convert BFCL's function-schema dialect into standard JSON Schema / OpenAI-tool
format, which is what ``langchain_ollama.ChatOllama.bind_tools`` (and Ollama's
native ``tools`` chat parameter) expect.

BFCL parameter "type" values seen in the v4 non-executable categories:
``dict``, ``string``, ``integer``, ``float``, ``boolean``, ``array``, ``tuple``,
``any``. These map onto JSON Schema as: dict->object, float->number, tuple->array
(BFCL tuples are just ordered lists in JSON), any->no type constraint.
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP = {
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
    "string": "string",
    "dict": "object",
}


def _convert_node(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t = node.get("type")

    if t == "dict":
        out["type"] = "object"
        props = node.get("properties")
        if props:
            out["properties"] = {k: _convert_node(v) for k, v in props.items()}
        if "required" in node:
            out["required"] = node["required"]
    elif t in ("array", "tuple"):
        out["type"] = "array"
        if "items" in node:
            out["items"] = _convert_node(node["items"])
    elif t == "any":
        pass  # no constraint -- leave "type" out entirely
    elif t in _TYPE_MAP:
        out["type"] = _TYPE_MAP[t]
    # unknown type strings are passed through as-is rather than dropped
    elif t is not None:
        out["type"] = t

    if "description" in node:
        out["description"] = node["description"]
    if "enum" in node:
        out["enum"] = node["enum"]
    if "default" in node:
        out["default"] = node["default"]
    return out


def bfcl_function_to_openai_tool(fn: dict[str, Any]) -> dict[str, Any]:
    """Convert one BFCL ``function`` entry to an OpenAI-style tool dict."""
    params = _convert_node(fn.get("parameters", {"type": "dict", "properties": {}}))
    if "type" not in params:
        params["type"] = "object"
    if "properties" not in params:
        params["properties"] = {}
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": params,
        },
    }


def build_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the OpenAI-tool-format list for a BFCL test case's offered functions."""
    return [bfcl_function_to_openai_tool(fn) for fn in case["function"]]
