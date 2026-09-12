"""Discover which models are actually pulled in the local Ollama fleet."""

from __future__ import annotations

import httpx

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def list_local_models(base_url: str = OLLAMA_BASE_URL) -> list[dict]:
    resp = httpx.get(f"{base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def get_chat_capable_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Return names of models that are chat/completion-capable (excludes
    embedding-only models such as nomic-embed-text)."""
    models = list_local_models(base_url)
    names = []
    for m in models:
        caps = m.get("capabilities", [])
        if "completion" in caps:
            names.append(m["name"])
    return sorted(names)


def get_tool_capable_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Return names of models that advertise the 'tools' capability."""
    models = list_local_models(base_url)
    return sorted(m["name"] for m in models if "tools" in m.get("capabilities", []))
