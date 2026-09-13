"""Thin synchronous HTTP client for the local Ollama server (no API key, no network calls)."""

from __future__ import annotations

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 120.0


def list_models() -> list[dict]:
    """Raw `/api/tags` response — the actually-pulled local models, verified at run time."""
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
    r.raise_for_status()
    return r.json().get("models", [])


def chat_capable_models() -> list[str]:
    """Names of locally pulled models that can do text completion.

    Excludes embedding-only models (e.g. nomic-embed-text, which reports only the
    'embedding' capability and cannot answer a chat prompt).
    """
    models = list_models()
    return sorted(m["name"] for m in models if "completion" in m.get("capabilities", []))


def generate(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Call Ollama's /api/generate (non-streaming) and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")
