"""Paths and the single hardened/vulnerable toggle used by the red-team audit.

Every module in redteam/ measures the SAME tool implementations twice: once with
HARDENED=False (the naive first-pass implementation) and once with HARDENED=True
(the fixed implementation). This is what lets one codebase produce an honest
before/after measurement without shipping two divergent copies of the tools.
The webapp and any real usage always run with HARDENED=True.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_DIR = ROOT / "data" / "eval"
SANDBOX_DIR = ROOT / "data" / "sandbox"
RECORDS_DB = ROOT / "data" / "records.db"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("MCP_LAB_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("MCP_LAB_EMBED_MODEL", "nomic-embed-text")


def hardened_default() -> bool:
    return os.environ.get("MCP_LAB_HARDENED", "1") != "0"
