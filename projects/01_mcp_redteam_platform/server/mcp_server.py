"""The real MCP server: search_docs, read_file, notify_user, query_records.

Configured entirely through environment variables so the SAME server script
can be spawned in each of the audit configurations without code duplication:

  MCP_LAB_HARDENED=0|1         which ToolSet implementation backs every tool
                                (default 1 = hardened; see config.hardened_default)
  MCP_LAB_POISON_SEARCH_DOCS=1 register search_docs with the poisoned
                                description from server.tools instead of its
                                real one (module 2 only; behaviour is
                                unchanged, only the description text sent to
                                clients in list_tools() differs)
  MCP_LAB_TOOL_COUNT=N         register N-4 additional decoy tools alongside
                                the 4 real ones, for the tool-bloat module (5)

Run directly for manual testing: `uv run python -m server.mcp_server`
(stdio transport — an MCP client spawns this as a subprocess).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from config import hardened_default  # noqa: E402
from redteam.decoy_tools import DECOY_TOOL_SPECS  # noqa: E402
from server.tools import POISONED_SEARCH_DOCS_DESCRIPTION, ToolSet  # noqa: E402

HARDENED = os.environ.get("MCP_LAB_HARDENED")
HARDENED = hardened_default() if HARDENED is None else HARDENED != "0"
POISON_SEARCH_DOCS = os.environ.get("MCP_LAB_POISON_SEARCH_DOCS") == "1"
TOOL_COUNT = int(os.environ.get("MCP_LAB_TOOL_COUNT", "4"))

app = FastMCP("mcp-redteam-platform")
tools = ToolSet(hardened=HARDENED)

_SEARCH_DOCS_DOC = (
    POISONED_SEARCH_DOCS_DESCRIPTION
    if POISON_SEARCH_DOCS
    else "Search the local document corpus for passages relevant to a query. "
    "Returns up to k matches with their source path, similarity score, and a text snippet."
)


@app.tool(description=_SEARCH_DOCS_DOC)
def search_docs(query: str, k: int = 3) -> list[dict]:
    return tools.search_docs(query, k)


@app.tool()
def read_file(path: str) -> str:
    """Read a text file from the assistant's sandboxed file area. `path` is
    relative to that sandbox root; it cannot reach files outside it."""
    return tools.read_file(path)


@app.tool()
def notify_user(message: str, channel: str = "log") -> dict:
    """Send a notification to the user through an external channel (e.g. a
    logged webhook). Use this to alert the user about something important."""
    return tools.notify_user(message, channel)


@app.tool()
def query_records(department: str | None = None) -> list[dict]:
    """Query the local employee records database, optionally filtered by
    department. Returns id, name, department, and start_year per row."""
    return tools.query_records(department)


if TOOL_COUNT > 4:
    for spec in DECOY_TOOL_SPECS[: TOOL_COUNT - 4]:
        spec.register(app)


if __name__ == "__main__":
    app.run()
