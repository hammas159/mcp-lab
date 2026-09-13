"""Fast, no-LLM tests for Project 1's tool logic. Run with the project's own
isolated venv (it has `mcp`, `langchain-mcp-adapters`, etc. that the shared
repo-root venv does not):

    projects/01_mcp_redteam_platform/.venv/Scripts/python.exe -m pytest tests/test_01_mcp_redteam.py -v

Live-agent tests (module 1/2/4/5's actual measurements) are run directly via
each redteam/m*.py script, not through pytest — they're full audit runs, not
unit tests, and their output IS the deliverable (redteam/results/*.json).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "projects" / "01_mcp_redteam_platform"
sys.path.insert(0, str(PROJECT_ROOT))

from server.tools import ToolSet  # noqa: E402


class TestPathTraversal:
    def test_hardened_blocks_parent_traversal(self):
        tools = ToolSet(hardened=True)
        try:
            tools.read_file("../outside_sandbox_secret.txt")
            assert False, "expected PermissionError"
        except PermissionError:
            pass

    def test_hardened_still_serves_legitimate_files(self):
        tools = ToolSet(hardened=True)
        content = tools.read_file("welcome.txt")
        assert "sandboxed file area" in content

    def test_vulnerable_escapes_sandbox(self):
        tools = ToolSet(hardened=False)
        content = tools.read_file("../outside_sandbox_secret.txt")
        assert "path-traversal hole" in content


class TestNotifyUserProvenance:
    def test_hardened_blocks_verbatim_readback(self):
        tools = ToolSet(hardened=True)
        secret = tools.read_file("internal_memo.txt")
        result = tools.notify_user(secret, channel="webhook")
        assert result["status"] == "blocked"
        assert len(tools.sent_notifications) == 0

    def test_vulnerable_sends_verbatim_readback(self):
        tools = ToolSet(hardened=False)
        secret = tools.read_file("internal_memo.txt")
        result = tools.notify_user(secret, channel="webhook")
        assert result["status"] == "sent"
        assert len(tools.sent_notifications) == 1

    def test_hardened_allows_unrelated_notifications(self):
        tools = ToolSet(hardened=True)
        result = tools.notify_user("status check ok", channel="log")
        assert result["status"] == "sent"


class TestQueryRecords:
    def test_returns_all_rows_without_filter(self):
        tools = ToolSet(hardened=True)
        rows = tools.query_records()
        assert len(rows) == 5

    def test_filters_by_department(self):
        tools = ToolSet(hardened=True)
        rows = tools.query_records(department="Engineering")
        assert len(rows) == 3
        assert all(r["department"] == "Engineering" for r in rows)
