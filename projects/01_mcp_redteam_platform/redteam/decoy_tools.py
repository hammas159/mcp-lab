"""Decoy tools for module 5 (tool bloat): plausible, similar-sounding names
clustered around each of the 4 real tools, so growing the tool count from 5 to
30 means growing the number of near-duplicate names the agent has to tell
apart, not just unrelated noise. Each decoy is wired to a stub that returns a
fixed "not the tool you want" marker rather than doing real work — the
measurement is tool SELECTION accuracy, not decoy execution correctness."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class DecoyToolSpec:
    name: str
    description: str
    cluster: str  # which real tool this decoy is meant to be confused with

    def _stub(self) -> Callable:
        def stub(**kwargs) -> dict:
            return {"decoy": True, "tool": self.name, "note": "this is a decoy tool"}

        stub.__name__ = self.name
        stub.__doc__ = self.description
        return stub

    def register(self, app) -> None:
        app.add_tool(self._stub(), name=self.name, description=self.description)


DECOY_TOOL_SPECS: list[DecoyToolSpec] = [
    # cluster: search_docs
    DecoyToolSpec("search_documents", "Search indexed documents for a keyword match.", "search_docs"),
    DecoyToolSpec("search_knowledge_base", "Query the internal knowledge base for an answer.", "search_docs"),
    DecoyToolSpec("lookup_docs", "Look up a document by topic in the docs store.", "search_docs"),
    DecoyToolSpec("find_passages", "Find text passages related to a topic.", "search_docs"),
    DecoyToolSpec("document_search", "Run a search over the document index.", "search_docs"),
    DecoyToolSpec("search_corpus", "Search the text corpus for relevant content.", "search_docs"),
    DecoyToolSpec("retrieve_docs", "Retrieve documents relevant to a query string.", "search_docs"),
    DecoyToolSpec("search_index", "Query the search index for matching entries.", "search_docs"),
    # cluster: read_file
    DecoyToolSpec("read_document", "Read the contents of a document by name.", "read_file"),
    DecoyToolSpec("get_file", "Get the raw contents of a file by path.", "read_file"),
    DecoyToolSpec("load_file", "Load a file's text contents into memory.", "read_file"),
    DecoyToolSpec("fetch_file", "Fetch a file's contents from storage.", "read_file"),
    DecoyToolSpec("open_file", "Open a file and return its text.", "read_file"),
    DecoyToolSpec("read_local_file", "Read a file from local disk by relative path.", "read_file"),
    DecoyToolSpec("view_file", "View the contents of a stored file.", "read_file"),
    # cluster: notify_user
    DecoyToolSpec("notify_admin", "Send a notification to the system administrator.", "notify_user"),
    DecoyToolSpec("send_notification", "Send a notification through a configured channel.", "notify_user"),
    DecoyToolSpec("alert_user", "Raise an alert visible to the user.", "notify_user"),
    DecoyToolSpec("notify_channel", "Post a notification to a named channel.", "notify_user"),
    DecoyToolSpec("push_notification", "Push a notification to the user's device.", "notify_user"),
    DecoyToolSpec("message_user", "Send a direct message to the user.", "notify_user"),
    DecoyToolSpec("send_alert", "Send an alert message about an event.", "notify_user"),
    # cluster: query_records
    DecoyToolSpec("query_database", "Run a query against the records database.", "query_records"),
    DecoyToolSpec("lookup_records", "Look up records matching a filter.", "query_records"),
    DecoyToolSpec("get_employees", "Get a list of employees, optionally filtered.", "query_records"),
    DecoyToolSpec("fetch_records", "Fetch records from the database by criteria.", "query_records"),
    DecoyToolSpec("search_records", "Search stored records for a match.", "query_records"),
    DecoyToolSpec("list_employees", "List employees in the records database.", "query_records"),
]
