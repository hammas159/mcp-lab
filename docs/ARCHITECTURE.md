# Architecture

[<- back to README](../README.md)

## Six projects, six virtual environments

Each project has **its own venv**. That is deliberate, not disorganisation.

Project 01 pins `mcp<2`, because `mcp>=2` renamed `FastMCP` to `MCPServer`. Forcing one
environment across all six would mean pinning every project to the oldest constraint any of
them has - so project 06 would inherit project 01's MCP pin for no reason, and a future
project could not use a newer library without breaking an older one.

The cost is disk space and six `uv pip install` runs. The benefit is that each project's
`pyproject.toml` states what **that project** actually needs.

## What is deliberately not shared

**No shared utility package.** Each project reimplements the small amount of Ollama plumbing
it needs. A shared `utils/` would couple six independent experiments, and the duplication is
perhaps forty lines.

**No shared results format beyond `results.json`.** Each project writes what its own
experiment produced.

## Project 06 does not use LangChain

It calls Ollama's `/api/embeddings` and `/api/chat` directly over HTTP with `httpx`.

Going through a framework wrapper for **two endpoints** was more code, not less. Each
project's README states which libraries it actually imports, rather than listing a stack it
nominally belongs to.

## MCP is real in project 01

The four tools in project 01 are served over **actual Model Context Protocol** via stdio
transport, loaded with `langchain_mcp_adapters.tools.load_mcp_tools` - not called as Python
functions with MCP mentioned in the README.

That matters for the red-team result: the attack surface being measured is the real one.

## Local only

Every project runs against a local Ollama instance. No API keys exist anywhere in this
repository, and no external inference is called.

The fleet:

| Model | Size |
|---|---|
| `qwen2.5:7b-instruct` | 4.7 GB |
| `llama3.2:3b` | 2.0 GB |
| `qwen2.5-coder:3b` | 1.9 GB |
| `qwen2.5:3b-instruct` | 1.9 GB |
| `granite3.3:2b` | 1.5 GB |
| `nomic-embed-text` | 274 MB |

## Real data, everywhere

Every benchmark is the published one, loaded from Hugging Face or a live API: HotpotQA,
BFCL v4, SWE-bench Lite, TruthfulQA, FEVER. **No synthetic corpus stands in for a dataset
anywhere in this repository.**
