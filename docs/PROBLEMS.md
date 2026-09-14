# Problems hit while building this

[<- back to README](../README.md)

Each project's own README has its full list. These are the ones that cost the most time or
changed how something was built.

## MCP 1.x and 2.x are not the same library

`mcp<2` exposes `mcp.server.fastmcp.FastMCP`. `mcp>=2` renamed it to `MCPServer`. Code
written against one fails on the other with an import error that does not explain itself.

**Fix:** project 01 pins `mcp<2` in its own venv. This is the main reason the projects do
not share an environment - see [ARCHITECTURE.md](ARCHITECTURE.md).

## A template signature change crashed every page load

`starlette` 1.6.0 changed `Jinja2Templates.TemplateResponse` to take the request **first**.
The old argument order raises no deprecation warning - it silently mis-binds and then
crashes inside Jinja2's cache, far from the actual mistake.

**Fix:** request-first everywhere. The symptom and the cause were several frames apart,
which is what made it expensive.

## Roughly 10,000 redundant embedding calls

Project 01 re-embedded the same corpus on every run of every red-team module. Five modules,
repeated runs, same documents each time.

**Fix:** an on-disk cache keyed by `(model, sha256(text))`. **116 s cold, 0.47 s warm.**

The lesson was not the cache - it was that nobody noticed until a run was timed.

## An attack that scored 0% everywhere

A prompt-injection payload wrapped in a `<system>` XML tag scored zero against every module.
The easy conclusion was "the assistant is robust".

**What was done instead:** investigated why. Plain imperative text -
`IMPORTANT INSTRUCTIONS FOR AI ASSISTANT:` - **worked** in the document-content vector where
the XML tag did not.

The finding is that **delivery vector matters more than payload phrasing**, and it only
exists because a zero was treated as a question rather than an answer.

## The FEVER loading script no longer runs

`datasets` removed script support in v4+, and `trust_remote_code` is rejected outright on
`datasets` 5.0.1. The dataset's own `fever.py` cannot execute.

**Fix:** load from the auto-generated `refs/convert/parquet` branch instead, and verify the
split shape (78,947 flattened rows regrouping to claims) matches the official `labelled_dev`
split described in the FEVER paper before using it.

## Both SWE-bench patches failed to apply

Not a bug in the pipeline - the pipeline worked. `git apply` genuinely rejected the model's
diffs with real stderr.

**What was done:** reported it as the result rather than retrying until something passed.
The artifacts (`results/*.model_patch.diff`, `*.raw_response.txt`) are committed so the
claim is checkable.
