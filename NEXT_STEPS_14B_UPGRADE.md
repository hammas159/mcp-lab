# Next steps: 14B model upgrade — resume point for a new session

Written 2026-09-13. Read this file back to Claude Code at the start of a new
session if this conversation's context/memory is lost, to pick up exactly
where this one left off.

## Where things stand

All 6 agentic AI projects in this repo (`D:\github\mcp-lab`) are built,
tested, and committed **locally** (7 commits on `main`, working tree clean,
121 tests passing). **Nothing has been pushed to GitHub yet.** See
`projects/*/README.md` for each project's real, measured findings.

Every project so far ran on the small end of the local Ollama fleet — no
model above 7B parameters was used anywhere:

| Model | Params | Used in |
|---|---|---|
| `qwen2.5:7b-instruct` | 7.6B | 01, 02, 06 (chat/reasoning), 03 + 05 (fleet) |
| `llama3.2:3b` | 3.2B | 03, 05 (fleet) |
| `qwen2.5-coder:3b` | 3.1B | **04 (SWE-bench, sole model)**, 03 (fleet) |
| `qwen2.5:3b-instruct` | 3.1B | 03, 05 (fleet) |
| `granite3.3:2b` | 2.5B | 03, 05 (fleet) |
| `nomic-embed-text` | 137M | embeddings (01, 02, 06) |

## The decision in progress

The user's machine has a **16GB GPU**. Project 4 (SWE-bench) is the one
clear task failure among the 6: `qwen2.5-coder:3b`'s generated patches failed
real `git apply` against the actual repos (fabricated git blob hashes, a
malformed hunk header, and once the wrong file entirely due to a retrieval
miss) — a legitimate, honestly-reported negative finding, not a bug.

The user wants to install a **14B model** and **re-run all 6 projects** with
it, to see whether a bigger model changes the findings — especially whether
it fixes Project 4. **Which specific 14B model(s) to use was not yet decided
when this file was written** — ask the user directly:

- `qwen2.5-coder:14b` alone (~9GB, code-focused, simplest, one download) — the
  direct upgrade path for Project 4; usable everywhere else too but not as
  strong at general reasoning as an instruct model.
- `qwen2.5-coder:14b` **+** `qwen2.5:14b-instruct` (~18GB total) — coder for
  Project 4, general-instruct for 01/02/06's chat work and added as a new
  fleet member in 03/05's model-comparison studies.
- Something else the user names.

Both fit comfortably in 16GB VRAM at Q4 quantization. Pull with
`ollama pull <name>` (not on the Bash PATH here — Ollama is at
`AppData\Local\Programs\Ollama`; probe `http://127.0.0.1:11434/api/tags`
instead of running `ollama` directly, per the working pattern already used in
this session).

## What "re-run all 6" actually means per project

This is **not a quick swap** — each project needs a real, complete re-run
against the new model, with its README's numbers updated to the *new* real
results (never mix old-model and new-model numbers in one table; consider
keeping the old numbers as a labeled comparison rather than overwriting them
outright, since "7B vs 14B" is itself an interesting finding worth keeping).

- **01** (`projects/01_mcp_redteam_platform/`): change `CHAT_MODEL` in
  `config.py` (or the `MCP_LAB_CHAT_MODEL` env var), rerun
  `.venv/Scripts/python -m redteam.run_all` (its own isolated venv — see its
  README's "why this project has its own venv"). Re-embedding is cached
  (`server/embeddings.py`) so only the chat-model calls actually redo.
- **02** (`hotpotqa_multihop_rag/`): swap the model in `pipeline.py`/
  `run_experiment.py`, rerun. Real run takes a while (LLM calls per
  question); was time-boxed to 15/50 questions for the end-to-end parts last
  time — consider the same again with a 14B model, since it will be slower
  per call.
- **03** (`bfcl_tool_calling/`): this project's whole point is comparing the
  fleet — **add** the 14B model(s) as new entries rather than replacing
  anything, then rerun `run_eval.py`. Check whether the 14B model actually
  supports Ollama's native tool-calling `tools` param before assuming it does
  (qwen2.5-coder:3b did not, and needed a fallback parser — a 14B coder model
  should support it properly, but verify).
- **04** (`swebench_coding_agent/`): swap `qwen2.5-coder:3b` → the new coder
  model in `patch_generator.py`, rerun against the same 2 instances (flask,
  requests) for a clean before/after comparison, or pick fresh instances if
  those two feel exhausted. Docker must be running. Time-box the Docker
  build/run step as before (30-45 min genuine-effort budget) and report
  honestly if it still doesn't work — a 14B model is not guaranteed to fix
  this; if it still fails, that itself is worth reporting (see "what a bigger
  model actually needs" below).
- **05** (`truthfulqa_hallucination/`): fleet comparison, same as 03 — add
  the 14B model(s), don't replace. Real run was ~2,190s for 2,000 calls at
  the small-model sizes; a 14B model will be meaningfully slower per call,
  budget accordingly (this is a 100-question × N-model × 2-strategy × 2-task
  grid — consider reducing sample size if wall-clock becomes impractical
  rather than silently truncating without saying so).
- **06** (`fever_fact_verification/`): swap the verdict model
  (`qwen2.5:7b-instruct` → 14B) in `verdict.py`/`pipeline.py`, rerun. Current
  accuracy (38.3%) is barely above the ~33% random baseline for a 3-way
  task — worth explicitly checking whether a bigger model closes that gap or
  whether (as suspected) the real bottleneck is retrieval quality, which a
  bigger *verdict* model wouldn't fix at all.

## What a bigger model actually needs (told to the user already, worth
## repeating here so it isn't re-litigated)

Project 4's own README already diagnoses that model size is only *part* of
why the patch failed — the other cause was a **retrieval miss** (wrong file
shown to the model). A 14B model fed the wrong file will still fail. If the
14B rerun still fails, check whether it's the same failure mode
(malformed/fabricated diff) or a different one (wrong target) before
concluding "bigger model didn't help" — they're different problems with
different fixes (model capability vs. retrieval quality).

## Operational hazards hit repeatedly this build — avoid repeating them

1. **Never run a broad process-matching kill** (e.g.
   `taskkill //F //IM python.exe //FI "..."`) while other agents/processes
   may be running concurrently on this machine — it killed unrelated live
   evaluation runs more than once during this build. Kill by specific PID
   instead.
2. **`uv run --python <path-to-some-venvs-python.exe>` does NOT mean "use
   that venv."** uv still resolves the *project* by walking up to the
   nearest `pyproject.toml` and can rebuild/replace *that* project's own
   `.venv`, ignoring the intended target. To run something in an isolated
   venv (like Project 01's), invoke its `python.exe`/`pytest` binary
   directly — never `uv run --python <that venv's python>`.
3. **Never write a numeric result you didn't actually produce by running the
   code.** Every README in this repo's numbers came from a real executed
   run — keep it that way for the 14B re-runs too.
4. Standing rule from the user, still in force: **commit locally, show push
   commands, wait for explicit approval before pushing anything to GitHub.**

See also the Claude Code memory file `six-agentic-projects.md` (in
`C:\Users\dell\.claude\projects\d--github\memory\`) for the same context in
a more compact form, and `PROJECTS.md` / the `six-lab-build` memory for the
wider 30-project portfolio this sits alongside.
