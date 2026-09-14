# Findings

[<- back to README](../README.md)

Every number here was produced by running the code in this repository. Each project's own
README has the full method and the complete result set.

---

## 01 - MCP Red-Team Platform

A real local assistant - FastAPI chat UI, LangChain tool-calling agent, four tools served
over **actual MCP** rather than direct Python calls - audited by five red-team modules.

**The finding:** the same prompt-injection payload behaves completely differently depending
on **where** it is delivered. A payload scoring **0%** in one vector reached **100%** in
another. The attack did not change; the delivery vector did.

Each module runs against the tools twice - `hardened=False` and `hardened=True` - so the
difference is the mitigation's measured effect rather than a claim about it.

---

## 02 - Multi-Hop RAG Audit (HotpotQA)

| Metric | Value |
|---|---:|
| Answer exact-match | 0.20 |
| Answer F1 | 0.436 |
| Retrieval recall of gold facts | **0.836** |
| Claimed vs **gold** - precision / recall / F1 | 0.81 / 0.406 / 0.523 |
| Claimed vs **fed** - precision (faithfulness) | **0.95** |
| Claimed vs **fed** - recall | 0.135 |
| Examples with at least one fabricated citation | **6%** |

**The finding:** retrieval is **not** the bottleneck. The gold facts were retrieved 83.6% of
the time while answer exact-match sat at 0.20 - the failure is downstream of retrieval.

Citations are scored **twice**, against the gold facts and against what was actually fed to
the model. That separates "cited the wrong thing" from "cited something it was never shown",
which one number cannot do. Precision against fed context is 0.95, so it rarely invents -
but recall is 0.135, so it rarely admits what it actually used.

---

## 03 - Tool-Calling Accuracy (BFCL v4)

5 models x 140 cases = **700 real calls**, 1,544 s wall time, **zero request errors**.

| Model | simple | multiple | parallel | parallel_multiple | **Overall** |
|---|---:|---:|---:|---:|---:|
| qwen2.5:7b-instruct | 95.0% | 92.5% | 90.0% | 70.0% | **87.9%** |
| qwen2.5:3b-instruct | 92.5% | 87.5% | 83.3% | 73.3% | **85.0%** |
| llama3.2:3b | 77.5% | 75.0% | 73.3% | 60.0% | **72.1%** |
| granite3.3:2b | 70.0% | 65.0% | 63.3% | 56.7% | **64.3%** |

**The finding:** the spread is **87.9% to 64.3%**, and every model's worst category is
`parallel_multiple`. Parallel multi-tool calls are where small models collapse - the 7B
drops 25 points from its own `simple` score.

Note the 3B-instruct beats the 7B on `parallel_multiple` (73.3% vs 70.0%), so the ordering
is not uniform across categories.

---

## 04 - SWE-bench Coding Agent

| Instance | Model patch | `git apply` result |
|---|---|---|
| `pallets__flask-4045` | 1,899 chars, real diff syntax, right file | `patch fragment without header at line 15` |
| `psf__requests-3362` | 488 chars, real diff syntax | `corrupt patch at line 11` |

**The finding:** both runs went the whole way - real checkout at the real `base_commit`,
real `test_patch` applied cleanly, real `pip install -e .`, real `git apply` inside the
container. **`git apply` rejected the model's own diffs.**

The failure is **diff formatting, not reasoning about code**. The model retrieved the right
files and produced plausible patches targeting them; it could not emit a valid unified diff.
That distinction is invisible in a pass/fail benchmark score.

---

## 05 - Hallucination Rate (TruthfulQA)

### MC1 accuracy, n=100 per cell

| Model | zero-shot | chain-of-thought | &Delta; |
|---|---:|---:|---:|
| qwen2.5:7b-instruct | **60%** | 25% | **-35 pts** |
| qwen2.5:3b-instruct | 45% | 23% | -22 pts |
| granite3.3:2b | 37% | 24% | -13 pts |
| llama3.2:3b | 36% | 22% | -14 pts |
| qwen2.5-coder:3b | 36% | 21% | -15 pts |

**The finding:** **not one model improved with chain-of-thought.** The largest model lost
the most - 35 points.

Both strategies run on identical questions with identical parsing, so the gap is
attributable to the prompt and nothing else. MC2 shows the same direction (7B: 0.411 &rarr;
0.197).

CoT is near-universally recommended. On a benchmark built from plausible misconceptions it
is actively harmful, which is worth knowing before applying it by default.

---

## 06 - Fact-Verification Agent (FEVER)

60 claims, label-stratified (20 SUPPORTS / 20 REFUTES / 20 NOT ENOUGH INFO), seed 42.
Evidence retrieved from **live Wikipedia** via the MediaWiki API and embedding similarity.

**Result: 38.3% accuracy** on three-way verification, against a 33.3% random baseline.

**The finding:** barely above chance. The pipeline is real end to end - real FEVER claims,
real live retrieval, real gold labels - and that is what it scores. Using live Wikipedia
rather than a frozen dump makes retrieval quality a real variable, which the project's
README discusses as a reproducibility cost.
