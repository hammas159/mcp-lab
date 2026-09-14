# Limitations

[<- back to README](../README.md)

## Small samples throughout

| Project | n |
|---|---|
| 02 HotpotQA | 50 questions |
| 03 BFCL | 140 cases per model |
| 04 SWE-bench | **2 instances** |
| 05 TruthfulQA | 100 questions per cell |
| 06 FEVER | 60 claims |

Project 04's two instances are the weakest. Two failures establish that the pipeline runs
end to end and that *these* patches were malformed - **not** a rate at which the model
produces invalid diffs.

## Single seed, single run

Results come from one run at a fixed seed. No variance estimates, no confidence intervals.
A difference of a few points between two models in project 03 should not be read as
meaningful without repeats.

## The fleet tops out at 7B

Every result uses models of 7B parameters or fewer. Nothing here says how a larger model
behaves, and several findings might not survive scale - particularly project 05's
chain-of-thought result, which is exactly the kind of effect that can reverse with
capability.

## Project 06 uses live Wikipedia

Evidence is retrieved from the live MediaWiki API, not a frozen dump. That makes the
pipeline realistic and the result **not exactly reproducible** - Wikipedia changes. The
project's README discusses this trade explicitly.

## Project 05's MC2 is an approximation

MC2 is computed by a heuristic rather than TruthfulQA's official scoring. The direction of
the effect matches MC1, but the absolute MC2 values should not be compared against published
numbers.

## Prompt sensitivity is not measured

Each project uses one prompt formulation. Project 05 demonstrates how much prompting
matters - a 35-point swing from one change - which means every other project's numbers are
also prompt-dependent in ways not quantified here.

## Project 01's red-team modules are self-built

The five audit modules are written for this project, not drawn from a published red-team
suite. The before/after comparison is internally valid - same tools, same payloads, hardening
toggled - but the coverage is not benchmarked against anything.
