# Future work

[<- back to README](../README.md)

## 1. A larger-model comparison across all six

Every result tops out at 7B. Each project is written so a larger model adds a **comparison
row** rather than requiring a rewrite.

**Project 04 first.** Its patches failed on diff *formatting*, not reasoning - and whether
that is a capability problem or a prompting problem is a real open question with a cheap
answer.

**Project 05 second**, because the chain-of-thought result is the one most likely to reverse
with scale, and a reversal would be as interesting as the original finding.

## 2. More instances for project 04

Two instances is enough to show the pipeline works and that these two patches were
malformed. It is not enough to state a rate. Twenty would be.

## 3. Repeats and variance

Every number is a single run at a fixed seed. Three to five repeats per cell would turn
point estimates into intervals, and would show whether the smaller gaps in project 03 are
real.

## 4. Prompt sensitivity as a measured variable

Project 05 shows a 35-point swing from one prompting change. That means every other
project's numbers are prompt-dependent in ways currently unquantified - and the same
generate-once/score-many design used elsewhere would measure it.

## 5. Fix the diff formatting in project 04

The model targeted the right files and produced plausible patches. Constrained decoding, a
diff-validating retry loop, or asking for whole files instead of diffs would each test a
different hypothesis about *why* the formatting failed.

## 6. Push the red-team modules further

Project 01 found that delivery vector matters more than payload phrasing. That suggests
enumerating vectors systematically - tool descriptions, tool results, document content,
filenames, error messages - rather than enumerating payloads.

## 7. Official scoring for project 05's MC2

Currently a heuristic. Using TruthfulQA's published scoring would make the MC2 numbers
comparable to the literature, rather than only internally consistent.
