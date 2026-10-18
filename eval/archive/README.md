# Archived evaluation harnesses

Nothing in this directory is imported by the active `python -m eval` CLI. It is
kept as read-only history: old questions, pools, runs, judgments, reports, and
the scripts that produced them.

## v1

Human-judged static candidate pools (~80 ids per question) with a small human
gold set. Retired because a frozen pool can only credit chunks it already
contains: a query-enhancement change that retrieved different, correct chunks
was scored as a regression. Human grades are no longer used.

## v2

Deeper static pools (250 candidates per question) graded by an LLM. Retired
because the pool still capped recall (relevant chunks outside the pool scored
zero) and the unanchored judge drifted badly against the human gold set,
over-grading entity mentions.

## v3

Iteratively pooled, human-anchored LLM judging: human grades were frozen seed
truth, anchors and probe holdouts calibrated the judge, and `harvest.py`
extended pools from run files. Retired because it depended on human grades and
pool provenance, mixed judge metadata, and kept two contradictory grading
modes. The run-driven part of the idea survives in `eval/grade.py`.

## v4

A pre-built candidate pool graded ahead of time, with three different
`graded_by` values in the judgments (`dots-studio/dots-3-note-preview:free`,
`qwen-3.8-27b (neuralwatt, pre-freellm sweep)`, and
`@cf/qwen/qwen3.8-27b`). Retired because upfront pool grading reintroduced the
static-pool ceiling and the mixed judges made scores incomparable.

## Scripts and tests

`scripts/` contains the retired modules (`score.py`, `harvest.py`,
`llm_grade.py`, `build_seeds.py`, `pilot.py`, `agreement.py`, `annotate.py`,
`audit_bias.py`, `compare_versions.py`, `build_seeds_v4.py`). `tests/` contains
their old unit tests. They are kept for reference only; their imports point at
modules that now live under `archive/scripts/` and are not collected by
`pytest tests/eval`.

The active benchmark is described in `eval/README.md`.
