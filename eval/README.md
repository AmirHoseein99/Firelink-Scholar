# Evaluation

This directory contains the IR test collection used to measure the retriever
(v1, v2, and the current v3), plus the tooling to score runs and grow the
judgment set.

## The incident that started this

A **query-enhancement technique** was tried on the retriever — and the
metrics went **down**. Inspection showed the drop was not real: the technique
retrieved *different* (often correct) chunks that the frozen v1 evaluation
pools had never judged, and unjudged ids score 0 by the "unjudged =
non-relevant" assumption. The evaluation set was punishing retrieval novelty
instead of measuring quality. That false regression is what prompted
rebuilding the eval set from scratch.

## Two measured failure points in the static-pool approach

The retriever was originally evaluated against static pools of candidates per
question (v1: ~80 human-judged chunks per question; v2: 250-candidate pools
graded by an LLM):

**1. The static-pool ceiling ("unjudged = non-relevant").** A frozen pool can
only credit chunks it already contains. The query-enhancement incident above
is this failure in its mildest form (v1 pools were shallow, ~80 ids). v2
deepened the pools to 250 — and still: 15 of the 17 human-relevant chunks for
q039 sat *outside* the pool (the cap crowded out entity matches). Any
retriever change that surfaced *different* — even *better* — chunks got them
scored as 0. Improvements read as regressions, and the eval set could never
tell us what a changed RAG system actually did.

**2. Unanchored LLM grading drift.** v2's LLM grader agreed with the human
gold grades on only **38.8% exact** (37% of pairs were human-0 → LLM-1: the
model over-graded keyword mentions, and 40% of all chunks landed on grade 1).
Scores shifted between judgment versions for reasons unrelated to the
retriever — the baseline itself was unstable.

## The v3 approach: an iteratively-pooled, human-anchored LLM-judged test collection

- **Human grades are frozen seed ground truth** (`eval/v3/seeds/human-judgments.json`)
  and always win: the LLM only fills ids without a human judgment.
- **Anchored LLM grading** (`eval/llm_grade.py`): every batch prompt embeds a
  known gold chunk (human grade 3) and contrast chunk (human grade 0) plus the
  rubric line *"merely mentions an expected entity = grade 0, not 1"*.
  Result: grades on retrieved chunks match the human gold 60/60 (was 2–7 of
  10 under v2), and the grade distribution is back to the human shape.
- **Probe holdout**: ~20% of human-judged ids are also graded by the LLM;
  `eval/agreement.py` reports judge-vs-human agreement after every run
  (pilot: binary 0.866, gate ≥ 0.8) so calibration is measured, not assumed.
- **Run-based judgment cycle** — the fix for the static-pool ceiling:
  1. `eval/score.py` scores a run and trips an **unjudged tripwire** (counts
     retrieved ids missing from the judgment set — a silent zero-score is
     impossible).
  2. `eval/harvest.py` extends the pools with exactly those ids.
  3. `eval/llm_grade.py` **delta-grades only the new ids** (id-level resume).
  4. Re-score: the judgment set has grown to cover whatever the system
     actually retrieves. Verified end-to-end: a simulated retriever drift
     surfaced 2 unjudged chunks (one genuinely relevant, graded 2) — absorbed
     by the cycle, moving prec@10 from 0.80 (pessimistic, unjudged=0) to 0.90
     at a cost of one grading batch.
- **Seed pools go beyond retrieval** (`eval/build_seeds.py`): a wiki link
  graph (infobox/template wikilinks + lead links, junk-capped) plus "gold
  pages" (full pages of human-relevant chunks) recover relevant chunks
  retrieval missed — q039 pool recall went 2/17 → 17/17.

In IR terms: a Cranfield-style graded test collection with **iterative
pooling** and an **anchored LLM-as-a-judge** calibrated against a human gold
standard — the qrels grow with the system instead of freezing and punishing
novelty.

The first validation of the new set is already scoped: re-run the
query-enhancement variant that once looked like a regression and score it on
both eval sets — the static pool should misjudge it; the v3 cycle should
measure it honestly.

## Usage

```bash
# Build seeds + anchors (pilot: --only q002,q005,q013,q023,q025,q039)
python -m eval.build_seeds seeds --only ...
python -m eval.build_seeds build-anchors --only ...

# Score a retriever (prints the unjudged tripwire)
python -m eval.score run --fn contextfromvectors --k 10

# If the tripwire fired: grow the judgment set, delta-grade, re-score
python -m eval.harvest --run eval/v3/runs/<run>.run.json
python -m eval.llm_grade
python -m eval.score score --run <new run> --run <baseline>

# Judge quality
python -m eval.agreement        # LLM-vs-human probe agreement
python -m eval.pilot            # Phase-0 gate report (6 pilot questions)
```

Chunk identity is the deterministic per-chunk UUID from
`src/retriever/indexing.py` (locked chunking parameters); re-indexing with
different parameters invalidates the judgment set.
