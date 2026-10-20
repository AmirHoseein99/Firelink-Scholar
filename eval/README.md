# Retrieval benchmark

This directory is the answer-key, run-driven evaluation harness. It replaces
the retired v1–v4 pooled benchmarks, which are preserved under
[`archive/`](archive/README.md).

The benchmark answers one question: **for each natural-language question, how
well does a retriever surface chunks that support a frozen, verifiable answer
key?**

## Why the old benchmarks were retired

Every old version graded a pool built before retrieval ran. A frozen pool caps
recall: a query-enhancement change that retrieves a different, correct chunk
was scored as a regression because that chunk was never in the pool. v1 used
human grades, v2/v4 graded pools up front, and v3 mixed human anchors with two
contradictory LLM-grading modes. None of those designs can compare retrieval
systems fairly when the systems retrieve different chunks.

## Data flow

```text
data/structured/*.json
        │  (claims + verbatim quotes, game-filtered candidates)
        ▼
  eval/answers.json  ──────────────► frozen answer keys
        │
retriever run ──► retrieved chunk ids
        │
        ▼
grade only ids not already in qrels  (append-only union)
        │
        ▼
  eval/qrels.json  ──► re-score EVERY saved run file
        │
        ▼
  eval/reports/scores.json
```

Qrels are never pre-built. A chunk is graded only because some evaluation run
retrieved it. Re-scoring every run after each grading pass means a run's score
is always computed against the same qrels snapshot, regardless of the order in
which systems were run.

## Files

| Path | Purpose |
| --- | --- |
| `questions.json` | 100 questions: `id`, `question`, `game`, `category`, `subcategory`, `expected_entities` |
| `dataset.json` | Pinned benchmark definition: model, prompt hashes, systems, `k`, metric cutoffs, qrels hash |
| `answers.json` | Frozen answer keys: claims, verbatim quotes, source path/section, verification status |
| `qrels.json` | Append-only grades for retrieved chunk ids, plus supported claim ids |
| `runs/<system>_k<k>.run.json` | Saved retrieval runs and refreshed metrics |
| `reports/question_review.json` | Question validator output |
| `reports/scores.json` | Latest comparison of every saved run |
| `archive/` | Retired harnesses and historical data; not imported by the CLI |

`data/structured` is the source of truth for answer keys. Candidate sections
are filtered to the question's game (`All` means cross-game), selected by
expected-entity/title matching plus BM25, and capped at 25 sections
(40 for cross-game questions).

## Quick start

```bash
# 1. Validate the frozen question set and the current data
python -m eval validate --questions-only
python -m eval status

# 2. Build answer keys (default backend: Free-LLM proxy on :3001)
python -m eval build-answers --only q001,q002,q009
python -m eval build-answers

# 3. Run one system or all systems, at the report cutoff k=10
python -m eval run --systems dense --k 10
python -m eval run --systems all --k 10

# 4. Choose a judge, then grade the union of new ids and re-score all runs.
#    No grading happens until this is set explicitly.
python -m eval configure-judge --model <judge-model>
python -m eval evaluate --systems all --k 10
#    or grade existing runs only, then score separately
python -m eval grade --run dense_k10.run.json --run bm25_k10.run.json
python -m eval score

# 5. Inspect
python -m eval status
python -m eval validate --completeness
```

Every command accepts `--only q001,q002` and `--limit N` for pilots.
`build-answers` also accepts `--start qNNN` and `--force`. `evaluate` and
`grade` accept `--force-judge`, `--allow-unverified`, `--batch-size`, and
`--concurrency` (default 1).

`--limit` without `--only` runs the first N questions. Pilot runs are merged
into the saved run file for those question ids, so they never silently replace
a full run.

Commands log to stderr at INFO: systems started, per-question grading progress,
missing/unparsed ids, scoring, and written files. `-v` adds DEBUG detail
(per-question retrieval counts, batch calls, id filtering, HTTP retries); `-q`
keeps warnings and errors only. Use `--log-level` for an explicit level and
`--log-file PATH` to append logs to a file. Flags may go before or after the
subcommand:

```bash
python -m eval run --systems dense --k 10 -v
python -m eval grade --run dense_k10 --log-file eval/reports/grade.log
```

## Answer keys

Each claim has:

- `text` — one self-contained fact;
- `importance` — `core` or `supporting`;
- `quote` — **verbatim** text from the cited section (whitespace changes only);
- `source.path` / `source.section` — a section in `data/structured`.

The builder resolves quotes back to the cited section and drops any claim whose
quote is not a whitespace-normalized substring. An answer is `verified` only
when it has at least three verified claims including a core claim. Otherwise it
is `needs_review`. Grading refuses unverified answers unless
`--allow-unverified` is passed explicitly, so a review flag cannot silently
enter qrels.

Single-game questions allow up to 8 claims; cross-game synthesis allows 12.

Expected entities are matched against page titles in the question's game. The
short-entity rule from the old harness is preserved: entities shorter than five
characters match exactly only. This is why `Sif` must be written
`Great Grey Wolf Sif` and `Gwyn` must not be used for DS1 (which has no Gwyn
page), and why `Dark` cannot flood a candidate set with every "Dark Souls"
page. Question cleanup and entity resolutions are recorded in
`reports/question_review.json`; the pre-cleanup file is
`archive/questions.before-cleanup.json`.

## Systems

| Name | Function | Notes |
| --- | --- | --- |
| `dense` | `contextfromvectors` | Plain dense retrieval |
| `enhanced` | `contextfromvectors_enhanced` | LLM query rewrite; non-deterministic and slow |
| `bm25` | `contextfromvectors_bm25` | BM25; first call builds the index |
| `hybrid` | `contextfromvectors_hybrid` | RRFs BM25 with **enhanced**, not with plain dense |

`k` is always passed explicitly. The systems are evaluated at `k=10` and
reported at `k=5`, `10`, and `20` (see `dataset.json`).

## Grading rubric

The judge grades each retrieved chunk against the answer-key claims, not
against entity mentions:

| Grade | Meaning |
| --- | --- |
| 3 | Directly states or supports a core claim |
| 2 | Substantially supports a claim, or partially supports a core claim |
| 1 | Relevant background that mentions expected entities but supports no claim |
| 0 | Unrelated or contradicts the answer key |

A correct, question-relevant chunk that is missing from the answer key is not
forced to 0; the judge grades it on the rubric and records
`other_relevant: true`. Graded chunks with grade 2 or 3 record the claim ids
they support in `supports_claims`.

### Judge pinning

`dataset.json` pins **one** judge model and the exact grading prompt hash. The
judge is intentionally **unconfigured** in the shipped dataset, because the
judge has not been chosen yet. Until `python -m eval configure-judge --model
<model>` is run, `grade` and `evaluate` fail before touching any answer,
chunk, or qrels file. `configure-judge` only writes the pin; it does not grade.

Once configured, `grade`/`evaluate` refuse to run when the backend model or
prompt hash differs, or when existing qrels carry a different `graded_by`
value. `--force-judge` overrides the gate and writes the new judge metadata on
the entries it touches; old grades are never silently rewritten. Historical v4
data is mixed-judge and is therefore not migrated into `qrels.json`.

The answer builder uses the Free-LLM proxy `@cf/qwen/qwen3.8-27b`. The judge is
a separate choice and may be a different model family; record it with
`configure-judge` before building qrels.

## Metrics

For every question and saved run:

- `ndcg@k` — graded relevance, using the full qrels ideal;
- `recall@k` / `mrr@k` / `precision@k` — binary relevance at grade ≥ 2;
- `claim_coverage@k` — fraction of **core** answer-key claims supported by at
  least one top-k chunk;
- `claim_coverage_all@k` — same, for core + supporting claims.

Unjudged retrieved ids score 0 and are counted per question. A run is
`partial: true` when any retrieved id is ungraded; `python -m eval score
--strict` exits 1 in that case. This is the old unjudged tripwire: the number
is never silently inflated.

**Important limitation.** Qrels contain only ids that evaluation runs
retrieved, so nDCG/recall/MRR/precision are relative to the accumulated
retrieved set. A system that retrieves genuinely new relevant chunks is not
punished (they are graded and added), but a system is still scored only against
chunks someone has retrieved. `claim_coverage@k` has the answer key as its
denominator and is the stable absolute metric. Do not compare recall against a
fixed corpus total; it is not that kind of collection.

## Chunk identity

The qrels key on the per-chunk UUID produced by
`src/retriever/indexing.py::create_document_id`, which hashes the metadata dict
and chunk text. The chunk id therefore depends on the locked chunking contract
(`chunk_size=1000`, `chunk_overlap=200`, `add_start_index=True`) and on the
section metadata shape. **Do not re-index with different chunking parameters
without rebuilding qrels.** Eval code never recomputes ids; it looks them up in
Qdrant.

`src/retriever/vector_store.py` owns the process-wide Qdrant client because the
local store takes an exclusive lock. Do not construct `QdrantClient` in eval
code, and do not run two eval commands that open Qdrant concurrently.

## Validation

`python -m eval validate` checks:

- question schema, ids, duplicates, allowed report categories, over-long
  compound questions, entity count, generic entities, and same-game entity
  resolution;
- every answer quote is grounded in the cited `data/structured` section;
- every qrels/run id still exists in Qdrant;
- every qrels entry matches the pinned judge and prompt hash;
- `questions.json` and `qrels.json` match the hashes in `dataset.json`.

`--questions-only` is fast and offline; `--completeness` additionally requires
a verified answer key and at least one qrels judgment for every question.

Report categories are the six groups `lore`, `boss`, `item_location`,
`mechanics_build`, `quests_exploration`, and `cross_game`. The old fine-grained
label is preserved as `subcategory`.

## Configuration

The Free-LLM proxy is the default backend for both the answer builder and the
judge:

```dotenv
FREELLM_BASE_URL=http://localhost:3001/v1
FREELLM_MODEL=@cf/qwen/qwen3.8-27b
FREELLM_API_KEY=
```

The proxy on `:3001` also serves `qwen3-embedding-0.6b` for dense retrieval.
`NEURALWATT_*` and `OPENROUTER_*` remain available as explicitly selected
backends; `FREELLM_*` values are no longer silently read from `OPENROUTER_*`.

## History

`archive/` contains the v1–v4 questions, pools, runs, judgments, reports,
retired scripts, and their old tests. Nothing there is imported by
`python -m eval`, and `pytest tests/eval` collects only the new tests.
