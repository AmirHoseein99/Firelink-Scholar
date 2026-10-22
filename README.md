# Firelink Scholar

*Come, sit beside the flame. Ask, and the ashes shall speak.*

A retrieval-augmented generation (RAG) question-answering system over the
[Dark Souls Fandom wiki](https://darksouls.fandom.com). It crawls the wiki,
cleans the wikitext into a structured corpus, indexes it into a local Qdrant
vector store, and answers questions with an LLM grounded in retrieved chunks.

## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Crawl | `src/crawler` | Fetches wiki pages (and optionally images) into `data/raw` |
| Clean | `src/cleaner` | Parses wikitext into structured JSON in `data/cleaned` |
| Refine | `src/refiner` | Normalises prose/tables into the final corpus in `data/structured` |
| Index | `src/retriever/indexing.py` | Chunks (1000/200, locked for eval) and embeds into local Qdrant |
| Retrieve | `src/retriever/querying.py` | Vector similarity search (`contextfromvectors`) |
| Answer | `src/retriever/generation.py` | Prompt + LLM call (any OpenAI-compatible endpoint) |

## Setup

```bash
uv sync
cp .env.example .env   # fill in LLM endpoint + keys; embeddings expect a local OpenAI-compatible server on :3001
```

## Usage

```bash
# Build the corpus
python -m src.crawler --raw-output data/raw --limit 100
python -m src.cleaner --input data/raw --output data/cleaned
python -m src.refiner --cleaned-root data/cleaned --structured-root data/structured

# Index and ask
python -m src.retriever.indexing
python -m src.retriever.generation
```

## Evaluation

The retriever is measured with a **run-driven answer-key benchmark**: answer
keys are frozen claims with verbatim quotes, and the LLM judge grades only
chunk ids that evaluation runs actually retrieve. Grading is append-only and
union-based; after each pass every saved run is re-scored against the same
qrels snapshot. The active benchmark replaces the retired v1–v4 pooled
harnesses, which are archived under `eval/archive/`.

The question set contains 100 natural-language questions spanning six report
categories (`lore`, `boss`, `item_location`, `mechanics_build`,
`quests_exploration`, and `cross_game`). Each answer key is built from the
structured corpus as a set of self-contained claims, each backed by a verbatim
quote from a cited section. An answer only becomes `verified` when it has at
least three verified claims including a core claim; unverified answers are
excluded from grading unless explicitly allowed. Questions and answer keys are
hashed in `eval/dataset.json` so a benchmark result can be tied to an exact
corpus and answer-key state.

Five retrieval systems are compared: plain dense, a dense-rewrite control
(`enhanced`), BM25, an RRF hybrid of BM25 with plain dense, and an enhanced
hybrid. The enhanced hybrid keeps dense retrieval on the original
natural-language question and uses the LLM rewrites lexically: BM25 runs over
the original question plus the combined and entity/step-focused rewrites, and
the lists are fused with a dense-dominant weighted RRF. Dense rewrites are
disabled by default because they degraded ranking with the current embedding
model. Retrieval depth is configurable; metrics are reported at cutoffs `k=5`,
`10`, and `20`. For each question the report includes
`ndcg@k`, `recall@k`, `mrr@k`, `precision@k`, and two claim-based measures:
`claim_coverage@k` over core claims and `claim_coverage_all@k` over core plus
supporting claims. Retrieved chunks that have not been graded count as 0 and
mark the run `partial`, so `python -m eval score --strict` can fail rather than
let an inflated number through.

### Results at retrieval depth k=30

Scores from `eval/reports/scores.csv` for the k=30 systems (100 questions, all
retrieved chunks judged, `partial=False`). Higher is better. `enhanced` has no
k=30 run; its k=10 and k=100 results are in the CSV.

| System | nDCG@5 | nDCG@10 | nDCG@20 | MRR@10 | claim_coverage@5 | claim_coverage@10 | claim_coverage@20 | recall@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bm25_k30` | 0.447 | 0.430 | 0.438 | 0.728 | 0.447 | 0.588 | 0.725 | 0.222 |
| `bm25_reranked_k30` | 0.648 | 0.618 | 0.604 | **0.916** | 0.601 | 0.754 | 0.828 | 0.318 |
| `dense_k30` | 0.542 | 0.533 | 0.531 | 0.807 | 0.533 | 0.693 | 0.803 | 0.276 |
| `dense_reranked_k30` | 0.648 | 0.629 | 0.626 | 0.908 | 0.613 | **0.789** | **0.863** | 0.331 |
| `hybrid_enhanced_k30` | 0.450 | 0.427 | 0.431 | 0.724 | 0.454 | 0.586 | 0.722 | 0.223 |
| `hybrid_k30` | 0.535 | 0.525 | 0.535 | 0.754 | 0.556 | 0.703 | 0.827 | 0.286 |
| `hybrid_reranked_k30` | **0.649** | **0.633** | **0.637** | 0.913 | **0.618** | 0.769 | 0.862 | **0.357** |

Reranking is the largest single gain in every family. `hybrid_reranked_k30` is
the best overall ranking quality, `dense_reranked_k30` leads on claim coverage,
and `bm25_reranked_k30` has the highest MRR but the weakest recall. The
`hybrid_enhanced` variant underperforms plain `hybrid` at this depth.

Two properties make results comparable across systems. First, the judge model
and grading-prompt hash are pinned; grading refuses to run until a judge is
chosen with `python -m eval configure-judge --model <model>`, and mixed judges
are rejected. Second, qrels key on the chunk UUIDs produced by the indexing
pipeline, so the chunking contract (`chunk_size=1000`, `chunk_overlap=200`,
`add_start_index=True`) is locked for evaluation; re-indexing with different
parameters requires rebuilding qrels. Because qrels accumulate only retrieved
chunks, nDCG/recall/MRR/precision are relative to that accumulated set, while
`claim_coverage@k` uses the answer key as its denominator and is the stable
absolute metric.

```bash
python -m eval status                 # questions, answers, qrels, runs
python -m eval build-answers          # freeze answer keys (Free-LLM)
python -m eval run --systems all --k 10
python -m eval configure-judge --model <judge-model>   # no grading yet
python -m eval evaluate --systems all --k 10   # grade new ids + re-score all
python -m eval score --strict         # fail if any retrieved id is ungraded
```

See [eval/README.md](eval/README.md) for the rubric, metrics, judge pinning,
chunk-identity lock, and the limitation that recall is relative to the
accumulated retrieved-and-judged set (`claim_coverage@k` is the stable
absolute metric).
