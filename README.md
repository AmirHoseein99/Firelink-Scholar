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
