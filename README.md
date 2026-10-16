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

The retriever is measured against a **v3 test collection**: human-anchored
LLM judging with iterative pooling, so the judgment set grows to cover
whatever the system actually retrieves instead of punishing novelty. See
[eval/README.md](eval/README.md) for the full methodology and commands.

```bash
python -m eval.score run --fn contextfromvectors --k 10   # score + unjudged tripwire
python -m eval.agreement                                  # LLM-vs-human agreement
```
