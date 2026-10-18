"""LLM-judged, run-driven answer-key retrieval benchmark.

Data flow::

    questions.json ──► answers.json (claims + verbatim quotes)
                             │
    retriever run ──► retrieved chunk ids
                             │
                 grade only new retrieved ids
                             │
                        qrels.json (append-only)
                             │
                  score ALL saved runs

There is no pre-built candidate pool and no upfront grading. Qrels are built
only from chunks returned by evaluation runs, so classic relevance metrics are
relative to the accumulated retrieved set. ``claim_coverage@k`` is the stable
absolute metric because its denominator is the answer key.

Chunk identity is the per-chunk UUID produced by ``create_document_id`` in
``src/retriever/indexing.py``. It is deterministic over ``(metadata, content)``
and therefore depends on the locked chunking parameters (``chunk_size=1000``,
``chunk_overlap=200``). Re-indexing with different parameters invalidates the
qrels.

Historical harnesses and data live under ``eval/archive/`` and are not imported
by the active CLI.
"""
