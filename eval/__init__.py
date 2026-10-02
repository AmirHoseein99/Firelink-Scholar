"""Retrieval evaluation set for the Dark Souls lore corpus.

Contract:
  - ``questions.json`` is the frozen question list. Do not edit once v1 is
    graded; iteration on the retriever must keep questions + qrels constant.
  - ``build_candidates.py`` produces a per-question candidate pool
    (``v1/candidates/qNNN.json``) by unioning dense retrieval (k=50), BM25
    retrieval (k=50), and a title-entity filter seeded from each question's
    ``expected_entities``.
  - ``annotate.py`` is the interactive CLI for manual grading on the integer
    0..3 scale defined in the planning doc.
  - ``aggregate.py`` validates chunk UUIDs against Qdrant and emits the
    canonical ``v1/qrels.json``.

Chunk identity is the per-chunk UUID produced by ``create_document_id`` in
``src/retriever/indexing.py``. It is deterministic over ``(metadata, content)``
and therefore depends on the locked chunking parameters
(``chunk_size=1000``, ``chunk_overlap=200``). Re-indexing with different
parameters invalidates the qrels.
"""
