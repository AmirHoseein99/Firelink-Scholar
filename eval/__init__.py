"""Retrieval evaluation set for the Dark Souls lore corpus.

Contract:
  - ``questions.json`` is the frozen question list. Do not edit once v1 is
    graded; iteration on the retriever must keep questions + qrels constant.
  - ``build_candidates.py`` produces a per-question candidate pool
    (``v1/candidates/qNNN.json``) by unioning dense retrieval, BM25 retrieval,
    and a title-entity filter seeded from each question's
    ``expected_entities``. Pool depths and the final cap are configurable:
    v1 used ``--k 50`` (implicit ``--cap 80``); v2 uses ``--dense-k 100
    --bm25-k 100 --cap 250``.
  - ``annotate.py`` is the interactive CLI for manual grading on the integer
    0..3 scale (3 directly answers, 2 substantial support, 1 tangential
    mention, 0 unrelated).
  - ``aggregate.py`` validates chunk UUIDs against Qdrant and emits the
    canonical ``v1/qrels.json``.
  - v2 flow (LLM-graded deeper pools): ``llm_grade.py`` batch-grades the
    ``v2/candidates`` pools with ``glm-5.3-flash`` via NeuralWatt
    (``NEURALWATT_*`` settings; resumable, atomic per-question flush).
    ``merge_qrels.py`` overlays the human ``v1/qrels.final.json`` grades
    (human wins) on the LLM grades, validates against Qdrant, writes
    ``v2/qrels.json`` in the ``qrels.final.json`` schema plus provenance, and
    emits an LLM-vs-human agreement report to ``v2/agreement.json``.
    ``score.py`` scores retrievers against either qrels file (unjudged = 0).

Chunk identity is the per-chunk UUID produced by ``create_document_id`` in
``src/retriever/indexing.py``. It is deterministic over ``(metadata, content)``
and therefore depends on the locked chunking parameters
(``chunk_size=1000``, ``chunk_overlap=200``). Re-indexing with different
parameters invalidates the qrels.
"""
