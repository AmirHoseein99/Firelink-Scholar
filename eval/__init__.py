"""Retrieval evaluation set for the Dark Souls lore corpus (v3).

Contract:
  - ``questions.json`` is the frozen question list. Do not edit; iteration on
    the retriever must keep questions + judgments stable.
  - ``v3/seeds/human-judgments.json`` is the archived v1 human ground truth
    (verbatim move of the old ``v1/qrels.final.json``). Human grades are never
    overwritten: in ``v3/judgments.json`` a human judgment always wins.
  - ``build_seeds.py`` builds the machinery:
      * ``v3/seeds/link-index.json`` — normalized entity -> pages linking to
        it (infobox/template-parameter wikilinks from ``data/processed``
        wikitext + lead-section links from ``data/cleaned`` markdown;
        denylist/navbox templates excluded; short entities match exactly
        only; the template tier is junk-capped at 80 ids/question).
      * ``v3/candidates/qNNN.json`` — seed pools ordered human -> template ->
        gold (full pages of human grade >= 2 chunks) -> dense-50 -> BM25-50,
        capped (human ids always kept).
      * ``v3/seeds/anchors.json`` — per question a gold anchor (human grade 3,
        lowest doc_id), a contrast anchor (human grade 0, lowest doc_id), and
        the probe holdout (every 5th human-judged id). Anchors are excluded
        from LLM grading; probes are graded by both sides.
  - ``llm_grade.py`` batch-grades only ids without a judgment (id-level
    resume: re-running retries just the missing ids), with the anchors and
    the "merely mentions an expected entity = grade 0, not 1" rubric line in
    every batch prompt, and merges into ``v3/judgments.json`` (schema:
    judgments + probe_llm + sources counters + grade_counts).
  - ``score.py`` scores retrievers against ``v3/judgments.json`` (unjudged
    counts as 0) and trips a visible ``n_unjudged`` counter per question.
  - The judgment cycle: run a retriever, ``harvest`` the run file to extend
    pools with unjudged ids, delta-grade with ``llm_grade``, re-score. New
    retrieved chunks can always earn credit — no static-pool ceiling.
  - ``pilot.py`` (Phase 0 gate) and ``agreement.py`` (LLM-vs-human probe
    report) live under ``eval/``; reports are written to ``v3/reports/``.

Chunk identity is the per-chunk UUID produced by ``create_document_id`` in
``src/retriever/indexing.py``. It is deterministic over ``(metadata, content)``
and therefore depends on the locked chunking parameters
(``chunk_size=1000``, ``chunk_overlap=200``). Re-indexing with different
parameters invalidates the judgments.
"""
