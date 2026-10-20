import logging

from src.retriever.enhancement import enhance_queries
from src.retriever.sparse_retrieval import bm25_retrieval
from src.retriever.vector_store import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)

RRF_CONSTANT = 60
HYBRID_POOL_FACTOR = 3
DENSE_FUSION_WEIGHT = 3.0
BM25_FUSION_WEIGHT = 1.0
PARAPHRASE_FUSION_WEIGHT = 0.1


def reciprocal_rank_score(
    ranked_lists: list[list],
    k: int = RRF_CONSTANT,
    weights: list[float] | None = None,
):
    """Fuse multiple ranked lists with weighted Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ranked list per retrieval system, each of
            ``(Document, score)`` pairs. Ranks start at 1 within each list.
        k : RRF smoothing constant (not the number of results).
        weights: Optional per-list weight. Defaults to 1.0 for every list.
    Returns:
        list of (document_id, rrf_score) sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights and ranked_lists must have the same length")

    scores: dict[str, float] = {}
    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, document in enumerate(ranked_list, start=1):
            doc_id = document[0].metadata.get("document_id")
            if not doc_id:
                continue
            scores[doc_id] = weight / (k + rank) + scores.get(doc_id, 0.0)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fuse(
    ranked_lists: list[list],
    k: int,
    weights: list[float] | None = None,
) -> list[tuple[object, float]]:
    """Weighted-RRF fuse ranked lists and return the top ``k`` documents."""
    fused = reciprocal_rank_score(ranked_lists, weights=weights)[:k]
    documents_by_id: dict[str, tuple[object, float]] = {}
    for ranked_list in ranked_lists:
        for document, score in ranked_list:
            doc_id = document.metadata.get("document_id")
            if doc_id:
                documents_by_id.setdefault(doc_id, (document, score))

    results = []
    for doc_id, rrf_score in fused:
        document = documents_by_id.get(doc_id)
        if document is not None:
            results.append((document[0], rrf_score))
    return results


def contextfromvectors(query: str, k: int = 50):
    """Return ranked [(Document, score), ...] for a query.

    List order is the ranking. Document.metadata["document_id"] carries the
    chunk UUID used by the evaluation set.
    """
    vector_db = get_vector_store()
    embeddings = get_embeddings()
    vector = embeddings.embed_query(query)
    return vector_db.similarity_search_with_score_by_vector(vector, k=k)


def contextfromvectors_enhanced(
    query: str,
    k: int = 5,
    *,
    use_rewrites: bool = False,
):
    """Dense retrieval with optional multi-query rewriting.

    Measured against the current embedding model, dense rewrites consistently
    hurt ranking quality: the original natural-language question is the best
    dense query. Rewrites are therefore disabled by default. Set
    ``use_rewrites=True`` to fuse the original query with the rewrites; the
    original list keeps a dominant weight so rewrites can only make small
    changes to the ranking.
    """
    if not use_rewrites:
        return contextfromvectors(query=query, k=k)

    queries = enhance_queries(query)
    queries = [item for item in queries if item.casefold() != query.casefold()]
    if not queries:
        return contextfromvectors(query=query, k=k)

    ranked_lists = [contextfromvectors(query=query, k=k)]
    ranked_lists.extend(contextfromvectors(query=item, k=k) for item in queries)
    weights = [1.0] + [0.05] * len(queries)
    logger.debug(
        "Enhanced dense retrieval fused %d rewrite list(s): %r",
        len(queries),
        queries,
    )
    return _fuse(ranked_lists, k, weights)


def contextfromvectors_bm25(query: str, k: int = 5):
    """BM25-only retrieval (sparse baseline).

    Same contract as contextfromvectors: ranked [(Document, score), ...]
    with Document.metadata["document_id"] carrying the chunk UUID.
    """
    return bm25_retrieval(query=query, k=k)


def contextfromvectors_hybrid(query: str, k: int = 5):
    """Fuse BM25 with plain dense retrieval using RRF."""
    candidate_k = k * HYBRID_POOL_FACTOR
    sparse_results = bm25_retrieval(query=query, k=candidate_k)
    dense_results = contextfromvectors(query=query, k=candidate_k)
    return _fuse([sparse_results, dense_results], k)


def contextfromvectors_hybrid_enhanced(query: str, k: int = 5):
    """Fuse dense retrieval with BM25 over the original and rewritten queries.

    Dense retrieval stays on the original question because dense rewrites hurt
    ranking. The rewrites are used lexically instead: BM25 gets the original
    query plus each entity/step-focused rewrite, which recovers named chunks
    that the natural-language dense query misses. The dense list keeps the
    largest weight so the baseline ordering dominates.
    """
    candidate_k = k * HYBRID_POOL_FACTOR
    rewrites = [
        item
        for item in enhance_queries(query)
        if item.casefold() != query.casefold()
    ]

    dense_results = contextfromvectors(query=query, k=candidate_k)
    sparse_results = [bm25_retrieval(query=query, k=candidate_k)]
    sparse_results.extend(
        bm25_retrieval(query=item, k=candidate_k) for item in rewrites
    )
    weights = [DENSE_FUSION_WEIGHT, BM25_FUSION_WEIGHT] + [
        PARAPHRASE_FUSION_WEIGHT
    ] * len(rewrites)

    logger.debug(
        "Hybrid enhanced retrieval: dense(original) + %d BM25 list(s)",
        len(sparse_results),
    )
    return _fuse([dense_results, *sparse_results], k, weights)
