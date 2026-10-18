import logging
from typing import List

from src.retriever.enhancement import enhance_query
from src.retriever.sparse_retrieval import bm25_retrieval
from src.retriever.vector_store import get_embeddings, get_vector_store


logger = logging.getLogger(__name__)

qwen_embeddings = get_embeddings()

qdrant = get_vector_store()

RRF_CONSTANT = 60


def reciprocal_rank_score(ranked_lists: List[List], k: int = RRF_CONSTANT):
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ranked list per retrieval system, each of
            ``(Document, score)`` pairs. Ranks start at 1 within each list.
        k : RRF smoothing constant (not the number of results).
    Returns:
        list of (document_id, rrf_score) sorted by descending score.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            doc_id = document[0].metadata.get("document_id")
            scores[doc_id] = 1.0 / (k + rank) + scores.get(doc_id, 0.0)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def contextfromvectors(query: str, k: int = 50,):
    """Return ranked [(Document, score), ...] for a query.

    List order is the ranking. Document.metadata["document_id"] carries the
    chunk UUID used by the evaluation set (eval/v3/judgments.json).
    """
    vector_db = get_vector_store()
    qwen_embeddings = get_embeddings()
    vector = qwen_embeddings.embed_query(query)
    retrieved = vector_db.similarity_search_with_score_by_vector(vector, k=k)

    return retrieved


def contextfromvectors_enhanced(query: str, k: int = 5):
    """Rewrite the query with an LLM (query enhancement), then retrieve.

    Same contract as contextfromvectors. Falls back to the raw query when
    enhancement fails, so it degrades to the baseline retriever.
    """
    

    return contextfromvectors(enhance_query(query), k=k)

def contextfromvectors_bm25(query: str, k: int = 5):
    """BM25-only retrieval (sparse baseline).

    Same contract as contextfromvectors: ranked [(Document, score), ...]
    with Document.metadata["document_id"] carrying the chunk UUID.
    """

    return bm25_retrieval(query=query, k=k)


def contextfromvectors_hybrid(query: str, k: int = 5):
    """
    Hybrid retrieval: combine sparse and dense retrieval results.

    """

    sparse_results = bm25_retrieval(query=query, k=k)
    dense_results = contextfromvectors_enhanced(
        query=query, k=k
        
    )

    rrf_rankings = reciprocal_rank_score(
        [sparse_results, dense_results], k=RRF_CONSTANT
    )[:k]

    logger.debug("Hybrid retrieval: %d unique documents fused.", len(rrf_rankings))

    documents_by_id = {}
    for doc, _score in sparse_results + dense_results:
        documents_by_id.setdefault(doc.metadata.get("document_id"), (doc, _score))

    return [(documents_by_id[doc_id][0], score) for doc_id, score in rrf_rankings]
    