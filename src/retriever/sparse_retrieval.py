import logging
import re

from rank_bm25 import BM25Okapi

from src.retriever.indexing import REPO_ROOT, build_documents, iter_wikipages

logger = logging.getLogger(__name__)

STRUCTURED_DATA_DIR = REPO_ROOT / "data" / "structured"

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, dropping punctuation."""
    return _TOKEN_RE.findall(text.lower())


def _build_bm25_index(data_dir=None):
    """Build the BM25 index and the matching documents list."""
    if data_dir is None:
        data_dir = STRUCTURED_DATA_DIR
    documents = []
    for _, wikipage_js in iter_wikipages(data_dir):
        documents.extend(list(build_documents(wikipage_js)))

    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, documents


_bm25_cache: dict = {}


def get_bm25_index(data_dir=None):
    """Return the cached (bm25, documents) pair, building it on first use."""
    if data_dir is None:
        data_dir = STRUCTURED_DATA_DIR
    if data_dir not in _bm25_cache:
        logger.info("Building BM25 index for %s ...", data_dir)
        _bm25_cache[data_dir] = _build_bm25_index(data_dir)
    return _bm25_cache[data_dir]


def bm25_retrieval(query, k=50):
    """
    Retrieve the top k documents from the corpus based on BM25 scores for the given query.

    Args:
        query (str): The input query string.
        k (int): The number of top documents to retrieve.
    """

    bm25, documents = get_bm25_index()

    # Tokenize the query
    query_tokens = tokenize(query)

    # Get BM25 scores for the query
    scores = bm25.get_scores(query_tokens)
    # Get the top k document indices based on scores
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Retrieve the top k documents and their scores
    top_k_documents = [(documents[i], scores[i]) for i in top_k_indices]

    return top_k_documents
