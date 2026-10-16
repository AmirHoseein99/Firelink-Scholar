import atexit
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import settings

logger = logging.getLogger(__name__)

qwen_embeddings = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:3001/v1",
    api_key=settings.freellm_api_key,
    check_embedding_ctx_length=False,
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=qwen_embeddings,
    path=settings.QDRANT_PATH,
    collection_name=settings.COLLECTION_NAME,
)


def _close_qdrant_client() -> None:
    """Close the module-level client before interpreter teardown.

    Without this, the client's __del__ runs during interpreter shutdown
    after module teardown has begun, and qdrant_local.close()'s lazy
    `import portalocker` fails with
    "ImportError: sys.meta_path is None, Python is likely shutting down".
    """
    client = getattr(qdrant, "client", None)
    if client is None:
        return
    try:
        client.close()
    except Exception:  # unactionable during exit; keep shutdown quiet
        pass


atexit.register(_close_qdrant_client)


def contextfromvectors(query: str, k: int = 5):
    """Return ranked [(Document, score), ...] for a query.

    List order is the ranking. Document.metadata["document_id"] carries the
    chunk UUID used by the evaluation set (eval/v3/judgments.json).
    """
    vector = qwen_embeddings.embed_query(query)
    retrieved = qdrant.similarity_search_with_score_by_vector(vector, k=k)

    lines = [f"query={query!r} k={k} -> {len(retrieved)} chunk(s)"]
    for i, (doc, score) in enumerate(retrieved, start=1):
        meta = doc.metadata or {}
        preview = " ".join(doc.page_content.split())[:100]
        lines.append(
            f"  [{i}] score={score:.3f} title={meta.get('title')!r} "
            f"section={meta.get('section')!r} id={meta.get('document_id')} "
            f"text={preview!r}"
        )
    logger.info("retrieved chunks:\n%s", "\n".join(lines))

    return retrieved
