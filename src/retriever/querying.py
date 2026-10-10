from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import settings

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


def contextfromvectors(query: str, k: int = 5):
    """Return ranked [(Document, score), ...] for a query.

    List order is the ranking. Document.metadata["document_id"] carries the
    chunk UUID used by the evaluation set (eval/v3/judgments.json).
    """
    vector = qwen_embeddings.embed_query(query)
    return qdrant.similarity_search_with_score_by_vector(vector, k=k)
