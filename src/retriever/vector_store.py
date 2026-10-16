"""Process-wide singleton access to the local Qdrant vector store.

Qdrant's local mode takes an exclusive file lock on the storage folder
(``settings.QDRANT_PATH``). Opening a second ``QdrantClient`` on the same
path -- even within the same process -- fails with::

    RuntimeError: Storage folder ... is already accessed by another instance
    of Qdrant client. If you require concurrent access, use Qdrant server
    instead.

Every module that needs the vector store, the Qdrant client, or the
embeddings must go through :func:`get_vector_store`, :func:`get_qdrant_client`
and :func:`get_embeddings`. This module guarantees exactly one instance of
each per process, so the exclusive lock is taken only once.

Note: the singleton is per process. Running two *separate* scripts (e.g.
``indexing`` and ``generation``) at the same time still conflicts, because
Qdrant local mode does not support concurrent access across processes at all.
"""

import atexit
import threading

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.config import settings

_lock = threading.RLock()

_embeddings: OpenAIEmbeddings | None = None
_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Return the shared embeddings instance for this process."""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = OpenAIEmbeddings(
                    model="qwen3-embedding-0.6b",
                    base_url="http://localhost:3001/v1",
                    api_key=settings.freellm_api_key,
                    check_embedding_ctx_length=False,
                )
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    """Return the single ``QdrantClient`` for this process.

    Only this function may construct a client for the local storage folder;
    a second client would fail on the exclusive file lock.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(path=settings.QDRANT_PATH)
                atexit.register(_close_client)
    return _client


def get_vector_store() -> QdrantVectorStore:
    """Return the shared ``QdrantVectorStore`` over the configured collection.

    If the collection does not exist yet (first run, before indexing), it is
    created here on the shared client -- same dense/COSINE config that
    ``QdrantVectorStore.from_documents([], ...)`` used to produce, so no
    second client (and no second lock) is ever needed.
    """
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                client = get_qdrant_client()
                embeddings = get_embeddings()
                if not client.collection_exists(settings.COLLECTION_NAME):
                    size = len(embeddings.embed_documents(["dummy_text"])[0])
                    client.create_collection(
                        collection_name=settings.COLLECTION_NAME,
                        vectors_config={
                            QdrantVectorStore.VECTOR_NAME: models.VectorParams(
                                size=size,
                                distance=models.Distance.COSINE,
                            )
                        },
                    )
                _vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=settings.COLLECTION_NAME,
                    embedding=embeddings,
                    # The collection was created by indexing.py with the same
                    # embedding model; skip the constructor's validation, which
                    # calls the embedding server and would make merely opening
                    # the store fail while the server is down.
                    validate_embeddings=False,
                    validate_collection_config=False,
                )
    return _vector_store


def _close_client() -> None:
    """Release the storage lock before interpreter teardown.

    Without this, the client's ``__del__`` runs during interpreter shutdown
    after module teardown has begun, and ``qdrant_local.close()``'s lazy
    ``import portalocker`` fails with
    "ImportError: sys.meta_path is None, Python is likely shutting down".
    """
    if _client is None:
        return
    try:
        _client.close()
    except Exception:  # unactionable during exit; keep shutdown quiet
        pass
