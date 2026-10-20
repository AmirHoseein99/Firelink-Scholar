"""Index the structured wiki JSON corpus into the local Qdrant collection.

Run as a script: ``python -m src.retriever.indexing``. Importing this module
has no side effects.

Chunk identity is locked (see eval/__init__.py): ``create_document_id`` is
deterministic over ``(metadata, content)`` and the splitter uses
``chunk_size=1000``, ``chunk_overlap=200``. Changing either invalidates qrels.
"""

import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.retriever.vector_store import (
    get_embeddings,
)
from src.retriever.vector_store import (
    get_vector_store as get_vector_store_singleton,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_DATA_DIR = REPO_ROOT / "data" / "structured"

# Locked chunking parameters; do not change without invalidating eval qrels.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

sections_to_ignore = [
    "id",
    "infobox",
    "infobox_extra",
    "infobox_raw",
    "schema_version",
    "tables",
    "source",
    "entity_type",
]


def create_embeddings() -> OpenAIEmbeddings:
    """Shared embeddings instance (process-wide singleton)."""
    return get_embeddings()


def get_vector_store(embeddings: OpenAIEmbeddings | None = None) -> QdrantVectorStore:
    """Shared vector store (process-wide singleton).

    ``embeddings`` is accepted for backward compatibility but ignored: the
    singleton owns the single QdrantClient allowed on the local storage
    folder. If the collection does not exist yet, the singleton creates it.
    """
    return get_vector_store_singleton()


def create_one_liner_metadata(wikipage_js, section_key: str):
    return f" {wikipage_js['source']['title']} -  {wikipage_js['source']['game']} - {wikipage_js['entity_type']} - {section_key}"


def create_sections(wikipage_js):
    return {
        key: wikipage_js[key]
        for key in list(wikipage_js.keys())
        if key not in sections_to_ignore
    }


def create_metadata(wikipage_js, section_key):
    return {
        "title": wikipage_js["source"]["title"],
        "game": wikipage_js["source"]["game"],
        "category": wikipage_js["source"]["categories"],
        "entity_type": wikipage_js["entity_type"],
        "section": section_key,
    }


def create_document_id(doc: Document) -> str:
    content = doc.page_content
    metadata = json.dumps(
        doc.metadata,
        sort_keys=True,
        ensure_ascii=False,
    )
    value = f"{metadata}:{content}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


def iter_wikipages(data_dir: Path = STRUCTURED_DATA_DIR):
    """Yield (path, parsed json) for every structured wikipage."""
    for path in sorted(data_dir.rglob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            yield path, json.load(f)


def build_documents(wikipage_js):
    """Split a wikipage into chunked Documents with deterministic chunk ids."""
    for section_key, section_content in create_sections(wikipage_js).items():
        if not section_content:
            continue

        items = section_content if isinstance(section_content, list) else [section_content]
        for item in items:
            page_content = (
                f"{create_one_liner_metadata(wikipage_js, section_key)} \n {section_key} : {item}"
            )
            document = Document(
                page_content=page_content,
                metadata=create_metadata(wikipage_js, section_key),
            )
            for split in text_splitter.split_documents([document]):
                split.metadata["document_id"] = create_document_id(split)
                yield split


def index_all(data_dir: Path = STRUCTURED_DATA_DIR, vector_store: QdrantVectorStore | None = None) -> int:
    if vector_store is None:
        vector_store = get_vector_store(create_embeddings())

    total_chunks = 0
    for path, wikipage_js in iter_wikipages(data_dir):
        print(wikipage_js["source"]["title"])
        documents = list(
            {
                document.metadata["document_id"]: document
                for document in build_documents(wikipage_js)
            }.values()
        )
        vector_store.add_documents(
            documents,
            ids=[document.metadata["document_id"] for document in documents],
        )
        total_chunks += len(documents)
    print(f"Indexed {total_chunks} chunks into {settings.QDRANT_PATH}")
    return total_chunks


def smoke_test(vector_store: QdrantVectorStore, embeddings: OpenAIEmbeddings):
    vector = embeddings.embed_query("who was knight artorias?")
    for doc in vector_store.similarity_search_by_vector(vector, k=5):
        print(doc.page_content)
        print(doc.metadata)


if __name__ == "__main__":
    embeddings = create_embeddings()
    store = get_vector_store(embeddings)
    index_all(vector_store=store)
    smoke_test(store, embeddings)
