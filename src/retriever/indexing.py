from email.mime import text

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.indexing import index
import os
from pathlib import Path
import json
from pydantic import ValidationError
import hashlib
import uuid

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
from langchain_openai import OpenAIEmbeddings
from src.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

qwen_embeddings = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:3001/v1",
    api_key=settings.freellm_api_key,
    check_embedding_ctx_length=False,
)

qdrant = QdrantVectorStore.from_documents(
    [],
    qwen_embeddings,
    path=settings.QDRANT_PATH,
    collection_name=settings.COLLECTION_NAME,
)

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


def create_document(metadata, content):
    try:
        return Document(
            page_content=content,
            metadata=metadata,
        )
    except ValidationError as e:
        print(f"ValidationError: {e} in file {wikipage_js.get('source', 'unknown')}")
        raise


def create_document_id(doc: Document) -> str:
    content = doc.page_content
    metadata = json.dumps(
        doc.metadata,
        sort_keys=True,
        ensure_ascii=False,
    )
    value = f"{metadata}:{content}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


for wikipage in os.walk(Path(__file__).parent.parent.parent / "data" / "structured"):
    for file in wikipage[2]:
        if file.endswith(".json"):
            with open(os.path.join(wikipage[0], file), "r", encoding="utf-8") as f:
                wikipage_js = json.load(f)
                print(wikipage_js["source"]["title"])
                section_to_document = create_sections(wikipage_js)
                for section_key, section_content in section_to_document.items():
                    if section_content:
                        if type(section_content) == list:
                            for section_content_item in section_content:
                                page_content = f"{create_one_liner_metadata(wikipage_js, section_key)} \n {section_key} : {section_content_item}"
                                metadata = create_metadata(wikipage_js, section_key)
                                document = create_document(metadata, page_content)

                                all_splits = text_splitter.split_documents([document])

                                splits_ids = []
                                for split in all_splits:
                                    document_id = create_document_id(split)
                                    split.metadata["document_id"] = document_id
                                    splits_ids.append(document_id)

                            qdrant.add_documents(documents=all_splits, ids=splits_ids)
                        else:
                            page_content = f"{create_one_liner_metadata(wikipage_js, section_key)} \n {section_key} : {section_content}"
                            metadata = create_metadata(wikipage_js, section_key)
                            document = create_document(metadata, page_content)
                            all_splits = text_splitter.split_documents([document])
                            splits_ids = []
                            for split in all_splits:
                                    document_id = create_document_id(split)
                                    split.metadata["document_id"] = document_id
                                    splits_ids.append(document_id)

                            qdrant.add_documents(documents=all_splits, ids=splits_ids)


vector = qwen_embeddings.embed_query("who was knight artorias?")
results = qdrant.similarity_search_by_vector(vector, k=5)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
