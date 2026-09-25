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

qwen_embeddings = OpenAIEmbeddings(
    model="qwen3-embedding-0.6b",
    base_url="http://localhost:3001/v1",
    api_key=settings.freellm_api_key,
    check_embedding_ctx_length=False,
)

COLLECTION_NAME = "darksoulswiki"
QDRANT_PATH = "/home/amirhossein/agents/Firelink-Scholar/data/qdrant"

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=qwen_embeddings,
    path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
    
)
def contextfromvectors(query):
    # try:
        # retrieval / RAG
        
    vector = qwen_embeddings.embed_query(query)
    results = qdrant.similarity_search_with_score_by_vector(vector, k=5)    
            
    page_contents = [doc[0].page_content for doc in results]
        
    # finally:
    #     qdrant.close()

    return page_contents