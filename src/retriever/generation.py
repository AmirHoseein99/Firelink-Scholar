import logging
import time
from datetime import datetime, timezone

from src.config import settings
from src.llm.client import OpenAICompatibleAPI
from src.retriever.enhancement import enhance_query
from src.retriever.querying import contextfromvectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("Generation")

model = OpenAICompatibleAPI(base_url="http://localhost:3001/v1", api_key=settings.freellm_api_key, model=settings.freellm_model, provider="FreeLlmApi")

user_query = input("write your question about darksouls universe : \n ")

run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
logger.info(f"[{run_id}] user query: {user_query!r}")

start = time.perf_counter()
enhanced_query = enhance_query(user_query)
logger.info(f"[{run_id}] enhanced query: {enhanced_query!r} ({time.perf_counter() - start:.2f}s)")

start = time.perf_counter()
ranked_documents = contextfromvectors(query=enhanced_query, k=20)
logger.info(
    f"[{run_id}] retrieved {len(ranked_documents)} documents in {time.perf_counter() - start:.2f}s; "
    f"scores: {[f'{score:.4f}' for _, score in ranked_documents]}"
)
for i, (doc, score) in enumerate(ranked_documents, start=1):
    document_id = doc.metadata.get("document_id", "unknown")
    snippet = " ".join(doc.page_content.split())[:80]
    logger.info(f"[{run_id}] doc {i}: id={document_id} score={score:.4f} snippet={snippet!r}")

retrieved_documents = [doc.page_content for doc, _ in ranked_documents]

messages = [
    {"role": "system", "content": "\n\n".join(retrieved_documents)},
    {"role": "user", "content": user_query},
]

start = time.perf_counter()
response = model.call(messages)
answer = response["choices"][0]["message"]["content"]
logger.info(f"[{run_id}] generated answer in {time.perf_counter() - start:.2f}s ({len(answer)} chars)")

print(50 * '#')
print(answer)
