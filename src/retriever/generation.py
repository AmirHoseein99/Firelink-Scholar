from src.config import settings
from src.llm.client import OpenAICompatibleAPI
from src.retriever.enhancement import enhance_query
from src.retriever.querying import contextfromvectors

model = OpenAICompatibleAPI(base_url="http://localhost:3001/v1", api_key=settings.freellm_api_key, model=settings.freellm_model, provider="FreeLlmApi")

user_query = input("write your question about darksouls universe : \n ")
enhanced_query = enhance_query(user_query)
ranked_documents = contextfromvectors(query=enhanced_query, k=20)

messages = [
    {"role": "system", "content": "\n\n".join(retrieved_documents)},
    # {"role": "system", "content": ""},
    {"role": "user", "content": user_query},
]

response = model.call(messages)
answer = response["choices"][0]["message"]["content"]

print(50 * '#')
print(answer)
