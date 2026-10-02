from src.config import settings
from src.llm.client import OpenAICompatibleAPI
from src.retriever.querying import contextfromvectors

model = OpenAICompatibleAPI(base_url="http://localhost:3001/v1", api_key=settings.freellm_api_key, model=settings.freellm_model, provider="FreeLlmApi")

user_query = input("write your question about darksouls universe : \n ")
retrieved_documents = [
    doc.page_content for doc, _ in contextfromvectors(query=user_query)
]

messages = [
    {"role": "system", "content": "\n\n".join(retrieved_documents)},
    # {"role": "system", "content": ""},
    {"role": "user", "content": user_query},
]

print(messages)

response = model.call(messages) 

print(50 * '#')
print(response['choices'][0]['message']['content'])