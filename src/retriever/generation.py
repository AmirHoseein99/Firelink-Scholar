from src.config import settings
from src.llm.client import OpenAICompatibleAPI
from src.retriever.prompt import build_prompt

model = OpenAICompatibleAPI(base_url=settings.NEURALWATT_BASE_URL, api_key=settings.NEURALWATT_API_KEY, model=settings.NEURALWATT_MODEL, provider="NeuralWatt")


def generate(query: str, k: int = 5) -> str:
    """Answer a question with RAG: retrieve k chunks, then query the LLM."""
    response = model.call(build_prompt(query, k=k))

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    user_query = input("write your question about darksouls universe : \n ")
    answer = generate(user_query, k=20)
    print(50 * '#')
    print(answer)
