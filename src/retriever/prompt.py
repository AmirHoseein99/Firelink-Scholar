from src.retriever.querying import contextfromvectors

INSTRUCTIONS = """\
You answer questions about the Dark Souls universe using ONLY the context
documents provided below. Never use your own knowledge; if the documents do
not contain the answer, say that the context does not cover it.

Every answer must cite the documents it is based on, using their [n] labels,
e.g. [1] or [2][5]. Do not cite a document you did not use."""


def build_prompt(query: str, k: int = 5) -> list[dict]:
    """Build the generation prompt: instructions + scored context, query as user."""
    retrieved = contextfromvectors(query=query, k=k)

    context_blocks = [
        f"[{i}] score={score:.3f} source={doc.metadata.get('title')} ({doc.metadata.get('section')})\n{doc.page_content}"
        for i, (doc, score) in enumerate(retrieved, start=1)
    ]

    context = "\n\n".join(context_blocks)

    return [
        {"role": "system", "content": f"{INSTRUCTIONS}\n\n{context}"},
        {"role": "user", "content": query},
    ]
