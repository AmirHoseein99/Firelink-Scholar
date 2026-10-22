import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def _logger_for(model_name):
    """Return a file logger named after the pretrained model."""
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip("/"))
    logger = logging.getLogger(f"pretrained_{key}")

    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            LOG_DIR / f"pretrained_{key}.log", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


logger = _logger_for(MODEL_NAME)


def rerank(query, docs, system=None):
    """
    Rerank the passages based on their relevance to the query using a cross-encoder model.

    Args:
        query (str): The input query string.
        docs (list of tuples): A list of tuples containing the document and its corresponding score.
        system (str | None): Name of the retrieval system that requested the
            rerank (e.g. "hybrid_reranked"), recorded in the log so results from
            different systems can be compared.

    Returns:
        list of tuples: A list of tuples containing the passage and its corresponding score,
                        sorted in descending order of relevance.
    """

    passages = [doc[0].page_content for doc in docs]
    passage_to_doc = {doc[0].page_content: doc for doc in docs}

    # Prepare the input for the model
    inputs = tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Get the model's output scores
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze().tolist()

    # Combine passages with their scores and sort them
    scored_passages = list(zip(passages, scores))
    scored_passages.sort(key=lambda x: x[1], reverse=True)

    scored_docs = [
        (passage_to_doc[passage][0], score) for passage, score in scored_passages
    ]

    for rank, (document, score) in enumerate(scored_docs, start=1):
        logger.info(
            "system=%s query=%r rank=%d doc_id=%s score=%.6f",
            system or "unknown",
            query,
            rank,
            document.metadata.get("document_id"),
            score,
        )

    return scored_docs
