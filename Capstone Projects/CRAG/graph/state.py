from typing import TypedDict

from langchain_core.documents import Document


class CRAGState(TypedDict, total=False):

    question: str

    docs: list[Document]
    good_docs: list[Document]

    retrieval_scores: list[float]
    verdict: str
    reason: str

    search_query: str
    web_docs: list[Document]

    strips: list[str]
    kept_strips: list[str]
    refined_context: str

    answer: str