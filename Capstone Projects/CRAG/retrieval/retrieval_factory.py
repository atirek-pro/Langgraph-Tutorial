from .interfaces import Retriever
from .semantic_retriever import SemanticRetriever
from .web_retriever import WebRetriever


class RetrievalFactory:

    @staticmethod
    def create_semantic_retriever(
        vector_store,
        k: int,
    ) -> Retriever:

        return SemanticRetriever(
            vector_store=vector_store,
            k=k,
        )

    @staticmethod
    def create_web_retriever(
        max_results: int,
    ) -> Retriever:

        return WebRetriever(
            max_results=max_results,
        )