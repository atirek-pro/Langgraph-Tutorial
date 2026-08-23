from langchain_core.documents import Document

from .interfaces import Retriever


class SemanticRetriever(Retriever):

    def __init__(
        self,
        vector_store,
        k: int,
    ):
        self.vector_store = vector_store
        self.k = k

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:

        return self.vector_store.similarity_search(
            query,
            k=self.k,
        )