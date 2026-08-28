from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .interfaces import EmbeddingProvider, VectorStore


class FAISSVectorStore(VectorStore):

    def create(
        self,
        documents: list[Document],
        embedding_provider: EmbeddingProvider,
    ):
        return FAISS.from_documents(
            documents,
            embedding_provider,
        )

    def save(
        self,
        vector_store,
        path: str,
    ) -> None:
        vector_store.save_local(path)