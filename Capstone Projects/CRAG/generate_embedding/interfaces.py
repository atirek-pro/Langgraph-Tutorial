from abc import ABC, abstractmethod
from langchain_core.documents import Document


class DocumentLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> list[Document]:
        pass


class DocumentSplitter(ABC):

    @abstractmethod
    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        pass


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        pass


class VectorStore(ABC):

    @abstractmethod
    def create(
        self,
        documents: list[Document],
        embedding_provider: EmbeddingProvider,
    ):
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass