from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        pass