from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .interfaces import DocumentSplitter


class RecursiveDocumentSplitter(DocumentSplitter):

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:

        chunks = self.splitter.split_documents(
            documents
        )

        for document in chunks:
            document.page_content = (
                document.page_content
                .encode("utf-8", "ignore")
                .decode("utf-8", "ignore")
            )

        return chunks