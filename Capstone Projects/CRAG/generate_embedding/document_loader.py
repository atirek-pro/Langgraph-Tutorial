import os

from .interfaces import DocumentLoader

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class PDFDocumentLoader(DocumentLoader):

    def load(self, path: str) -> list[Document]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        return PyPDFLoader(path).load()