from .interfaces import (
    DocumentLoader,
    DocumentSplitter,
    EmbeddingProvider,
    VectorStore,
)


class EmbeddingPipeline:

    def __init__(
        self,
        loader: DocumentLoader,
        splitter: DocumentSplitter,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.loader = loader
        self.splitter = splitter
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def build(
        self,
        document_paths: list[str],
    ):

        # 1. Load documents
        documents = []

        for path in document_paths:
            documents.extend(
                self.loader.load(path)
            )

        # 2. Split documents
        chunks = self.splitter.split(
            documents
        )

        # 3. Create vector store
        store = self.vector_store.create(
            documents=chunks,
            embedding_provider=self.embedding_provider,
        )

        return store