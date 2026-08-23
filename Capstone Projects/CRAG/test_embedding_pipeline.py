from dotenv import load_dotenv
from generate_embedding.document_loader import PDFDocumentLoader
from generate_embedding.document_splitter import (
    RecursiveDocumentSplitter,
)
from generate_embedding.embedding_provider import (
    GeminiEmbeddingProvider,
)
from generate_embedding.vector_store import FAISSVectorStore
from generate_embedding.embedding_pipeline import EmbeddingPipeline

load_dotenv()

DOCUMENT_PATHS = [
    "./documents/book1.pdf",
    "./documents/book2.pdf",
    "./documents/book3.pdf",
]

VECTOR_STORE_PATH = "./faiss_vector_store"


def main():

    print("=" * 80)
    print("TESTING EMBEDDING PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Create concrete implementations
    # ---------------------------------------------------------

    loader = PDFDocumentLoader()

    splitter = RecursiveDocumentSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    embedding_provider = GeminiEmbeddingProvider(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISSVectorStore()

    # ---------------------------------------------------------
    # 2. Inject dependencies into the pipeline
    # ---------------------------------------------------------

    pipeline = EmbeddingPipeline(
        loader=loader,
        splitter=splitter,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # ---------------------------------------------------------
    # 3. Build the vector store
    # ---------------------------------------------------------

    store = pipeline.build(
        document_paths=DOCUMENT_PATHS
    )

    print("\nEmbedding pipeline completed successfully.")

    # ---------------------------------------------------------
    # 4. Save FAISS
    # ---------------------------------------------------------

    store.save_local(
        VECTOR_STORE_PATH
    )

    print(
        f"FAISS vector store saved to: "
        f"{VECTOR_STORE_PATH}"
    )

    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()