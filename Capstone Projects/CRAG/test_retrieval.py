from dotenv import load_dotenv

from retrieval.retrieval_factory import RetrievalFactory
from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings,
    )
from langchain_community.vectorstores import FAISS

load_dotenv()


VECTOR_STORE_PATH = "./faiss_vector_store"

QUERY = "Explain the bias-variance tradeoff"


def main():

    print("=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load existing FAISS vector store
    # ---------------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # ---------------------------------------------------------
    # 2. Create semantic retriever through factory
    # ---------------------------------------------------------

    semantic_retriever = (
        RetrievalFactory.create_semantic_retriever(
            vector_store=vector_store,
            k=4,
        )
    )

    # ---------------------------------------------------------
    # 3. Test semantic retrieval
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL")
    print("=" * 80)

    semantic_docs = semantic_retriever.retrieve(
        QUERY
    )

    print(
        f"\nRetrieved {len(semantic_docs)} documents."
    )

    for index, document in enumerate(
        semantic_docs,
        start=1,
    ):
        print(f"\n--- Document {index} ---")
        print(
            document.page_content[:500]
        )

    # ---------------------------------------------------------
    # 4. Create web retriever through factory
    # ---------------------------------------------------------

    web_retriever = (
        RetrievalFactory.create_web_retriever(
            max_results=5,
        )
    )

    # ---------------------------------------------------------
    # 5. Test web retrieval
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("WEB RETRIEVAL")
    print("=" * 80)

    web_docs = web_retriever.retrieve(
        QUERY
    )

    print(
        f"\nRetrieved {len(web_docs)} web documents."
    )

    for index, document in enumerate(
        web_docs,
        start=1,
    ):
        print(f"\n--- Web Document {index} ---")

        print(
            f"Title: "
            f"{document.metadata.get('title', '')}"
        )

        print(
            f"URL: "
            f"{document.metadata.get('url', '')}"
        )

        print(
            document.page_content[:500]
        )

    # ---------------------------------------------------------
    # 6. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()