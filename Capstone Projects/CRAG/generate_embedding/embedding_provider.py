from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .interfaces import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):

    def __init__(self, model: str):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=model
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return self.embedding_model.embed_documents(
            texts
        )