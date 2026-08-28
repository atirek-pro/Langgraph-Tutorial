from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from .interfaces import Retriever


class WebRetriever(Retriever):

    def __init__(self, max_results: int):

        self.search_tool = TavilySearchResults(
            max_results=max_results,
        )

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:

        results = self.search_tool.invoke(
            {"query": query}
        )

        documents = []

        for result in results:

            if not isinstance(result, dict):
                continue

            content = result.get(
                "content",
                "",
            ).strip()

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "retrieval_source": "tavily",
                        "tavily_score": result.get("score"),
                    },
                )
            )

        return documents