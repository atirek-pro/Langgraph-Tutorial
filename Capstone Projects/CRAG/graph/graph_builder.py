from langgraph.graph import START, END
from langgraph.graph import StateGraph

from .state import CRAGState
from .router import RetrievalRouter
from .nodes import (
    RetrievalNode,
    RetrievalEvaluationNode,
    QueryRewriteNode,
    WebRetrievalNode,
    KnowledgeRefinementNode,
    AnswerGenerationNode,
)


class CRAGGraphBuilder:

    def __init__(
        self,
        retrieval_node: RetrievalNode,
        evaluation_node: RetrievalEvaluationNode,
        query_rewrite_node: QueryRewriteNode,
        web_retrieval_node: WebRetrievalNode,
        refinement_node: KnowledgeRefinementNode,
        answer_generation_node: AnswerGenerationNode,
        router: RetrievalRouter,
    ):
        self.graph = StateGraph(CRAGState)

        self.retrieval_node = retrieval_node
        self.evaluation_node = evaluation_node
        self.query_rewrite_node = query_rewrite_node
        self.web_retrieval_node = web_retrieval_node
        self.refinement_node = refinement_node
        self.answer_generation_node = answer_generation_node
        self.router = router

    def add_nodes(self):

        self.graph.add_node(
            "retrieve",
            self.retrieval_node,
        )

        self.graph.add_node(
            "evaluate_retrieval",
            self.evaluation_node,
        )

        self.graph.add_node(
            "rewrite_query",
            self.query_rewrite_node,
        )

        self.graph.add_node(
            "web_search",
            self.web_retrieval_node,
        )

        self.graph.add_node(
            "refine",
            self.refinement_node,
        )

        self.graph.add_node(
            "generate",
            self.answer_generation_node,
        )

    def add_edges(self):

        # ------------------------------------------------------
        # INITIAL FLOW
        # ------------------------------------------------------

        self.graph.add_edge(
            START,
            "retrieve",
        )

        self.graph.add_edge(
            "retrieve",
            "evaluate_retrieval",
        )

        # ------------------------------------------------------
        # CORRECT PATH
        # ------------------------------------------------------

        self.graph.add_edge(
            "refine",
            "generate",
        )

        # ------------------------------------------------------
        # INCORRECT PATH
        # ------------------------------------------------------

        self.graph.add_edge(
            "rewrite_query",
            "web_search",
        )

        self.graph.add_edge(
            "web_search",
            "refine",
        )


    def add_conditional_edges(self):

        self.graph.add_conditional_edges(
            "evaluate_retrieval",
            self.router.route,
            {
                "refine": "refine",
                "rewrite_query": "rewrite_query",
            },
        )

    def compile(self):

        return self.graph.compile()
