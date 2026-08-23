from ai_model.t5_model_loader import T5ModelLoader
from ai_model.gemini_model_loader import GeminiModelLoader
from ai_model.t5_relevance_evaluator import T5RelevanceEvaluator

from retrieval.retrieval_factory import RetrievalFactory

from graph.nodes import (
    RetrievalNode,
    RetrievalEvaluationNode,
    QueryRewriteNode,
    WebRetrievalNode,
    KnowledgeRefinementNode,
    AnswerGenerationNode,
)

from graph.router import RetrievalRouter
from graph.graph_builder import CRAGGraphBuilder


class CRAGApplication:

    def __init__(
        self,
        t5_model_path: str,
        t5_weights_file: str,
        device,
        vector_store,
        retrieval_lower_threshold: float,
        retrieval_upper_threshold: float,
        strip_threshold: float = -0.5,
    ):

        # ======================================================
        # 1. MODEL LOADERS
        # ======================================================

        self.t5_loader = T5ModelLoader(
            model_path=t5_model_path,
            weights_file=t5_weights_file,
            device=device,
        )

        self.gemini_loader = GeminiModelLoader(
            model_name="gemini-2.5-flash"
        )

        # ======================================================
        # 2. RETRIEVERS
        # ======================================================

        self.semantic_retriever = (
            RetrievalFactory.create_semantic_retriever(
                vector_store=vector_store,
                k=4,
            )
        )

        self.web_retriever = (
            RetrievalFactory.create_web_retriever(
                max_results=5,
            )
        )

        # ======================================================
        # 3. RELEVANCE EVALUATOR
        # ======================================================

        self.t5_evaluator = T5RelevanceEvaluator(
            model_loader=self.t5_loader,
            max_length=512,
        )

        # ======================================================
        # 4. GRAPH NODES
        # ======================================================

        self.retrieval_node = RetrievalNode(
            retriever=self.semantic_retriever,
        )

        self.evaluation_node = RetrievalEvaluationNode(
            evaluator=self.t5_evaluator,
            lower_threshold=retrieval_lower_threshold,
            upper_threshold=retrieval_upper_threshold,
        )

        self.query_rewrite_node = QueryRewriteNode(
            model_loader=self.gemini_loader,
        )

        self.web_retrieval_node = WebRetrievalNode(
            retriever=self.web_retriever,
        )

        self.refinement_node = KnowledgeRefinementNode(
            evaluator=self.t5_evaluator,
            threshold=strip_threshold,
        )

        self.answer_generation_node = AnswerGenerationNode(
            model_loader=self.gemini_loader,
        )

        # ======================================================
        # 5. ROUTER
        # ======================================================

        self.router = RetrievalRouter()

        # ======================================================
        # 6. GRAPH BUILDER
        # ======================================================

        self.graph_builder = CRAGGraphBuilder(
            retrieval_node=self.retrieval_node,
            evaluation_node=self.evaluation_node,
            query_rewrite_node=self.query_rewrite_node,
            web_retrieval_node=self.web_retrieval_node,
            refinement_node=self.refinement_node,
            answer_generation_node=self.answer_generation_node,
            router=self.router,
        )

        # ======================================================
        # 7. BUILD GRAPH
        # ======================================================

        self.graph_builder.add_nodes()
        self.graph_builder.add_edges()
        self.graph_builder.add_conditional_edges()

        self.graph = self.graph_builder.compile()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def run(
        self,
        question: str,
    ):

        initial_state = {
            "question": question,
        }

        return self.graph.invoke(
            initial_state
        )

    def save_flow_png(
        self,
        output_path: str = "crag_flow.png",
    ):
        """
        Save the compiled CRAG graph as a PNG image.
        """

        png_bytes = self.graph.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as file:
            file.write(png_bytes)

        return output_path