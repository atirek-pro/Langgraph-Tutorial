import re

from .state import CRAGState

from prompts.crag_prompts import query_rewrite_prompt, answer_prompt
from retrieval.interfaces import Retriever
from ai_model.interfaces import ModelLoader, RelevanceEvaluator


class RetrievalNode:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def __call__(self, state: CRAGState) -> CRAGState:

        documents = self.retriever.retrieve(
            state["question"]
        )

        return {
            "docs": documents
        }


class RetrievalEvaluationNode:

    def __init__(
        self,
        evaluator: RelevanceEvaluator,
        lower_threshold: float,
        upper_threshold: float,
    ):
        self.evaluator = evaluator
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def __call__(
        self,
        state: CRAGState,
    ) -> CRAGState:

        question = state["question"]
        retrieved_docs = state.get("docs", [])

        scores: list[float] = []
        good_docs = []

        # --------------------------------------------------------
        # Evaluate every retrieved document
        # --------------------------------------------------------

        for document in retrieved_docs:

            score = self.evaluator.evaluate(
                question=question,
                document=document.page_content,
            )

            scores.append(score)

            # Documents above the lower threshold
            # can contribute to the internal knowledge path.
            if score > self.lower_threshold:
                good_docs.append(document)

        # --------------------------------------------------------
        # No documents retrieved
        # --------------------------------------------------------

        if not scores:

            return {
                "retrieval_scores": [],
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": "No documents were retrieved.",
            }

        # --------------------------------------------------------
        # CORRECT
        # --------------------------------------------------------

        if any(
            score > self.upper_threshold
            for score in scores
        ):

            reason = (
                "At least one retrieved document "
                "scored above the upper threshold "
                f"{self.upper_threshold}."
            )

            verdict = "CORRECT"

        # --------------------------------------------------------
        # INCORRECT
        # --------------------------------------------------------

        elif all(
            score < self.lower_threshold
            for score in scores
        ):

            reason = (
                "All retrieved documents scored "
                "below the lower threshold "
                f"{self.lower_threshold}."
            )

            verdict = "INCORRECT"

            # Discard internal documents
            good_docs = []

        # --------------------------------------------------------
        # AMBIGUOUS
        # --------------------------------------------------------

        else:

            reason = (
                "Retrieved documents produced mixed "
                "or uncertain relevance scores."
            )

            verdict = "AMBIGUOUS"

        return {
            "retrieval_scores": scores,
            "good_docs": good_docs,
            "verdict": verdict,
            "reason": reason,
        }


class QueryRewriteNode:

    def __init__(
        self,
        model_loader: ModelLoader,
    ):
        self.model_loader = model_loader
        self.model = None

    def _ensure_model_loaded(self):

        if self.model is None:
            self.model = self.model_loader.load()

    def __call__(
        self,
        state: CRAGState,
    ) -> CRAGState:

        self._ensure_model_loaded()

        question = state["question"]

        output = (
            query_rewrite_prompt | self.model
        ).invoke(
            {
                "question": question,
            }
        )

        rewritten_query = (
            output.content.strip()
        )

        # Preserve original fallback behavior
        if not rewritten_query:
            rewritten_query = question

        return {
            "search_query": rewritten_query
        }

class WebRetrievalNode:

    def __init__(
        self,
        retriever,
    ):
        self.retriever = retriever

    def __call__(
        self,
        state: CRAGState,
    ) -> CRAGState:

        search_query = state["search_query"]

        # Retrieve documents from Tavily
        web_docs = self.retriever.retrieve(
            search_query
        )

        # Preserve documents that were already
        # considered relevant by internal retrieval.
        existing_good_docs = state.get(
            "good_docs",
            [],
        )

        # Combine internal good documents with
        # externally retrieved documents.
        combined_docs = (
            existing_good_docs + web_docs
        )

        return {
            "web_docs": web_docs,
            "good_docs": combined_docs,
        }

class KnowledgeRefinementNode:

    def __init__(
        self,
        evaluator: RelevanceEvaluator,
        threshold: float,
    ):
        self.evaluator = evaluator
        self.threshold = threshold

    def __call__(
        self,
        state: CRAGState,
    ) -> CRAGState:

        question = state["question"]

        # IMPORTANT:
        # Original code uses good_docs here,
        # not docs.
        refinement_docs = state["good_docs"]

        if not refinement_docs:
            return {
                "strips": [],
                "kept_strips": [],
                "refined_context": "",
            }

        # --------------------------------------------------------
        # COMBINE DOCUMENTS
        # --------------------------------------------------------

        context = "\n\n".join(
            document.page_content
            for document in refinement_docs
        ).strip()

        # --------------------------------------------------------
        # DECOMPOSE
        # --------------------------------------------------------

        strips = self._decompose_to_sentences(
            context
        )

        # --------------------------------------------------------
        # FILTER
        # --------------------------------------------------------

        kept_strips = []

        for strip in strips:

            score = self.evaluator.evaluate(
                question,
                strip,
            )

            if score >= self.threshold:
                kept_strips.append(strip)

        # --------------------------------------------------------
        # RECOMPOSE
        # --------------------------------------------------------

        refined_context = "\n".join(
            kept_strips
        ).strip()

        return {
            "strips": strips,
            "kept_strips": kept_strips,
            "refined_context": refined_context,
        }

    @staticmethod
    def _decompose_to_sentences(
        text: str,
    ) -> list[str]:

        # EXACT logic from original implementation

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) > 20
        ]

class AnswerGenerationNode:

    def __init__(
        self,
        model_loader: ModelLoader,
    ):
        self.model_loader = model_loader
        self.model = None

    def _ensure_model_loaded(self):

        if self.model is None:
            self.model = self.model_loader.load()

    def __call__(
        self,
        state: CRAGState,
    ) -> CRAGState:

        self._ensure_model_loaded()

        output = (
            answer_prompt | self.model
        ).invoke(
            {
                "question": state["question"],
                "refined_context": state[
                    "refined_context"
                ],
            }
        )

        return {
            "answer": output.content
        }