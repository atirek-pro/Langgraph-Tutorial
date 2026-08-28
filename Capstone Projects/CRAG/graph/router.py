from typing import Literal

from .state import CRAGState


class RetrievalRouter:

    def route(
        self,
        state: CRAGState,
    ) -> Literal[
        "refine",
        "rewrite_query",
    ]:

        verdict = state["verdict"]

        if verdict == "CORRECT":
            return "refine"

        # INCORRECT and AMBIGUOUS
        # both use the external knowledge path.
        return "rewrite_query"