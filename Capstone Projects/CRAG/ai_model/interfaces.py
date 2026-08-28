from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class ModelLoader(ABC, Generic[T]):

    @abstractmethod
    def load(self) -> T:
        pass

class RelevanceEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        question: str,
        document: str,
    ) -> float:
        pass