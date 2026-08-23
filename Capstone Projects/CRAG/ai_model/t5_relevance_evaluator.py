import torch

from .interfaces import ModelLoader, RelevanceEvaluator


class T5RelevanceEvaluator(RelevanceEvaluator):

    def __init__(
        self,
        model_loader: ModelLoader,
        max_length: int,
    ):
        self.model_loader = model_loader
        self.max_length = max_length

        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self):

        if self.model is None:
            self.tokenizer, self.model = (
                self.model_loader.load()
            )

    def evaluate(
        self,
        question: str,
        document: str,
    ) -> float:

        self._ensure_model_loaded()

        separator = (
            self.tokenizer.sep_token
            or "</s>"
        )

        model_input = (
            f"{question} {separator} {document}"
        )

        inputs = self.tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        return outputs.logits.squeeze().item()