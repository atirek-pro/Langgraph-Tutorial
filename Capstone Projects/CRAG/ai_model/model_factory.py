import torch

from .interfaces import ModelLoader
from .t5_model_loader import T5ModelLoader
from .gemini_model_loader import GeminiModelLoader


class ModelFactory:

    @staticmethod
    def create_t5_loader(
        model_path: str,
        weights_file: str,
        device: torch.device,
    ) -> ModelLoader:

        return T5ModelLoader(
            model_path=model_path,
            weights_file=weights_file,
            device=device,
        )

    @staticmethod
    def create_gemini_loader(
        model_name: str,
    ) -> ModelLoader:

        return GeminiModelLoader(
            model_name=model_name,
        )