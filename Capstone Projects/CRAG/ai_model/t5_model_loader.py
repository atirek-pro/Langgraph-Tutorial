import os

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from safetensors.torch import load_file

from .interfaces import ModelLoader


class T5ModelLoader(ModelLoader):

    def __init__(
        self,
        model_path: str,
        weights_file: str,
        device: torch.device,
    ):
        self.model_path = model_path
        self.weights_file = weights_file
        self.device = device

    def load(self):

        # Validate model files
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"T5 model directory not found: "
                f"{self.model_path}"
            )

        if not os.path.isfile(self.weights_file):
            raise FileNotFoundError(
                f"T5 weights not found: "
                f"{self.weights_file}"
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
        )

        # Load configuration
        config = AutoConfig.from_pretrained(
            self.model_path
        )

        # Create model
        model = AutoModelForSequenceClassification.from_config(
            config
        )

        # Load weights
        state_dict = load_file(
            self.weights_file,
            device="cpu",
        )

        model.load_state_dict(
            state_dict,
            strict=False,
        )

        # Move model to device
        model.to(self.device)
        model.eval()

        return tokenizer, model