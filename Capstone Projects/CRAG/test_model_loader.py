import torch
from dotenv import load_dotenv

from ai_model.model_factory import ModelFactory


load_dotenv()


T5_MODEL_PATH = "./filtering_model"
T5_WEIGHTS_FILE = "./filtering_model/model-001.safetensors"
GEMINI_MODEL_NAME = "gemini-2.5-flash"


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("TESTING MODEL LOADERS")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Create T5 loader through factory
    # ---------------------------------------------------------

    print("\nLoading T5 model...")

    t5_loader = ModelFactory.create_t5_loader(
        model_path=T5_MODEL_PATH,
        weights_file=T5_WEIGHTS_FILE,
        device=device,
    )

    tokenizer, model = t5_loader.load()

    print("T5 model loaded successfully.")
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Model: {model.__class__.__name__}")

    # ---------------------------------------------------------
    # 2. Create Gemini loader through factory
    # ---------------------------------------------------------

    print("\nInitializing Gemini model...")

    gemini_loader = ModelFactory.create_gemini_loader(
        model_name=GEMINI_MODEL_NAME,
    )

    gemini_model = gemini_loader.load()

    print("Gemini model initialized successfully.")
    print(f"Model: {gemini_model.__class__.__name__}")

    # ---------------------------------------------------------
    # 3. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL MODEL LOADER TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()