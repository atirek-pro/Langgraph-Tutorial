import torch

from application import CRAGApplication
from dotenv import load_dotenv

load_dotenv()

def main():

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------

    t5_model_path = "path/to/t5/model"
    t5_weights_file = "path/to/model.safetensors"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # These must be the values from your original code.
    retrieval_lower_threshold = 0.59
    retrieval_upper_threshold = -0.99

    # Your existing vector store
    vector_store = "./faiss_vector_store"

    # ------------------------------------------------------
    # Create application
    # ------------------------------------------------------

    app = CRAGApplication(
        t5_model_path=t5_model_path,
        t5_weights_file=t5_weights_file,
        device=device,
        vector_store=vector_store,
        retrieval_lower_threshold=retrieval_lower_threshold,
        retrieval_upper_threshold=retrieval_upper_threshold,
    )

    # ------------------------------------------------------
    # Generate graph visualization
    # ------------------------------------------------------

    output_path = app.save_flow_png(
        "crag_flow.png"
    )

    print(
        f"CRAG graph saved to: {output_path}"
    )


if __name__ == "__main__":
    main()