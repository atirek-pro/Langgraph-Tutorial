import os
import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from safetensors.torch import load_file


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./filtering_model"

WEIGHTS_FILE = os.path.join(
    MODEL_PATH,
    "model-001.safetensors"
)

# CRAG knowledge refinement threshold
FILTER_THRESHOLD = -0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CRAG KNOWLEDGE REFINEMENT - T5 STRIP CLASSIFIER")
print("=" * 80)

print(f"Device:     {device}")
print(f"Model path: {MODEL_PATH}")
print(f"Weights:    {WEIGHTS_FILE}")
print(f"Threshold:  {FILTER_THRESHOLD}")


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}"
    )

if not os.path.isfile(WEIGHTS_FILE):
    raise FileNotFoundError(
        f"Model weights not found: {WEIGHTS_FILE}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

print("Tokenizer loaded successfully.")

print(
    f"Tokenizer: {tokenizer.__class__.__name__}"
)


# ============================================================
# LOAD CONFIG
# ============================================================

print("\n" + "=" * 80)
print("LOADING CONFIG")
print("=" * 80)

config = AutoConfig.from_pretrained(
    MODEL_PATH
)

print(
    f"Model type: {config.model_type}"
)

print(
    f"Architecture: {config.architectures}"
)

print(
    f"Number of labels: {config.num_labels}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 80)
print("CREATING T5 CLASSIFIER")
print("=" * 80)

model = AutoModelForSequenceClassification.from_config(
    config
)

print(
    "T5 sequence-classification model created."
)


# ============================================================
# LOAD SAFETENSORS
# ============================================================

print("\n" + "=" * 80)
print("LOADING MODEL WEIGHTS")
print("=" * 80)

state_dict = load_file(
    WEIGHTS_FILE,
    device="cpu"
)

print(
    f"Loaded {len(state_dict)} tensors."
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

load_result = model.load_state_dict(
    state_dict,
    strict=False
)

missing_keys = load_result.missing_keys
unexpected_keys = load_result.unexpected_keys


print(
    f"\nMissing keys: {len(missing_keys)}"
)

print(
    f"Unexpected keys: {len(unexpected_keys)}"
)


# ============================================================
# HANDLE SHARED T5 EMBEDDINGS
# ============================================================

if (
    "transformer.shared.weight" in state_dict
    and
    set(missing_keys) <= {
        "transformer.encoder.embed_tokens.weight",
        "transformer.decoder.embed_tokens.weight",
    }
    and
    len(unexpected_keys) == 0
):

    print(
        "\nUsing T5 shared embedding weights."
    )

    shared_weight = state_dict[
        "transformer.shared.weight"
    ]

    model.transformer.shared.weight.data.copy_(
        shared_weight
    )

    model.transformer.encoder.embed_tokens.weight = (
        model.transformer.shared.weight
    )

    model.transformer.decoder.embed_tokens.weight = (
        model.transformer.shared.weight
    )

    print(
        "Shared embeddings configured successfully."
    )

elif missing_keys or unexpected_keys:

    print(
        "\nWARNING: Checkpoint has unexpected "
        "architecture mismatches."
    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model.to(device)
model.eval()

print(
    f"\nModel loaded on {device}."
)


# ============================================================
# T5 STRIP CLASSIFIER
# ============================================================

def classify_strip(
    question: str,
    strip: str
):
    """
    Classify one retrieved knowledge strip.

    Returns:
        score: raw T5 relevance score
        keep: whether the strip passes the CRAG
              knowledge-refinement threshold
    """

    print("\n" + "-" * 80)

    print(
        f"Question:\n{question}"
    )

    print(
        f"\nStrip:\n{strip}"
    )

    # --------------------------------------------------------
    # CRAG evaluator input
    # --------------------------------------------------------

    separator = tokenizer.sep_token

    if separator is None:
        separator = "</s>"

    model_input = (
        f"{question} {separator} {strip}"
    )

    print(
        f"\nModel input:\n{model_input}"
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # RAW RELEVANCE SCORE
    # --------------------------------------------------------

    score = outputs.logits.squeeze().item()

    # --------------------------------------------------------
    # KNOWLEDGE REFINEMENT DECISION
    # --------------------------------------------------------

    keep = score >= FILTER_THRESHOLD

    print(
        f"\nT5 relevance score: {score:.6f}"
    )

    print(
        f"Filter threshold: {FILTER_THRESHOLD}"
    )

    if keep:

        print(
            "Decision: KEEP"
        )

    else:

        print(
            "Decision: DROP"
        )

    return score, keep


# ============================================================
# TEST DATA
# ============================================================

question = (
    "What is the bias-variance tradeoff?"
)


relevant_strip = (
    "The bias-variance tradeoff describes the balance "
    "between a model's ability to fit the training data "
    "and its ability to generalize to unseen data. "
    "High bias can cause underfitting, while high "
    "variance can cause overfitting."
)


irrelevant_strip = (
    "The Transformer architecture uses self-attention "
    "mechanisms to process relationships between tokens "
    "in a sequence."
)


partially_relevant_strip = (
    "Overfitting occurs when a machine learning model "
    "learns patterns that are too specific to the "
    "training data and performs poorly on unseen data."
)


# ============================================================
# TEST 1
# ============================================================

print("\n\n")
print("#" * 80)
print("# TEST 1 - RELEVANT STRIP")
print("#" * 80)

relevant_score, relevant_keep = classify_strip(
    question,
    relevant_strip
)


# ============================================================
# TEST 2
# ============================================================

print("\n\n")
print("#" * 80)
print("# TEST 2 - IRRELEVANT STRIP")
print("#" * 80)

irrelevant_score, irrelevant_keep = classify_strip(
    question,
    irrelevant_strip
)


# ============================================================
# TEST 3
# ============================================================

print("\n\n")
print("#" * 80)
print("# TEST 3 - PARTIALLY RELEVANT STRIP")
print("#" * 80)

partial_score, partial_keep = classify_strip(
    question,
    partially_relevant_strip
)


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")

print("=" * 80)
print("KNOWLEDGE REFINEMENT SUMMARY")
print("=" * 80)

print(
    f"\nFilter threshold: {FILTER_THRESHOLD}"
)


print("\nRelevant strip:")
print(
    f"Score = {relevant_score:.6f}"
)

print(
    f"Decision = "
    f"{'KEEP' if relevant_keep else 'DROP'}"
)


print("\nIrrelevant strip:")
print(
    f"Score = {irrelevant_score:.6f}"
)

print(
    f"Decision = "
    f"{'KEEP' if irrelevant_keep else 'DROP'}"
)


print("\nPartially relevant strip:")
print(
    f"Score = {partial_score:.6f}"
)

print(
    f"Decision = "
    f"{'KEEP' if partial_keep else 'DROP'}"
)


print("\n" + "=" * 80)
print("TEST COMPLETED")
print("=" * 80)