from typing import List, TypedDict, Literal
import os
import re

import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from safetensors.torch import load_file

from langchain_community.tools.tavily_search import (
    TavilySearchResults,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOCUMENT_PATHS = [
    "./documents/book1.pdf",
    "./documents/book2.pdf",
    "./documents/book3.pdf",
]


# ------------------------------------------------------------
# T5 MODEL
# ------------------------------------------------------------

FILTERING_MODEL_PATH = "./filtering_model"

FILTERING_WEIGHTS_FILE = os.path.join(
    FILTERING_MODEL_PATH,
    "model-001.safetensors",
)


# ------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------

RETRIEVAL_K = 4


# ------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ------------------------------------------------------------
# T5 INPUT
# ------------------------------------------------------------

T5_MAX_LENGTH = 512


# ------------------------------------------------------------
# CRAG RETRIEVAL EVALUATION THRESHOLDS
# ------------------------------------------------------------

RETRIEVAL_UPPER_THRESHOLD = 0.59
RETRIEVAL_LOWER_THRESHOLD = -0.99


# ------------------------------------------------------------
# KNOWLEDGE REFINEMENT THRESHOLD
# ------------------------------------------------------------

STRIP_FILTER_THRESHOLD = -0.5


# ------------------------------------------------------------
# TAVILY
# ------------------------------------------------------------

TAVILY_MAX_RESULTS = 5


# ------------------------------------------------------------
# FAISS
# ------------------------------------------------------------

FAISS_INDEX_PATH = "./faiss_vector_store"

FAISS_INDEX_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "index.faiss",
)

FAISS_METADATA_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "index.pkl",
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. STARTUP
# ============================================================

print()
print("=" * 100)
print("CORRECTIVE RAG")
print("=" * 100)

print(f"Device: {device}")

print(
    f"T5 filtering model: "
    f"{FILTERING_MODEL_PATH}"
)

print(
    f"Retrieval upper threshold: "
    f"{RETRIEVAL_UPPER_THRESHOLD}"
)

print(
    f"Retrieval lower threshold: "
    f"{RETRIEVAL_LOWER_THRESHOLD}"
)

print(
    f"Knowledge refinement threshold: "
    f"{STRIP_FILTER_THRESHOLD}"
)

print(
    f"Tavily max results: "
    f"{TAVILY_MAX_RESULTS}"
)


# ============================================================
# 4. LOAD T5 TOKENIZER
# ============================================================

print()
print("=" * 100)
print("4. LOADING T5 TOKENIZER")
print("=" * 100)


if not os.path.isdir(FILTERING_MODEL_PATH):

    raise FileNotFoundError(
        f"T5 model directory not found: "
        f"{FILTERING_MODEL_PATH}"
    )


if not os.path.isfile(FILTERING_WEIGHTS_FILE):

    raise FileNotFoundError(
        f"T5 weights not found: "
        f"{FILTERING_WEIGHTS_FILE}"
    )


filter_tokenizer = AutoTokenizer.from_pretrained(
    FILTERING_MODEL_PATH,
    use_fast=False,
)


print(
    "Tokenizer loaded successfully."
)

print(
    f"Tokenizer: "
    f"{filter_tokenizer.__class__.__name__}"
)


# ============================================================
# 5. LOAD T5 CONFIGURATION
# ============================================================

print()
print("=" * 100)
print("5. LOADING T5 CONFIGURATION")
print("=" * 100)


filter_config = AutoConfig.from_pretrained(
    FILTERING_MODEL_PATH
)


print(
    f"Model type: "
    f"{filter_config.model_type}"
)

print(
    f"Architecture: "
    f"{filter_config.architectures}"
)

print(
    f"Number of labels: "
    f"{filter_config.num_labels}"
)


if filter_config.num_labels != 1:

    raise ValueError(
        "Expected the CRAG T5 evaluator to have "
        "exactly one output logit."
    )


# ============================================================
# 6. CREATE T5 CLASSIFICATION MODEL
# ============================================================

print()
print("=" * 100)
print("6. CREATING T5 CLASSIFICATION MODEL")
print("=" * 100)


filtering_model = (
    AutoModelForSequenceClassification.from_config(
        filter_config
    )
)


print(
    "T5 sequence-classification model created."
)


# ============================================================
# 7. LOAD T5 SAFETENSORS
# ============================================================

print()
print("=" * 100)
print("7. LOADING T5 MODEL WEIGHTS")
print("=" * 100)


print(
    f"Loading: "
    f"{FILTERING_WEIGHTS_FILE}"
)


state_dict = load_file(
    FILTERING_WEIGHTS_FILE,
    device="cpu",
)


print(
    f"Loaded {len(state_dict)} tensors."
)


# ============================================================
# 8. APPLY T5 MODEL WEIGHTS
# ============================================================

print()
print("=" * 100)
print("8. APPLYING T5 MODEL WEIGHTS")
print("=" * 100)


load_result = filtering_model.load_state_dict(
    state_dict,
    strict=False,
)


missing_keys = load_result.missing_keys
unexpected_keys = load_result.unexpected_keys


print(
    f"Missing keys: "
    f"{len(missing_keys)}"
)

print(
    f"Unexpected keys: "
    f"{len(unexpected_keys)}"
)


# ============================================================
# 9. HANDLE T5 SHARED EMBEDDINGS
# ============================================================

expected_shared_embedding_keys = {
    "transformer.encoder.embed_tokens.weight",
    "transformer.decoder.embed_tokens.weight",
}


if (
    "transformer.shared.weight" in state_dict
    and
    set(missing_keys).issubset(
        expected_shared_embedding_keys
    )
    and
    len(unexpected_keys) == 0
):

    print()
    print(
        "Configuring T5 shared embeddings..."
    )


    shared_weight = state_dict[
        "transformer.shared.weight"
    ]


    filtering_model.transformer.shared.weight.data.copy_(
        shared_weight
    )


    filtering_model.transformer.encoder.embed_tokens.weight = (
        filtering_model.transformer.shared.weight
    )


    filtering_model.transformer.decoder.embed_tokens.weight = (
        filtering_model.transformer.shared.weight
    )


    print(
        "T5 shared embeddings configured successfully."
    )


elif missing_keys or unexpected_keys:

    print()
    print(
        "WARNING: Unexpected checkpoint structure."
    )

    print(
        f"Missing keys: {missing_keys}"
    )

    print(
        f"Unexpected keys: {unexpected_keys}"
    )

    raise RuntimeError(
        "T5 checkpoint does not match the "
        "expected CRAG T5 architecture."
    )


# ============================================================
# 10. CLEAN CPU STATE DICT
# ============================================================

del state_dict


# ============================================================
# 11. MOVE T5 MODEL TO DEVICE
# ============================================================

print()
print("=" * 100)
print("11. MOVING T5 MODEL TO DEVICE")
print("=" * 100)


filtering_model.to(device)
filtering_model.eval()


print(
    f"T5 model ready on {device}."
)


# ============================================================
# 12. LOAD OR CREATE FAISS VECTOR STORE
# ============================================================

print()
print("=" * 100)
print("12. LOAD OR CREATE FAISS VECTOR STORE")
print("=" * 100)


faiss_exists = (
    os.path.isdir(FAISS_INDEX_PATH)
    and os.path.isfile(FAISS_INDEX_FILE)
    and os.path.isfile(FAISS_METADATA_FILE)
)


# ============================================================
# CASE 1 — EXISTING FAISS
# ============================================================

if faiss_exists:

    print()
    print(
        "Existing FAISS vector store found."
    )

    print(
        f"FAISS path: {FAISS_INDEX_PATH}"
    )


    print()
    print(
        "Initializing Gemini embeddings..."
    )


    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


    print(
        "Gemini embeddings initialized."
    )


    print()
    print(
        "Loading FAISS vector store..."
    )


    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


    print(
        "FAISS vector store loaded successfully."
    )


# ============================================================
# CASE 2 — CREATE FAISS
# ============================================================

else:

    print()
    print(
        "FAISS vector store not found."
    )

    print(
        "Building vector store from PDF documents..."
    )


    # --------------------------------------------------------
    # 12.1 LOAD PDF DOCUMENTS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("12.1 LOADING PDF DOCUMENTS")
    print("-" * 100)


    docs: List[Document] = []


    for document_path in DOCUMENT_PATHS:

        print(
            f"Loading: {document_path}"
        )


        if not os.path.isfile(document_path):

            raise FileNotFoundError(
                f"PDF not found: "
                f"{document_path}"
            )


        loaded_docs = PyPDFLoader(
            document_path
        ).load()


        docs.extend(
            loaded_docs
        )


    print()
    print(
        f"Loaded {len(docs)} PDF pages."
    )


    # --------------------------------------------------------
    # 12.2 CHUNK DOCUMENTS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("12.2 CHUNKING DOCUMENTS")
    print("-" * 100)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


    chunks = text_splitter.split_documents(
        docs
    )


    for document in chunks:

        document.page_content = (
            document.page_content
            .encode(
                "utf-8",
                "ignore",
            )
            .decode(
                "utf-8",
                "ignore",
            )
        )


    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # 12.3 INITIALIZE GEMINI EMBEDDINGS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("12.3 INITIALIZING GEMINI EMBEDDINGS")
    print("-" * 100)


    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


    print(
        "Gemini embeddings initialized."
    )


    # --------------------------------------------------------
    # 12.4 CREATE FAISS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("12.4 CREATING FAISS VECTOR STORE")
    print("-" * 100)


    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )


    print(
        "FAISS vector store created successfully."
    )


    # --------------------------------------------------------
    # 12.5 SAVE FAISS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("12.5 SAVING FAISS VECTOR STORE")
    print("-" * 100)


    os.makedirs(
        FAISS_INDEX_PATH,
        exist_ok=True,
    )


    vector_store.save_local(
        FAISS_INDEX_PATH
    )


    print(
        f"FAISS vector store saved to:"
        f"\n{FAISS_INDEX_PATH}"
    )


# ============================================================
# 13. CREATE RETRIEVER
# ============================================================

print()
print("=" * 100)
print("13. CREATE RETRIEVER")
print("=" * 100)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": RETRIEVAL_K,
    },
)


print(
    "Retriever created successfully."
)

print(
    f"Retrieval K: {RETRIEVAL_K}"
)


# ============================================================
# 14. INITIALIZE GEMINI LLM
# ============================================================

print()
print("=" * 100)
print("14. INITIALIZING GEMINI LLM")
print("=" * 100)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


print(
    "Gemini 2.5 Flash initialized."
)


# ============================================================
# 15. QUERY REWRITE PROMPT
# ============================================================

query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a search query optimization assistant.

Rewrite the user's question into a concise,
fact-focused search query that will work well
with a web search engine.

Rules:
- Preserve the original intent.
- Do not answer the question.
- Do not add facts that are not present in the question.
- Remove unnecessary conversational wording.
- Keep important entities, names, dates, and technical terms.
- Return ONLY the rewritten search query.
""",
        ),
        (
            "human",
            "User question:\n{question}",
        ),
    ]
)


# ============================================================
# 16. LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    question: str

    # --------------------------------------------------------
    # Internal retrieval
    # --------------------------------------------------------

    docs: List[Document]

    # Documents entering knowledge refinement.
    #
    # CORRECT:
    #     internal retrieved documents
    #
    # INCORRECT:
    #     Tavily web documents
    #
    good_docs: List[Document]

    # --------------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------------

    retrieval_scores: List[float]

    verdict: str

    reason: str

    # --------------------------------------------------------
    # Query rewriting
    # --------------------------------------------------------

    search_query: str

    # --------------------------------------------------------
    # Web search
    # --------------------------------------------------------

    web_docs: List[Document]

    # --------------------------------------------------------
    # Knowledge refinement
    # --------------------------------------------------------

    strips: List[str]

    kept_strips: List[str]

    refined_context: str

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    answer: str


# ============================================================
# 17. RETRIEVAL
# ============================================================

def retrieve(
    state: State,
) -> State:

    question = state["question"]


    print()
    print("=" * 100)
    print("RETRIEVAL")
    print("=" * 100)


    print(
        f"Question: {question}"
    )


    retrieved_docs = retriever.invoke(
        question
    )


    print(
        f"Retrieved documents: "
        f"{len(retrieved_docs)}"
    )


    return {
        "docs": retrieved_docs,
    }


# ============================================================
# 18. T5 DOCUMENT EVALUATOR
# ============================================================

def evaluate_document(
    question: str,
    document: Document,
):

    """
    Evaluate one question-document pair using
    the fine-tuned CRAG T5 evaluator.
    """

    separator = filter_tokenizer.sep_token


    if separator is None:

        separator = "</s>"


    model_input = (
        f"{question} "
        f"{separator} "
        f"{document.page_content}"
    )


    inputs = filter_tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=T5_MAX_LENGTH,
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = filtering_model(
            **inputs
        )


    score = (
        outputs.logits
        .squeeze()
        .item()
    )


    return score


# ============================================================
# 19. RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    state: State,
) -> State:

    question = state["question"]

    retrieved_docs = state["docs"]


    print()
    print("=" * 100)
    print("T5 RETRIEVAL EVALUATION")
    print("=" * 100)


    scores: List[float] = []

    good_docs: List[Document] = []


    # --------------------------------------------------------
    # Evaluate every retrieved document
    # --------------------------------------------------------

    for index, document in enumerate(
        retrieved_docs,
        start=1,
    ):

        score = evaluate_document(
            question=question,
            document=document,
        )


        scores.append(
            score
        )


        print()
        print("-" * 100)

        print(
            f"DOCUMENT {index}/{len(retrieved_docs)}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source', 'unknown')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page', 'unknown')}"
        )

        print(
            f"T5 relevance score: "
            f"{score:.6f}"
        )


        # ----------------------------------------------------
        # Documents above lower threshold can contribute
        # to the internal knowledge path.
        # ----------------------------------------------------

        if score > RETRIEVAL_LOWER_THRESHOLD:

            good_docs.append(
                document
            )


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


    # ========================================================
    # CRAG ROUTING
    # ========================================================

    # --------------------------------------------------------
    # CORRECT
    # --------------------------------------------------------

    if any(
        score > RETRIEVAL_UPPER_THRESHOLD
        for score in scores
    ):

        reason = (
            f"At least one retrieved document "
            f"scored above the upper threshold "
            f"{RETRIEVAL_UPPER_THRESHOLD}."
        )

        verdict = "CORRECT"


    # --------------------------------------------------------
    # INCORRECT
    # --------------------------------------------------------

    elif all(
        score < RETRIEVAL_LOWER_THRESHOLD
        for score in scores
    ):

        reason = (
            f"All retrieved documents scored "
            f"below the lower threshold "
            f"{RETRIEVAL_LOWER_THRESHOLD}."
        )

        verdict = "INCORRECT"

        # Discard internal documents.
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


    print()
    print("=" * 100)
    print("RETRIEVAL EVALUATION RESULT")
    print("=" * 100)


    print(
        f"Scores: {scores}"
    )

    print(
        f"Upper threshold: "
        f"{RETRIEVAL_UPPER_THRESHOLD}"
    )

    print(
        f"Lower threshold: "
        f"{RETRIEVAL_LOWER_THRESHOLD}"
    )

    print(
        f"Verdict: {verdict}"
    )

    print(
        f"Reason: {reason}"
    )


    return {
        "retrieval_scores": scores,
        "good_docs": good_docs,
        "verdict": verdict,
        "reason": reason,
    }


# ============================================================
# 20. QUERY REWRITE NODE
# ============================================================

def rewrite_query(
    state: State,
) -> State:

    """
    Rewrite the original user question into a
    search-optimized query.

    This node is ONLY reached when internal
    retrieval is classified as INCORRECT.
    """

    question = state["question"]


    print()
    print("=" * 100)
    print("QUERY REWRITE")
    print("=" * 100)


    print(
        f"Original question:\n{question}"
    )


    # --------------------------------------------------------
    # Gemini rewrites the query
    # --------------------------------------------------------

    output = (
        query_rewrite_prompt | llm
    ).invoke(
        {
            "question": question,
        }
    )


    rewritten_query = (
        output.content
        .strip()
    )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not rewritten_query:

        rewritten_query = question


    print()
    print(
        f"Rewritten search query:\n"
        f"{rewritten_query}"
    )


    return {
        "search_query": rewritten_query,
    }


# ============================================================
# 21. TAVILY WEB SEARCH NODE
# ============================================================

def web_search(
    state: State,
) -> State:

    """
    Search the web using the rewritten query.

    The resulting documents are passed into the
    SAME knowledge refinement node used for internal
    retrieval.
    """

    question = state["question"]

    search_query = state["search_query"]


    print()
    print("=" * 100)
    print("TAVILY WEB SEARCH")
    print("=" * 100)


    print(
        f"Original question:\n{question}"
    )

    print(
        f"\nSearch query:\n{search_query}"
    )


    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    tavily_api_key = os.getenv(
        "TAVILY_API_KEY"
    )


    if not tavily_api_key:

        raise RuntimeError(
            "TAVILY_API_KEY is not configured. "
            "Add TAVILY_API_KEY to your .env file."
        )


    # --------------------------------------------------------
    # TAVILY TOOL
    # --------------------------------------------------------

    tavily_search = TavilySearchResults(
        max_results=TAVILY_MAX_RESULTS,
        tavily_api_key=tavily_api_key,
    )


    # --------------------------------------------------------
    # SEARCH USING REWRITTEN QUERY
    # --------------------------------------------------------

    search_results = tavily_search.invoke(
        {
            "query": search_query,
        }
    )


    print()
    print(
        f"Tavily returned "
        f"{len(search_results)} results."
    )


    # --------------------------------------------------------
    # CONVERT TO DOCUMENTS
    # --------------------------------------------------------

    web_docs: List[Document] = []


    for index, result in enumerate(
        search_results,
        start=1,
    ):

        if not isinstance(
            result,
            dict,
        ):

            continue


        title = result.get(
            "title",
            "",
        )


        url = result.get(
            "url",
            "",
        )


        content = result.get(
            "content",
            "",
        )


        score = result.get(
            "score",
            None,
        )


        if not content:

            continue


        content = content.strip()


        if not content:

            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "source": url,
            "title": title,
            "url": url,
            "retrieval_source": "tavily",
            "search_query": search_query,
        }


        if score is not None:

            metadata[
                "tavily_score"
            ] = score


        # ----------------------------------------------------
        # LangChain Document
        # ----------------------------------------------------

        web_document = Document(
            page_content=content,
            metadata=metadata,
        )


        web_docs.append(
            web_document
        )


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print()
        print("-" * 100)

        print(
            f"WEB RESULT {index}"
        )

        print(
            f"Title: {title}"
        )

        print(
            f"URL: {url}"
        )


        if score is not None:

            print(
                f"Tavily score: {score}"
            )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if not web_docs:

        print()
        print(
            "Tavily did not return usable results."
        )

    else:

        print()
        print(
            f"Converted {len(web_docs)} "
            f"web results into LangChain Documents."
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Web documents become good_docs so that the SAME
    # refine() node handles them.
    # --------------------------------------------------------

    return {
        "web_docs": web_docs,
        "good_docs": web_docs,
    }


# ============================================================
# 22. SENTENCE DECOMPOSITION
# ============================================================

def decompose_to_sentences(
    text: str,
) -> List[str]:

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


# ============================================================
# 23. T5 STRIP CLASSIFIER
# ============================================================

def classify_strip(
    question: str,
    strip: str,
):

    """
    Run the CRAG T5 evaluator at the
    knowledge-strip level.

    Score >= -0.5:
        KEEP

    Score < -0.5:
        DROP
    """

    separator = filter_tokenizer.sep_token


    if separator is None:

        separator = "</s>"


    model_input = (
        f"{question} "
        f"{separator} "
        f"{strip}"
    )


    inputs = filter_tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=T5_MAX_LENGTH,
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = filtering_model(
            **inputs
        )


    score = (
        outputs.logits
        .squeeze()
        .item()
    )


    keep = (
        score >= STRIP_FILTER_THRESHOLD
    )


    return score, keep


# ============================================================
# 24. KNOWLEDGE REFINEMENT
# ============================================================

def refine(
    state: State,
) -> State:

    """
    Shared knowledge refinement node.

    It can receive:

    1. Internal FAISS documents
    2. Tavily web documents

    The source of the documents does not matter.

    Both follow:

        Documents
            ↓
        Sentence decomposition
            ↓
        T5 strip evaluation
            ↓
        KEEP / DROP
            ↓
        Recomposition
    """

    question = state["question"]

    refinement_docs = state["good_docs"]


    print()
    print("=" * 100)
    print("KNOWLEDGE REFINEMENT")
    print("=" * 100)


    if not refinement_docs:

        print(
            "No documents available for refinement."
        )


        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }


    print(
        f"Documents entering refinement: "
        f"{len(refinement_docs)}"
    )


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

    strips = decompose_to_sentences(
        context
    )


    print(
        f"Generated {len(strips)} knowledge strips."
    )


    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    kept_strips: List[str] = []


    for index, strip in enumerate(
        strips,
        start=1,
    ):

        score, keep = classify_strip(
            question=question,
            strip=strip,
        )


        print()
        print("-" * 100)

        print(
            f"STRIP {index}/{len(strips)}"
        )

        print(
            f"Text:\n{strip}"
        )

        print(
            f"\nT5 score: "
            f"{score:.6f}"
        )

        print(
            f"Strip threshold: "
            f"{STRIP_FILTER_THRESHOLD}"
        )


        if keep:

            print(
                "Decision: KEEP"
            )


            kept_strips.append(
                strip
            )

        else:

            print(
                "Decision: DROP"
            )


    # --------------------------------------------------------
    # RECOMPOSE
    # --------------------------------------------------------

    refined_context = "\n".join(
        kept_strips
    ).strip()


    print()
    print("=" * 100)
    print("KNOWLEDGE REFINEMENT COMPLETE")
    print("=" * 100)


    print(
        f"Original strips: "
        f"{len(strips)}"
    )

    print(
        f"Kept strips: "
        f"{len(kept_strips)}"
    )

    print(
        f"Dropped strips: "
        f"{len(strips) - len(kept_strips)}"
    )


    return {
        "strips": strips,
        "kept_strips": kept_strips,
        "refined_context": refined_context,
    }


# ============================================================
# 25. AMBIGUOUS ROUTE
# ============================================================

def ambiguous_node(
    state: State,
) -> State:

    """
    Placeholder for the next CRAG phase.
    """

    return {
        "answer": (
            "Retrieval was classified as AMBIGUOUS. "
            "The internal + external knowledge path "
            "will be implemented in the next CRAG phase.\n\n"
            f"Reason: {state['reason']}"
        )
    }


# ============================================================
# 26. ROUTER AFTER RETRIEVAL EVALUATION
# ============================================================

def route_after_evaluation(
    state: State,
) -> Literal[
    "refine",
    "rewrite_query",
    "ambiguous",
]:

    verdict = state["verdict"]


    if verdict == "CORRECT":

        return "refine"


    if verdict == "INCORRECT":

        return "rewrite_query"


    return "ambiguous"


# ============================================================
# 27. FINAL ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. "
            "Answer ONLY using the provided "
            "refined context. "
            "Do not use outside knowledge. "
            "If the refined context is empty "
            "or insufficient, say exactly: "
            "'I don't know based on the provided books.'",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Refined context:\n"
            "{refined_context}",
        ),
    ]
)


# ============================================================
# 28. FINAL ANSWER GENERATION
# ============================================================

def generate(
    state: State,
) -> State:

    print()
    print("=" * 100)
    print("ANSWER GENERATION")
    print("=" * 100)


    output = (
        answer_prompt | llm
    ).invoke(
        {
            "question": state["question"],
            "refined_context": state[
                "refined_context"
            ],
        }
    )


    return {
        "answer": output.content,
    }


# ============================================================
# 29. BUILD LANGGRAPH
# ============================================================

print()
print("=" * 100)
print("BUILDING LANGGRAPH")
print("=" * 100)


graph = StateGraph(
    State
)


# ============================================================
# NODES
# ============================================================

graph.add_node(
    "retrieve",
    retrieve,
)


graph.add_node(
    "evaluate_retrieval",
    evaluate_retrieval,
)


# IMPORTANT:
# Query rewrite is now an actual LangGraph node.

graph.add_node(
    "rewrite_query",
    rewrite_query,
)


graph.add_node(
    "web_search",
    web_search,
)


graph.add_node(
    "refine",
    refine,
)


graph.add_node(
    "generate",
    generate,
)


graph.add_node(
    "ambiguous",
    ambiguous_node,
)


# ============================================================
# INITIAL FLOW
# ============================================================

graph.add_edge(
    START,
    "retrieve",
)


graph.add_edge(
    "retrieve",
    "evaluate_retrieval",
)


# ============================================================
# RETRIEVAL EVALUATION ROUTING
# ============================================================

graph.add_conditional_edges(
    "evaluate_retrieval",

    route_after_evaluation,

    {
        # Correct retrieval:
        #
        # evaluate → refine
        "refine": "refine",

        # Incorrect retrieval:
        #
        # evaluate → rewrite → web search
        "rewrite_query": "rewrite_query",

        # Ambiguous:
        "ambiguous": "ambiguous",
    },
)


# ============================================================
# CORRECT PATH
# ============================================================

graph.add_edge(
    "refine",
    "generate",
)


# ============================================================
# INCORRECT PATH
# ============================================================

# The critical missing connection from the previous version.

graph.add_edge(
    "rewrite_query",
    "web_search",
)


graph.add_edge(
    "web_search",
    "refine",
)


# ============================================================
# GENERATION
# ============================================================

graph.add_edge(
    "generate",
    END,
)


# ============================================================
# AMBIGUOUS PATH
# ============================================================

graph.add_edge(
    "ambiguous",
    END,
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


print(
    "LangGraph compiled successfully."
)


# ============================================================
# 30. SAVE GRAPH DIAGRAM
# ============================================================

try:

    with open(
        "CRAG-complete-flow.png",
        "wb",
    ) as f:

        f.write(
            app.get_graph().draw_mermaid_png()
        )


    print(
        "Graph diagram saved to "
        "CRAG-complete-flow.png"
    )


except Exception as exc:

    print(
        "Could not save graph diagram:"
    )

    print(exc)


# ============================================================
# 31. RUN APPLICATION
# ============================================================

question = (
    "Explain the bias-variance tradeoff"
)


print()
print("=" * 100)
print("RUNNING CORRECTIVE RAG")
print("=" * 100)

print(
    f"Question: {question}"
)


result = app.invoke(
    {
        "question": question,

        "docs": [],

        "good_docs": [],

        "retrieval_scores": [],

        "verdict": "",

        "reason": "",

        "search_query": "",

        "web_docs": [],

        "strips": [],

        "kept_strips": [],

        "refined_context": "",

        "answer": "",
    }
)


# ============================================================
# 32. RETRIEVAL RESULT
# ============================================================

print()
print("=" * 100)
print("CRAG RESULT")
print("=" * 100)


print(
    f"\nRetrieval verdict: "
    f"{result['verdict']}"
)


print(
    f"Reason: "
    f"{result['reason']}"
)


# ============================================================
# 33. RETRIEVAL SCORES
# ============================================================

print()
print("=" * 100)
print("RETRIEVAL SCORES")
print("=" * 100)


for index, score in enumerate(
    result["retrieval_scores"],
    start=1,
):

    print(
        f"Document {index}: "
        f"{score:.6f}"
    )


# ============================================================
# 34. QUERY REWRITE RESULT
# ============================================================

if result.get("search_query"):

    print()
    print("=" * 100)
    print("REWRITTEN SEARCH QUERY")
    print("=" * 100)

    print(
        result["search_query"]
    )


# ============================================================
# 35. WEB SEARCH RESULTS
# ============================================================

if result["web_docs"]:

    print()
    print("=" * 100)
    print("TAVILY WEB RESULTS USED")
    print("=" * 100)


    for index, document in enumerate(
        result["web_docs"],
        start=1,
    ):

        print()
        print(
            f"--- Web Result {index} ---"
        )

        print(
            f"Title: "
            f"{document.metadata.get('title', '')}"
        )

        print(
            f"URL: "
            f"{document.metadata.get('url', '')}"
        )


# ============================================================
# 36. FINAL ANSWER
# ============================================================

print()
print("=" * 100)
print("FINAL ANSWER")
print("=" * 100)


print(
    result["answer"]
)


# ============================================================
# 37. REFINED CONTEXT
# ============================================================

print()
print("=" * 100)
print("REFINED CONTEXT")
print("=" * 100)


if result["refined_context"]:

    print(
        result["refined_context"]
    )

else:

    print(
        "No refined context was generated."
    )


# ============================================================
# 38. KEPT STRIPS
# ============================================================

print()
print("=" * 100)
print("KEPT STRIPS")
print("=" * 100)


if result["kept_strips"]:

    for index, strip in enumerate(
        result["kept_strips"],
        start=1,
    ):

        print(
            f"\n{index}. {strip}"
        )

else:

    print(
        "No strips were kept."
    )


# ============================================================
# 39. RETRIEVED INTERNAL DOCUMENT SUMMARY
# ============================================================

print()
print("=" * 100)
print("RETRIEVED INTERNAL DOCUMENTS")
print("=" * 100)


for index, document in enumerate(
    result["docs"],
    start=1,
):

    print()
    print(
        f"--- Document {index} ---"
    )

    print(
        f"Source: "
        f"{document.metadata.get('source', 'unknown')}"
    )

    print(
        f"Page: "
        f"{document.metadata.get('page', 'unknown')}"
    )


    if index <= len(
        result["retrieval_scores"]
    ):

        print(
            f"T5 score: "
            f"{result['retrieval_scores'][index - 1]:.6f}"
        )