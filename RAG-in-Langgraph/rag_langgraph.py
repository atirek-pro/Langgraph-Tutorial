import os
from pathlib import Path

# =============================================================================
# Configure Hugging Face cache BEFORE importing HuggingFaceEmbeddings
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
EMBEDDING_DIR = MODELS_DIR / "embeddings"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(EMBEDDING_DIR)

from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =============================================================================
# LLM
# =============================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# =============================================================================
# Embedding Model
# =============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# =============================================================================
# Create or Load FAISS Vector Store
# =============================================================================

index_file = VECTOR_STORE_DIR / "index.faiss"

if index_file.exists():
    print("Loading existing FAISS vector store...")

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

else:
    print("Creating new FAISS vector store...")

    loader = PyPDFLoader("intro-to-ml.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vector_store.save_local(str(VECTOR_STORE_DIR))

    print("Vector store saved!")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# =============================================================================
# Tool
# =============================================================================

@tool
def rag_tool(query: str):
    """
    Retrieve relevant information from the PDF document.
    Use this tool when the user asks factual or conceptual
    questions that may be answered from the stored documents.
    """

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
    }

tools = [rag_tool]

llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# LangGraph State
# =============================================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

# with open("rag-langgraph.png", "wb") as f:
#     f.write(chatbot.get_graph().draw_mermaid_png())

# =============================================================================
# Example
# =============================================================================

result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the difference between supervised and unsupervised learning?"
            )
        ]
    }
)

print(result["messages"][-1].content)