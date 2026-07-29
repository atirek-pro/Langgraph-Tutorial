import os
from pathlib import Path

# =============================================================================
# Configure Hugging Face cache BEFORE importing anything that pulls in
# HuggingFace / sentence-transformers (must happen before those imports,
# since the cache location can get resolved at import time)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
EMBEDDING_DIR = MODELS_DIR / "embeddings"
VECTOR_STORE_DIR = BASE_DIR / "rag_vector_store"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(EMBEDDING_DIR)

# -------------------
# Remaining imports (safe now that HF cache env vars are set)
# -------------------
import tempfile
import asyncio
import threading
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CHATBOT_PROJECT"

from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite
import requests

# -------------------
# Dedicated async loop for backend tasks
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop (doesn't block)."""
    return _submit_async(coro)

# -------------------
# 1. LLM + embeddings
# -------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(EMBEDDING_DIR),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _thread_index_path(thread_id: str) -> str:
    """Folder on disk where a given thread's FAISS index is persisted."""
    return str(VECTOR_STORE_DIR / str(thread_id))


def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.

    Checks the in-memory cache first; if not found there, tries to load a
    previously persisted FAISS index for that thread from disk so retrievers
    survive process restarts without re-uploading the PDF.
    """
    if not thread_id:
        return None

    thread_id = str(thread_id)

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    index_path = _thread_index_path(thread_id)
    if os.path.isdir(index_path):
        try:
            vector_store = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            _THREAD_RETRIEVERS[thread_id] = retriever
            return retriever
        except Exception as e:
            print(f"Failed to load persisted FAISS index for thread {thread_id}: {e}")

    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    The index is persisted to disk under rag_vector_store/<thread_id> so it
    can be reloaded on a later run without re-ingesting the PDF.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Persist the index to disk for this thread
        index_path = _thread_index_path(thread_id)
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


mcp_client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": r"D:\AI Engineering\Agentic AI\Langgraph-Tutorial\myenv\Scripts\python.exe",
            "args": [r"D:\AI Engineering\Agentic AI\Langgraph-Tutorial\MCP\mcp_server.py"],
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(mcp_client.get_tools())
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return []

mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, rag_tool, *mcp_tools]
llm_with_tools = model.bind_tools(tools) if tools else model

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# -------------------
# 6. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS or os.path.isdir(_thread_index_path(thread_id))


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})