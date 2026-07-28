import os
import asyncio
import threading
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CHATBOT_PROJECT"

from typing import TypedDict, Annotated
from langchain_core.tools import tool, BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
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
# 1. LLM
# -------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# -------------------
# 2. Tools
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
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = model.bind_tools(tools) if tools else model

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
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
# 7. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())