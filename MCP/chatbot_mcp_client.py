from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# MCP client for the local FastMCP calculator server
client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": r"D:\AI Engineering\Agentic AI\Langgraph-Tutorial\myenv\Scripts\python.exe",
            # point this at wherever you saved calculator_mcp_server.py
            "args": [r"D:\AI Engineering\Agentic AI\Langgraph-Tutorial\MCP\mcp_server.py"],
        }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot


async def main():

    chatbot = await build_graph()

    # running the graph — a query that actually maps to a calculator tool
    result = await chatbot.ainvoke({
        "messages": [HumanMessage(
            content="What is the factorial of 7, and what's the square root of 144?"
        )]
    })

    print(result['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())