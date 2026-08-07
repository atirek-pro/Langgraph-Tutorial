"""Example of deleting older messages in a LangGraph agent to manage short-term memory.

This script shows how an agent can remove earlier messages from its working memory
once the conversation grows too large. The idea is to keep the conversation manageable
and prevent the context window from becoming too crowded.
"""

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import RemoveMessage

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the language model client.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def chat(state: MessagesState):
    """Send the current conversation state to the model and return the assistant reply."""

    # The model receives the current message list from the state.
    response = model.invoke(state["messages"])

    # Return the model response as a new message in the state.
    return {"messages": [response]}


def delete_old_messages(state: MessagesState):
    """Remove older messages once the conversation becomes too long."""

    msgs = state["messages"]

    # If there are more than 10 messages, delete the earliest 6.
    # This simulates a simple cleanup policy for short-term memory.
    if len(msgs) > 10:
        to_remove = msgs[:6]

        # RemoveMessage tells LangGraph to delete these specific messages from the state.
        # Each message is removed by its unique message id.
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    # If the conversation is still small, do nothing.
    return {}


# Build a state graph with two nodes:
# 1. chat -> sends messages to the LLM
# 2. cleanup -> removes older messages after each reply
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("cleanup", delete_old_messages)

# Flow:
# START -> chat -> cleanup -> END
builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup")  # Run deletion after each response
builder.add_edge("cleanup", "__end__")

# Use an in-memory checkpointer so the conversation state is preserved across turns.
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1"}}

# Run multiple turns in the same conversation thread.
# Each turn will add a new user message and then trigger cleanup if needed.
graph.invoke({"messages": [{"role": "user", "content": "Hi, I'm Nitish"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Tell me about LangGraph"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Langchain"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Quantum Mechanics"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Gen AI"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, config)

# Inspect the final state after cleanup.
snap = graph.get_state(config)
print("Stored messages after cleanup:", len(snap.values["messages"]))