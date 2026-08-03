"""Example of short-term memory trimming in a LangGraph agent.

This script shows a simplified way to mimic short-term memory by keeping only the
most recent conversation history that fits within a fixed token budget before sending
messages to the LLM.
"""

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the language model client.
# In a real application, you would usually configure the model with your API key.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Maximum number of tokens allowed for the short-term memory window.
# This is the budget we use to decide how much history to keep before calling the model.
MAX_TOKENS = 150


def call_model(state: MessagesState):
    """Process the current conversation state and trim old messages before calling the model."""

    # The state contains the full chat history for this thread so far.
    # We do not want to send all past messages to the model because that can become
    # expensive and may overflow the context window.
    messages = trim_messages(
        state["messages"],
        strategy="last",  # Keep the most recent messages that fit within the budget.
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS,
    )

    # Show the trimmed conversation size for learning/debugging purposes.
    print("Current Token Count ->", count_tokens_approximately(messages=messages))

    # Print the trimmed messages so the reader can see what context is actually preserved.
    for message in messages:
        print(message.content)

    # Send the trimmed conversation to the model.
    response = model.invoke(messages)

    # Return the model's reply so it becomes the next message in the state.
    return {"messages": [response]}


# Build a simple state graph with one node that handles the model call.
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# The checkpointer allows the graph to remember the conversation across multiple invokes.
# This is important because the agent should behave as if it has memory within the thread.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# A thread id groups all messages belonging to the same conversation.
config = {"configurable": {"thread_id": "chat-1"}}

# Each invoke adds a new user message to the same conversation thread.
# The graph will reuse previous state and trim the message history before calling the model.
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Nitish."}]},
    config,
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "I am learning LangGraph."}]},
    config,
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Can you explain short term memory?"}]},
    config,
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config,
)

# Print the final reply from the latest model call.
print(result["messages"][-1].content)

# Print the full saved state for this thread to show the conversation history that was preserved.
for item in graph.get_state({"configurable": {"thread_id": "chat-1"}}).values["messages"]:
    print(item.content)
    print("-" * 120)

# Limitations of trimming:
# 1. Important older context may be lost if it does not fit in the token budget.
# 2. The model may forget details that were useful for long-term consistency.
# 3. Trimming is a simple heuristic; it does not understand which information is truly important.
# 4. If the conversation becomes too long, the agent may appear less coherent or less personalized.
# 5. This approach is useful for efficiency, but it is not a perfect substitute for true memory management.