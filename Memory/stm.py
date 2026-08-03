from langgraph.graph import StateGraph, START, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# This example teaches a core LangGraph idea: a graph can be stateless by default,
# but it can also remember conversation state when we add a checkpointer.
# The goal is to show that a model may not remember earlier user information
# unless the workflow preserves that context across runs.

# Load environment variables from the .env file so API keys and settings are available.
load_dotenv()

# Create the LLM instance that will respond to the messages in the workflow.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# This function is the graph node. It receives the current conversation state,
# sends the message history to the LLM, and returns the model's reply.
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Build a simple state graph using LangGraph's built-in MessagesState.
# This represents a workflow with one node that processes chat messages.
builder = StateGraph(MessagesState)

# Add the custom node to the graph so it can be executed during the workflow.
builder.add_node("call_model", call_model)

# Connect the workflow start point to the node so execution begins there.
builder.add_edge(START, "call_model")

# Compile the graph into a runnable LangGraph application.
# At this point, the graph is stateless and does not preserve memory between calls.
graph = builder.compile()

# First demonstration: without memory, the model is asked about the user's name
# in a separate invocation. This shows that the conversation context is not preserved.
print("Graph without memory:")
print("\n")
print(graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Atirek."}]}))
print(graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}))
print("\n")

# To teach memory in LangGraph, we add an InMemorySaver checkpointer.
# This stores the graph state between runs for the same thread, allowing the model
# to use earlier messages as part of the ongoing conversation context.
checkpointer = InMemorySaver()
graph_with_stm = builder.compile(checkpointer=checkpointer)

# Each thread ID acts like a separate conversation session.
# This lets us compare memory behavior for different conversation histories.
config = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}

# Second demonstration: with memory enabled, the same user name is provided first,
# and the next question asks the model to recall it. This illustrates short-term memory.
print("Graph with memory:")
print("\n")
print(graph_with_stm.invoke({"messages": [{"role": "user", "content": "Hi! My name is Atirek."}]}, config=config))
print(graph_with_stm.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config))
print("\n")

# This final block shows the saved conversation state for the thread.
# It helps learners inspect what the checkpointer is actually preserving.
snap = graph_with_stm.get_state(config)
vals = snap.values
for m in vals.get("messages", []):
    print("-", type(m).__name__, ":", m.content)

# Limitations of in-memory storage:
# - Data is lost when the process stops or the application restarts.
# - It is not suitable for production systems that need durable, long-term memory.
# - It does not scale well for many users or large conversation histories.
# - It is mainly useful for learning, testing, and small demos.