from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# ****** SECTION 1: Read-only memory chat ******
# This section seeds a store with example user details and
# builds a graph that reads the store but does not write to it.
store = InMemoryStore()
user_id = "u1"
user_details = ("user", user_id, "details")

store.put(user_details, "profile_1", {"data": "Name: Atirek"})
store.put(user_details, "profile_2", {"data": "Profession: Teaches AI on Instagram"})
store.put(user_details, "preference_1", {"data": "Prefers concise answers"})
store.put(user_details, "preference_2", {"data": "Likes examples in Python"})
store.put(user_details, "project_1", {"data": "Building MCP servers (Python-based project)"})

print("\n****** SECTION 1: Seeded store with long-term memory ******")
print(f"user_id: {user_id}")
print("Stored memory items:")
for item in store.search(user_details):
    print(" -", item.value["data"])

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored
assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    - always address the user by name (e.g., "Sure, Atirek...") when appropriate
    - refer to known projects, tools, or preferences (e.g., "your MCP server python-based project")
    - adjust the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..."
say "Since your project is built with TypeScript..."

Use personalization especially in:
    - greetings and transitions
    - help or guidance tailored to tools and frameworks the user uses
    - follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile.

The user's memory (which may be empty) is provided as: {user_details_content}
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    items = store.search(namespace)
    if items:
        user_details_content = "\n".join(f"- {item.value.get('data', '')}" for item in items)
    else:
        user_details_content = ""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details_content)
    system_msg = SystemMessage(content=system_prompt)

    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}
print("\n****** SECTION 1: Run read-only memory chat ******")
print("User message: Explain gen ai in simple terms.")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Explain gen ai in simple terms."}]},
    config,
)
print("Assistant response:")
print(result["messages"][-1].content)

# ****** SECTION 2: Create new memories only ******
# This section builds a graph that decides what to store from
# the user's latest input, but it does not use memory when answering.
store = InMemoryStore()
extractor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[str] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = extractor_llm.with_structured_output(MemoryDecision)

def remember_only_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")
    last_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "Extract LONG-TERM memories from the user's message.\n"
                    "Only store stable, user-specific info (identity, preferences, ongoing projects).\n"
                    "Do NOT store transient info.\n"
                    "Return should_write=false if nothing is worth storing.\n"
                    "Each memory should be a short atomic sentence."
                )
            ),
            {"role": "user", "content": last_msg},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            store.put(namespace, str(uuid.uuid4()), {"data": mem})

    return {"messages": [{"role": "assistant", "content": "Noted."}]}

builder = StateGraph(MessagesState)
builder.add_node("remember", remember_only_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}
print("\n****** SECTION 2: Run new-memory creation graph ******")
for user_message in [
    "Hi my name is Atirek",
    "I teach AI on instagram",
    "My favorite programming language is Python",
]:
    res = graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    print("User message:", user_message)
    print("Assistant:", res["messages"][-1].content)

print("Stored memories after SECTION 2:")
for item in store.search(("user", "u1", "details")):
    print(" -", item.value["data"])

# ****** SECTION 3: Create memories with deduplication ******
# This section uses the existing memory store to avoid saving duplicates.
store = InMemoryStore()
memory_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return an empty list.
"""

def chat_creates_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    existing_items = store.search(namespace)
    existing_texts = [item.value.get("data", "") for item in existing_items if item.value.get("data")]
    user_details_content = "\n".join(f"- {text}" for text in existing_texts) if existing_texts else "(empty)"

    last_text = state["messages"][-1].content
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=user_details_content)),
            {"role": "user", "content": f"USER MESSAGE:\n{last_text}"},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text})

    return {"messages": [{"role": "assistant", "content": "Noted."}]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_creates_memory_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}
print("\n****** SECTION 3: Run deduplicated memory creation graph ******")

r1 = graph.invoke({"messages": [{"role": "user", "content": "My name is Atirek"}]}, config)
print("User message: My name is Atirek")
print("Assistant:", r1["messages"][-1].content)

r2 = graph.invoke({"messages": [{"role": "user", "content": "I like Python for programming."}]}, config)
print("User message: I like Python for programming.")
print("Assistant:", r2["messages"][-1].content)

print("Stored memories after SECTION 3:")
for item in store.search(("user", "u1", "details")):
    print(" -", item.value["data"])


# ****** SECTION 4: Merged Workflow ******
# merged flow starts here: this section extracts/stores new memories, then runs a memory-aware chat
store = InMemoryStore()

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Atirek...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user’s memory (which may be empty) is provided as: {user_details_content}
"""

memory_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # existing memory
    items = store.search(ns)
    existing = "\n".join(it.value["data"] for it in items) if items else "(empty)"

    # last user message
    last_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_msg},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(ns, str(uuid.uuid4()), {"data": mem.text})

    return {}  # no message change

chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details or "(empty)"
        )
    )

    response = chat_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

with open("chat-with-LTM-memory-workflow.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

config = {"configurable": {"user_id": "u1"}}

print("\n****** SECTION 4: Run merged memory+chat workflow (this output is for SECTION 4) ******")

result = graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Atirek Gupta"}]}, config)
print(result['messages'][-1].content)

for it in store.search(("user", "u1", "details")):
    print(it.value["data"])

result = graph.invoke({"messages": [{"role": "user", "content": "I teach AI on Instagram"}]}, config)
print(result['messages'][-1].content)

for it in store.search(("user", "u1", "details")):
    print(it.value["data"])

result = graph.invoke({"messages": [{"role": "user", "content": "Explain GenAI simply"}]}, config)
print(result['messages'][-1].content)

for it in store.search(("user", "u1", "details")):
    print(it.value["data"])