import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class CrashState(TypedDict):
    input: str
    Step1: str
    Step2: str
    Step3: str

def step_1(state: CrashState):
    print("✅ Step 1 Executed")
    return {"Step1": "done", "input": state["input"]}

def step_2(state: CrashState):
    print(" Step 2 hanging...... now manually interpret")
    time.sleep(30) # simulating long running hang
    return {"Step2": "done"}

def step_3(state: CrashState):
    print("✅ Step 3 Executed")
    return {"done": True}

builder = StateGraph(CrashState)

builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

try:
    print("Running graph: Please manually interept during step 2.")
    graph.invoke({"input": "Start"}, config={"configurable": {"thread_id": "thread_1"}})
except KeyboardInterrupt:
    print("Kernel manually interpreted (crash simulated)")

print("\n")
print("Final State of the graph")
print(graph.get_state({"configurable": {"thread_id": "thread_1"}}))

print("\n")
print("Intermediate State of the graph")
print(list(graph.get_state_history({"configurable": {"thread_id": "thread_1"}})))

# Re-run to show fault tolerance resume
final_state = graph.invoke(None, config={"configurable": {"thread_id": "thread_1"}})
print("\n Final State: ", final_state)