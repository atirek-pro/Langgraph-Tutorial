# Benifits of Persistence in Langgraph
# 1. Short-term memory
# 2. Fault Tollerance
# 3. Can introduce HITL(human-In-The-Loop)
# 4. Time Travel

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    prompt = f"generate a joke on the topic {state['topic']}"
    response = model.invoke(prompt).content
    return {'joke': response}

def generate_explanation(state: JokeState):
    prompt = f"write an explanation for the joke - {state['joke']}"
    response = model.invoke(prompt).content
    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config_1 = {"configurable": {"thread_id": "1"}}
print(workflow.invoke({'topic': "pizza"}, config=config_1))

print("Workflow final state values")
print(workflow.get_state(config_1))

print("Workflow Intermediate state values")
print(list(workflow.get_state_history(config_1)))

# Time Travel
history = list(workflow.get_state_history(config_1))
checkpoint_to_travel = history[0].parent_config["configurable"]["checkpoint_id"]

print(workflow.get_state({"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_to_travel}}))
print(workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_to_travel}}))
print(list(workflow.get_state_history(config_1)))