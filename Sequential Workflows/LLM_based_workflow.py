import os
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# Replace with a valid Gemini model and ensure API credentials are configured.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define State
class LLMState(TypedDict):
    question: str
    answer: str

def llm_qa(state: LLMState) -> LLMState:

    question = state['question']

    prompt = f"Answer the following question: {question}"

    answer = model.invoke(prompt).content

    state['answer'] = answer

    return state

# Define Graph
graph = StateGraph(LLMState)

# Add Nodes to the Graph
graph.add_node('llm_qa', llm_qa)

# Add edge to the graph
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state = {
    'question': "What is the capital of france?",
}

final_state = workflow.invoke(initial_state)

print(f"Final State: {final_state}")

# Visualize the graph
with open("LLM-Workflow.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())