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
class BlogState(TypedDict):
    topic: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    
    title = state['topic']

    prompt = f'Generate a detailed outline for a blog on the topic: {title}'
    outline = model.invoke(prompt).content

    state['outline'] = outline

    return state

def create_blog(state: BlogState) -> BlogState:
    
    title = state['topic']
    outline = state['outline']

    prompt = f"Write a detailed blog on the title - {title} based on the following outline \n {outline}"
    content = model.invoke(prompt).content

    state['content'] = content

    return state

# Define Graph
graph = StateGraph(BlogState)

# Add Nodes to the Graph
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# Add edge to the graph
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

# Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state = {
    'topic': "How AI can help propgrammers increase productivity",
}

final_state = workflow.invoke(initial_state)

# print(f"Outline: \n {final_state['outline']}")
print(f"Content: \n {final_state['content']}")

# Visualize the graph
with open("Prompt-Chaining-Workflow.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

# Execute the graph
initial_state = {
    'topic': "How AI can help propgrammers increase productivity",
}

final_state = workflow.invoke(initial_state)