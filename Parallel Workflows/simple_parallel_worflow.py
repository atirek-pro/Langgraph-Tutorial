from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    
    sr: float
    bpb: float
    boundry_percent: float

    summary: str

def calculate_sr(state: BatsmanState) -> BatsmanState:

    sr = (state["runs"] / state["balls"]) * 100
    
    # passing partial state to the next node specifical done when building parallel workflows
    return {'sr': sr}

def calculate_bpb(state: BatsmanState) -> BatsmanState:

    bpb = (state["balls"]) / (state["fours"] + state["sixes"])
    
    return {'bpb': bpb}

def calculate_boundry_percent(state: BatsmanState) -> BatsmanState:

    boundry_percent = (((state["fours"] * 4) + (state["sixes"] * 6)) / state["runs"]) * 100
    return {'boundry_percent': boundry_percent}

def summary(state: BatsmanState) -> str:

    summary = f"""Strike Rate -> {state['sr']}
                  Balls per boundary -> {state['bpb']}
                  Boundry Percent -> {state['boundry_percent']}
            """
    
    return {'summary': summary}

# Graph
graph = StateGraph(BatsmanState)

# Nodes
graph.add_node("calculate_sr", calculate_sr)
graph.add_node("calculate_bpb", calculate_bpb)
graph.add_node("calculate_boundry_percent", calculate_boundry_percent)
graph.add_node("summary", summary)

# Edges
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundry_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundry_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

# Visualize the graph
with open("simple_parallel_workflow.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

# Execute the graph
initial_state = {
    'runs': 100,
    'balls': 50,
    'fours': 6,
    'sixes': 4
}

final_state = workflow.invoke(initial_state)
print(final_state)