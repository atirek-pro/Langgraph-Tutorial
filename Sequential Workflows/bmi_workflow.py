from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import TypedDict

# Define State
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight / (height**2)

    state['bmi'] = round(bmi, 2)

    return state

def lable_bmi(state: BMIState) -> BMIState:
    
    bmi = state['bmi']
    
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif 18.5 <= bmi < 25:
        state['category'] = "Normal"
    elif 25 <= bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"
    
    return state

# Define Graph
graph = StateGraph(BMIState)

# Add Nodes to the Graph
graph.add_node('calulate_bmi', calculate_bmi)
graph.add_node('lable_bmi', lable_bmi)

# Add edge to the graph
graph.add_edge(START, 'calulate_bmi')
graph.add_edge('calulate_bmi', 'lable_bmi')
graph.add_edge('lable_bmi', END)

# Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state = {
    'weight_kg': 80,
    'height_m': 1.73
}

final_state = workflow.invoke(initial_state)

print(f"Final State: {final_state}")

# Visualize the graph
with open("BMI-Workflow.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())