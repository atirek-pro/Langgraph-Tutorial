from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Structured Output Schema
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed Feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    average_score: float

def evaluate_language(state: UPSCState):

    prompt = f"Evaluate the langauge quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"language_feedback": output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: UPSCState):

    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"analysis_feedback": output.feedback, 'individual_scores': [output.score]}

def evaluate_thoughts(state: UPSCState):

    prompt = f"Evaluate the clarity of the thought of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"clarity_feedback": output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: UPSCState):

    # summary feedback
    prompt = f"Based on the following feedbacks create a summarized feedback \n Langiage feedback - {state['language_feedback']} \n Depth of Analysis Feedback - {state['analysis_feedback']} \n Clarity of thought feedback - {state['clarity_feedback']}"
    overall_feedback = model.invoke(prompt).content

    # avg calculation
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'average_score': avg_score}

# Graph
graph = StateGraph(UPSCState)

# Nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thoughts", evaluate_thoughts)
graph.add_node("final_evaluation", final_evaluation)

# Edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thoughts')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thoughts', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

# Visualize the graph
with open("parallelization_workflow.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

# Execute the graph
with open("essay.txt", "r", encoding="utf-8") as file:
    essay = file.read()

initial_state = {
    'essay': essay
}

final_state = workflow.invoke(initial_state)
print(final_state)