from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

load_dotenv()

generator_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluator_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
optimizer_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
    

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

structured_evaluator_llm = evaluator_model.with_structured_output(TweetEvaluation)

def generate_tweet(state: TweetState):
    topic = state["topic"]
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X Influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on the topic: "{topic}".
        Rules:
        - Do not use question answer format
        - Max 280 characters
        - The observational humor, irony, sarcasm, or clutural references.
        - Think in meme logic, punchies, or relatable takes
        - Use simple, day to day english.
        - This is version {state['iteration'] + 1}
        """)
    ]
    response = generator_model.invoke(messages).content

    return {"tweet": response, "tweet_history": [response]}

    

def evaluate_tweet(state: TweetState):
    
    # prompt
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
                    Evaluate the following tweet:
                    Tweet: "{state['tweet']}"
                    Use the criteria below to evaluate the tweet:
                        1. Originality – Is this fresh, or have you seen it a hundred times before?  
                        2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
                        3. Punchiness – Is it short, sharp, and scroll-stopping?  
                        4. Virality Potential – Would people retweet or share it?  
                        5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?
                    Auto-reject if:
                        - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
                        - It exceeds 280 characters
                        - It reads like a traditional setup-punchline joke
                        - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)
                    
                    ### Respond ONLY in structured format:
                        - evaluation: "approved" or "needs_improvement"  
                        - feedback: One paragraph explaining the strengths and weaknesses 
                    """)
        ]

    response = structured_evaluator_llm.invoke(messages)

    return {'evaluation':response.evaluation, 'feedback': response.feedback, "feedback_history": [response]}

def optimize_tweet(state: TweetState):
    
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
                    Improve the tweet based on this feedback: "{state['feedback']}"
                    Topic: "{state['topic']}"
                    Original Tweet: {state['tweet']}
                    Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
                    """)
        ]

    response = optimizer_model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, "tweet_history": [response]}

def route_evaluation(state: TweetState):
    if state["evaluation"] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {'approved': END, 'needs_improvement': 'optimize'}
)
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

# with open("solving_quadratic_equation_workflow.png", "wb") as f:
#     f.write(workflow.get_graph().draw_mermaid_png())

initial_state = {
    "topic": "Indian Railways",
    "iteration": 1,
    "max_iteration": 5
}

final_state = workflow.invoke(initial_state)
print(final_state)