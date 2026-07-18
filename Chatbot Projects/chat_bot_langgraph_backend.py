from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# with open("chatbot.png", "wb") as f:
#     f.write(chatbot.get_graph().draw_mermaid_png())

# initial_state = {
#     'messages': [HumanMessage(content='What is the capital of india?')]
# }

# final_state = chatbot.invoke(initial_state)
# print(final_state)

# thread_id = '1'

# while True:
#     user_message = input('Type here: ')
#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break
    
#     config = {'configurable': {'thread_id': thread_id}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)

#     print("AI: ", response['messages'][-1].content)
