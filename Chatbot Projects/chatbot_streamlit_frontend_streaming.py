import uuid
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from chat_bot_langgraph_backend import chatbot, retrieve_all_threads

# ***************************************************** Utility Functions******************************************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

def get_last_user_message(thread_id):
    """Get the last user question from a conversation thread for display"""
    try:
        messages = load_conversation(thread_id)
        # Find the last HumanMessage in reverse order
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                # Truncate to reasonable length for display
                content = message.content[:50]
                if len(message.content) > 50:
                    content += "..."
                return content
        # If no user message found, it's an empty conversation
        return None
    except:
        return None


def normalize_text_content(content):
    """Convert streamed LangChain content into a plain text string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("content"), list):
                    parts.append(normalize_text_content(item["content"]))
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "".join(parts)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        if isinstance(content.get("content"), list):
            return normalize_text_content(content["content"])

    return str(content or "")


def stream_assistant_reply(user_input, thread_id):
    """Stream only the final assistant reply and skip tool messages."""
    config = {
        'configurable': {'thread_id': thread_id},
        'metadata': {'thread_id': thread_id},
        'run_name': 'chat_turn'
    }

    def iter_stream():
        for message_chunk, _ in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=config,
            stream_mode='messages'
        ):
            if isinstance(message_chunk, ToolMessage):
                continue

            if isinstance(message_chunk, AIMessage):
                text = normalize_text_content(getattr(message_chunk, 'content', message_chunk))
                if text:
                    yield text

    return iter_stream()


# ***************************************************** Session Set-up******************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ***************************************************** Side-Bar UI******************************************************
st.sidebar.title("ChatVerse")

if st.sidebar.button("Create New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    # Display the last user question instead of thread_id
    button_label = get_last_user_message(thread_id)
    # Skip empty conversations
    if button_label is not None:
        if st.sidebar.button(button_label, key=str(thread_id)):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []

            for message in messages:
                if isinstance(message, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                
                temp_messages.append({'role': role, 'content': message.content})
            
            st.session_state['message_history'] = temp_messages


# **********************************************************Loading the Conversation history*********************************** 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here!")

if user_input:
    
    # Adding the user message to the history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            stream_assistant_reply(user_input, st.session_state['thread_id'])
        )

    # Adding the ai message to the history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})