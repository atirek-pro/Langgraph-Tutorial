import uuid
import queue
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from chat_bot_langgraph_backend import chatbot, retrieve_all_threads, submit_async_task

# ***************************************************** Utility Functions******************************************************
def extract_text(content):
    """
    Normalize a LangChain message's `.content` into a plain string.

    Some providers (e.g. Gemini via langchain_google_genai) can emit content as:
      - a plain string: "hello"
      - a list of content blocks: [{'type': 'text', 'text': 'hello'}, ...]
      - a list containing non-text / tool-related blocks that should be
        skipped for display purposes.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get('text'):
                    parts.append(block['text'])
        return "".join(parts)
    return str(content)


def stream_sync(async_gen_func, *args, **kwargs):
    """
    Bridges an async generator running on the backend's dedicated event loop
    (see chat_bot_langgraph_backend._ASYNC_LOOP) to a plain synchronous
    iterator that Streamlit's main thread can consume with a normal `for`
    loop, just like the old chatbot.stream(...) did.
    """
    q = queue.Queue()
    _SENTINEL = object()

    async def _runner():
        try:
            async for item in async_gen_func(*args, **kwargs):
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    submit_async_task(_runner())

    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


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
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                text = extract_text(message.content)
                content = text[:50]
                if len(text) > 50:
                    content += "..."
                return content
        return None
    except Exception:
        return None


def build_history_from_thread(messages):
    """
    Convert raw LangGraph message objects (Human/AI/Tool) into a list of dicts
    that carries both the assistant text AND any tool calls it made, so the
    sidebar reload can re-render the stacked tool-usage components too.
    """
    history = []
    # Map tool_call_id -> ToolMessage content, built first so we can attach
    # results to the AIMessage that requested them.
    tool_results = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results[message.tool_call_id] = {
                'name': message.name,
                'content': message.content,
            }

    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({'role': 'user', 'content': extract_text(message.content)})

        elif isinstance(message, AIMessage):
            tool_calls_display = []
            for tc in (message.tool_calls or []):
                result = tool_results.get(tc.get('id'), {})
                tool_calls_display.append({
                    'name': tc.get('name'),
                    'args': tc.get('args'),
                    'output': result.get('content'),
                })
            # Skip intermediate AI messages that only carry tool calls and no
            # content of their own but still have nothing to show.
            text = extract_text(message.content)
            if text or tool_calls_display:
                history.append({
                    'role': 'assistant',
                    'content': text,
                    'tool_calls': tool_calls_display,
                })
        # ToolMessages themselves aren't rendered as their own chat bubble;
        # they're folded into the AIMessage's tool_calls above.

    return history


def render_tool_calls(tool_calls):
    """Render a stack of tool-usage status components above the AI text."""
    for tc in tool_calls:
        name = tc.get('name') or 'tool'
        with st.status(f"🔧 Used tool: `{name}`", state="complete", expanded=False):
            if tc.get('args') is not None:
                st.markdown("**Input**")
                st.json(tc['args'])
            if tc.get('output') is not None:
                st.markdown("**Output**")
                st.write(tc['output'])


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
    button_label = get_last_user_message(thread_id)
    if button_label is not None:
        if st.sidebar.button(button_label, key=str(thread_id)):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            st.session_state['message_history'] = build_history_from_thread(messages)


# **********************************************************Loading the Conversation history***********************************
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant' and message.get('tool_calls'):
            render_tool_calls(message['tool_calls'])
        if message['content']:
            st.markdown(message['content'])

user_input = st.chat_input("Type here!")

if user_input:

    # Adding the user message to the history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Config Setup
    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {
            'thread_id': st.session_state['thread_id']
        },
        'run_name': 'chat_turn'
    }

    with st.chat_message("assistant"):
        # Container that holds the stacked tool-usage status boxes (filled in
        # as tool calls stream in), placed above the streaming text.
        tool_stack_container = st.container()
        text_placeholder = st.empty()

        full_text = ""
        # index -> accumulated {name, args (str, being built), id}
        tool_call_accum = {}
        # tool_call_id -> the st.status(...) object, so we can update it live
        status_boxes = {}
        # tool_call_id -> tool name (for finalizing the status label)
        tool_names = {}
        # Preserve tool calls (with args/output) for saving into history
        finalized_tool_calls = []

        chunk_count = 0
        try:
            for message_chunk, metadata in stream_sync(
                chatbot.astream,
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                chunk_count += 1

                # ---- Tool result coming back ----
                if isinstance(message_chunk, ToolMessage):
                    tool_id = message_chunk.tool_call_id
                    name = tool_names.get(tool_id, message_chunk.name)

                    if tool_id in status_boxes:
                        status_boxes[tool_id].update(
                            label=f"✅ Used tool: `{name}`", state="complete"
                        )
                        with status_boxes[tool_id]:
                            st.markdown("**Output**")
                            st.write(extract_text(message_chunk.content))

                    finalized_tool_calls.append({
                        'name': name,
                        'args': tool_call_accum.get(tool_id, {}).get('args_display'),
                        'output': extract_text(message_chunk.content),
                    })
                    continue

                # ---- AI message chunk: plain text and/or streamed tool calls ----
                chunk_text = extract_text(getattr(message_chunk, 'content', None))
                if chunk_text:
                    full_text += chunk_text
                    text_placeholder.markdown(full_text + "▌")

                for tc_chunk in (getattr(message_chunk, 'tool_call_chunks', None) or []):
                    idx = tc_chunk.get('index', 0)
                    entry = tool_call_accum.setdefault(idx, {'name': '', 'args': '', 'id': None})

                    if tc_chunk.get('name'):
                        entry['name'] += tc_chunk['name']
                    if tc_chunk.get('args'):
                        entry['args'] += tc_chunk['args']
                    if tc_chunk.get('id'):
                        entry['id'] = tc_chunk['id']

                    tool_id = entry['id']
                    if tool_id and entry['name'] and tool_id not in status_boxes:
                        tool_names[tool_id] = entry['name']
                        sb = tool_stack_container.status(
                            f"🔧 Calling tool: `{entry['name']}`...", state="running"
                        )
                        status_boxes[tool_id] = sb
                        tool_call_accum[tool_id] = entry

                    if tool_id:
                        entry['args_display'] = entry['args']

        except Exception as e:
            # Surface the real error instead of leaving the spinner hanging
            # with no visible feedback.
            st.error(f"Streaming failed after {chunk_count} chunk(s): {e}")
            st.exception(e)

        if chunk_count == 0:
            st.warning(
                "No chunks were received from chatbot.astream(). This means the "
                "call itself is hanging/blocking on the backend side (not a UI "
                "bug) — check your graph/checkpointer setup."
            )

        text_placeholder.markdown(full_text if full_text else "*(no text content received)*")
        ai_message = full_text

    # Adding the ai message (with any tool calls) to the history
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message,
        'tool_calls': finalized_tool_calls,
    })