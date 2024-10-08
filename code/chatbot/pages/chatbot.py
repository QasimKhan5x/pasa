import streamlit as st
from streamlit.logger import get_logger

from langchain_core.messages import HumanMessage
from graph import react_graph

# App title
st.set_page_config(page_title="PASA", page_icon="💬")
st.header('Personalisation-aware eCommerce Shopping Assistant')
st.write('Enhancing Chatbot Interactions through Context Awareness')
logger = get_logger('Langchain-Chatbot')

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))


greeting = (
    "Welcome! You can ask me to help you find products, answer questions about a product, "
    "or explore related items. Just describe what you're looking for (e.g., I need a nutrient "
    "rich moisturizer around $30), and I'll assist!"
)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": greeting}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

thread_id = 0
def clear_chat_history():
    global thread_id
    thread_id += 1
    st.session_state.messages = [{"role": "assistant", "content": greeting}]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_chatbot_response(prompt_input):
    global thread_id
    config = {"configurable": {"thread_id": str(thread_id)}}
    result = react_graph.invoke(
        {"messages": [HumanMessage(prompt_input)]}, config=config
    )
    response = result["messages"][-1].content

    return response


# User-provided prompt
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Fetching products from catalog..."):
            full_response = generate_chatbot_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    # Log the question and answer
    print_qa(generate_chatbot_response, prompt, full_response)
