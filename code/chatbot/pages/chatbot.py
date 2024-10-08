import streamlit as st
from streamlit.logger import get_logger

from langchain_core.messages import HumanMessage
from graph import react_graph

# App configuration
st.set_page_config(page_title="PASA", page_icon="💬")
st.header('Personalisation-aware eCommerce Shopping Assistant')
st.write('Enhancing Chatbot Interactions through Context Awareness')
logger = get_logger('Langchain-Chatbot')

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls, question, answer))

greeting = (
    "Welcome! You can ask me to help you find products, answer questions about a product, "
    "or explore related items. Just describe what you're looking for (e.g., I need a nutrient "
    "rich moisturizer around $30), and I'll assist!"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": greeting}
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = 0

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.thread_id += 1
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

def generate_chatbot_response(prompt_input):
    config = {"configurable": {"thread_id": str(st.session_state.thread_id)}}
    try:
        result = react_graph.invoke(
            {"messages": [HumanMessage(prompt_input)]}, config=config
        )
        response = result["messages"][-1].content
    except Exception as e:
        response = "I'm sorry, but I encountered an error while processing your request."
        logger.error(f"Error generating response: {e}")

    return response

suggestions = [
    "Can you show me a night cream that helps with anti-aging?",
    "I want to find a nice luxury skincare set for my mom as a Mother's Day gift.",
    "What's a good vitamin C serum under $40 that reduces dark spots?",
    "I need a unique birthday gift for a friend who's really into natural makeup."
]

def handle_suggestion(suggestion):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": suggestion})
    
    # Display user message
    with st.chat_message("user"):
        st.write(suggestion)
    
    # Generate and append assistant response
    with st.chat_message("assistant"):
        with st.spinner("Fetching products from catalog..."):
            full_response = generate_chatbot_response(suggestion)
            st.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Log the question and answer
    print_qa("PASA-Chatbot", suggestion, full_response)

# Sidebar: Suggestion Buttons
st.sidebar.markdown("### Quick Suggestions")
for suggestion in suggestions:
    if st.sidebar.button(suggestion):
        handle_suggestion(suggestion)

# Main Page: User Input
user_input = st.chat_input("Ask me anything!")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and append assistant response
    with st.chat_message("assistant"):
        with st.spinner("Fetching products from catalog..."):
            full_response = generate_chatbot_response(user_input)
            st.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Log the question and answer
    print_qa("PASA-Chatbot", user_input, full_response)
