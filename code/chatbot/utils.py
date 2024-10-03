import os
import streamlit as st
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from utils2 import llm_precise
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
logger = get_logger('Langchain-Chatbot')


def enable_chat_history(func):
    def execute(*args, **kwargs):
        if os.environ.get("OPENAI_API_KEY"):
            # Handle current_page and cache clearing
            current_page = func.__qualname__
            if "current_page" not in st.session_state:
                st.session_state["current_page"] = current_page
            if st.session_state["current_page"] != current_page:
                try:
                    st.cache_resource.clear()
                    del st.session_state["current_page"]
                    del st.session_state["messages"]
                except:
                    pass

            # Initialize messages if not present
            if "messages" not in st.session_state:
                greeting = (
                    "Welcome! You can ask me to help you find products, answer questions about a product, "
                    "or explore related items. Just describe what you're looking for (e.g., I need a nutrient "
                    "rich moisturizer), and I'll assist!"
                )
                st.session_state["messages"] = [{"role": "assistant", "content": greeting}]
            
            # Execute the main function first
            func(*args, **kwargs)
            
            # Then render all messages
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        else:
            func(*args, **kwargs)
    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm():
    available_llms = ["gpt-4o-mini", "llama3.1:70b"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
    )

    if llm_opt == "llama3.1:70b":
        llm = llm_precise
    elif llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=os.environ.get("OPENAI_API_KEY"))
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v