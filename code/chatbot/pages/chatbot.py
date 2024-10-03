import utils
import streamlit as st
from langchain_core.messages import HumanMessage
from graph import react_graph


st.set_page_config(page_title="PASA", page_icon="💬")
st.header('Personalisation-aware eCommerce Shopping Assistant')
st.write('Enhancing Chatbot Interactions through Context Awareness')

# chatbot.py

class ContextChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.graph = react_graph
        self.config = {"configurable": {"thread_id": "1"}}


    @utils.enable_chat_history
    def main(self):
        graph = self.graph
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                # st_cb = StreamHandler(st.empty())
                result = graph.invoke(
                    {"messages": [HumanMessage(user_query)]},
                    config=self.config
                )
                response = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(ContextChatbot, user_query, response)

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()