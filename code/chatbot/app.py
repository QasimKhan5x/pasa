# app.py

# Streamlit page configuration
# title: Homepage
# icon: 🏠

import streamlit as st

st.set_page_config(
    page_title="eCommerce Chatbot",
    page_icon='⭐',
    layout='wide'
)

st.header("Personalization-Aware ECommerce Shopping Assistant")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/QasimKhan5x/pasa)
[![linkedin ](https://img.shields.io/badge/Qasim%20Khan-blue?logo=linkedin&color=gray)](https://www.linkedin.com/in/qasimkh/)
""")
st.write("""
This is a chatbot designed to be a generative recommender system. It uses a dataset of amazon products containing beauty and personal care products. You can use the chatbot to ask for recommendations based on your preferences.
All you need to do is ask what sort of product you are looking for and it will find the most suitable products for you.
         
Here are a few examples of how you can interact with the chatbot:

- **Greetings**: If you send a greeting or introductory message, the chatbot will return a help message explaining what the system can do.
- **Product Search**: You want to find a specific product or browse for products matching certain criteria, e.g., "I am looking for a shampoo that helps with hair loss, is sulfate-free, and costs around $20."
- **Information Retrieval**: Ask for details about a specific product or type of product, such as specifications, reviews, or usage information, e.g., "Can you tell me the ingredients of this shampoo?" or "How highly rated is this product?"
- **Reviews**: Ask what customers are saying about this product.
- **Recommendations**: Ask a broad query based on a certain use-case e.g., asking for gift recommendations, or asking for something that helps with a specific problem like e.g., "I have dry skin, what products do you recommend?"
- **Comparisons**: Ask for a comparison between two products, e.g., "How does the first shampoo compare to the second one?"
         
To get started, please navigate to the chatbot section in the sidebar and start chatting with the chatbot.
""")