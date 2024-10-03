from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from utils2 import OverallState
from intent import get_intent
from entity import entity_identification_chain
from others import product_reference_chain, product_reference_list_chain, explain_product, explain_reviews, compare_products
from product_search import product_search
from recommendation import recommendation


def intent_router(state: MessagesState) -> OverallState:
    user_input = state["messages"][-1]
    return {"intent": get_intent(user_input)}

def entity_identification(state: OverallState) -> OverallState:        
    user_input = state["messages"][-1]
    entities = entity_identification_chain.invoke({"query": user_input})
    return {"entities": entities}

def hello(state: OverallState) -> MessagesState:
    greeting = "Welcome! You can ask me to help you find products, answer questions about a product, or explore related items. Just describe what you're looking for (e.g., I need a nutrient rich moisturizer), and I'll assist!"
    return {"messages": AIMessage(greeting)}

def bye(state: OverallState) -> MessagesState:
    return {"messages": AIMessage("Goodbye!")}

def product_reference(state: OverallState) -> OverallState:
    user_input = state["messages"][-1]
    # last 3 messages are the product details
    product_index = product_reference_chain.invoke(
        {"history": state["messages"][-5:-1], "query": user_input}
    ).product_index
    if product_index == -1:
        if "product_index" in state and state["product_index"] is not None:
            product_index = state["product_index"]
    return {"product_index": product_index}

def product_list_reference(state: OverallState) -> OverallState:
    user_input = state["messages"][-1]
    # last 3 messages are the product details
    product_indices = product_reference_list_chain.invoke(
        {"history": state["messages"][-5:-1], "query": user_input}
    ).product_references
    if product_indices == []:
        if "product_indices" in state and state["product_indices"] is not None:
            product_indices = state["product_indices"]
    return {"product_indices": product_indices}

def information_retrieval(state: OverallState) -> MessagesState:
    query = state["messages"][-1].content
    product_id = state["product_ids"][state["product_index"]]
    response = explain_product(query, product_id)
    return {"messages": response}

def reviews(state: OverallState) -> MessagesState:
    query = state["messages"][-1].content
    product_id = state["product_ids"][state["product_index"]]
    response = explain_reviews(query, product_id)
    return {"messages": response}

def comparison(state: OverallState) -> MessagesState:
    query = state["messages"][-1].content
    product_indices = state["product_indices"]
    product_ids = [state["product_ids"][index] for index in product_indices]
    response = compare_products(query, product_ids)
    return {"messages": response}


# Graph
builder = StateGraph(OverallState, input=MessagesState, output=MessagesState)

# Define nodes: these do the work
builder.add_node("intent_router", intent_router)
builder.add_node("entity_identification", entity_identification)
builder.add_node("product_reference", product_reference)
builder.add_node("product_list_reference", product_list_reference)
builder.add_node("greetings", hello)
builder.add_node("product_search", product_search)
builder.add_node("information_retrieval", information_retrieval)
builder.add_node("reviews", reviews)
builder.add_node("comparison", comparison)
builder.add_node("recommendation", recommendation)
builder.add_node("bye", bye)

# Define edges: these determine how the control flow moves
builder.add_edge(START, "intent_router")
# edge from intent_router to each intent node
intent_dict = {
    "greetings": "greetings",
    "product_search": "entity_identification",
    "information_retrieval": "product_reference",
    "reviews": "product_reference",
    "comparison": "product_list_reference",
    "recommendation": "entity_identification",
    "bye": "bye",
    "noclass": "greetings"
}
builder.add_conditional_edges("intent_router", lambda state: state["intent"], intent_dict)
builder.add_conditional_edges("entity_identification", lambda state: state["intent"], ["product_search", "recommendation"])
builder.add_conditional_edges("product_reference", lambda state: state["intent"], ["information_retrieval", "reviews"])
builder.add_conditional_edges("product_list_reference", lambda state: state["intent"], ["comparison"])
builder.add_edge("greetings", END)
builder.add_edge("product_search", END)
builder.add_edge("information_retrieval", END)
builder.add_edge("reviews", END)
builder.add_edge("comparison", END)
builder.add_edge("recommendation", END)
builder.add_edge("bye", END)

# Writes the state in every step of the graph
memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)