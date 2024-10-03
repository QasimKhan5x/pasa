from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from utils2 import (
    ProductRankingList,
    subcategory_searcher,
    llm_precise,
    graphdb,
    OverallState,
    summary_searcher,
    retrieve_and_rerank,
    format_product_ranking_list,
)

template = """
Below is a list of products, with each product containing formatted details such as attributes and keywords. 
Please help me analyze each product and rank them based on how well they match my query. 
Additionally, you will provide a brief, conversational explanation for why each product is ranked where it is and how well it meets my needs. 
Keep your responses concise.

Return the results as a list of tuples where the first element is the product id, and the second element is a brief explanation for the ranking.

<user>{query}</user>
<products>{product_details}</products>
""".strip()

product_search_ranking_template = PromptTemplate(
    template=template, input_variables=["product_details", "query"]
)

product_search_llm = llm_precise.with_structured_output(ProductRankingList)
product_search_chain = product_search_ranking_template | product_search_llm


def create_user_query(entities):
    attributes = entities.get("attributes", {})
    if attributes:
        attribute_str = ";".join([f"{attr}={attributes[attr]}" for attr in attributes])
    else:
        attribute_str = ""
    query = f"I'm looking for a {entities['category']}"
    if attribute_str:
        query += f" with {attribute_str}"
    if entities.get("keywords") and len(entities["keywords"]) > 0:
        query += f" that is {', '.join(entities['keywords'])}"
    return query


def format_product_details(product_dict):
    product_id = product_dict["product_id"]

    # Format attributes
    attributes = product_dict.get("attributes", [])
    attributes_str = ";".join(
        [f"{attr['attribute_name']}={attr['attribute_value']}" for attr in attributes]
    )

    # Format keywords
    keywords = product_dict.get("keywords", [])
    keywords_str = ", ".join(keywords)

    # Create the final formatted string
    formatted_details = f"product_id: {product_id}\n"
    formatted_details += f"attributes: {attributes_str}\n"
    formatted_details += f"keywords: {keywords_str}"

    return formatted_details


def create_product_details(final_products):
    product_details = []
    for product in final_products:
        formatted_details = format_product_details(product)
        product_details.append(formatted_details)
    return "\n\n".join(product_details)


def get_products_in_subcategories(included_categories, price_range=None, debug=False):
    # Convert the list of categories into a comma-separated string for the query
    categories_str = ", ".join(
        f"\"{category['document']}\"" for category in included_categories
    )

    # Construct the Cypher query
    cypher_query = "MATCH (p:Product)-[:BELONGS_TO]->(sc:Subcategory)"

    if price_range:
        if "lt" in price_range:
            cypher_query += f"WHERE p.price < {price_range['lt']} "
        elif "around" in price_range:
            price = price_range["around"]
            cypher_query += f""",\n (p)-[:AROUND_PRICE]->(pr:PriceRange)
    WHERE {price} >= pr.lower_limit AND {price} <= pr.upper_limit """
        cypher_query += f"AND sc.name IN [{categories_str}] RETURN p.product_id"
    else:
        cypher_query += f"WHERE sc.name IN [{categories_str}] RETURN p.product_id"
    # Run the query and return the results
    if debug:
        print(cypher_query)
    results = graphdb.run_query(cypher_query)
    return [record["p.product_id"] for record in results]


def collect_attributes_and_keywords_for_products(product_ids):
    cypher_query = f"""
    MATCH (p:Product)
    WHERE p.product_id IN {product_ids}
    OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:Attribute)
    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
    RETURN p.product_id as product_id,
           collect(DISTINCT {{attribute_name: a.name, attribute_value: a.value}}) AS attributes, 
           collect(DISTINCT k.name) AS keywords
    """.strip()
    results = graphdb.run_query(cypher_query)
    return results


def product_search(state: OverallState) -> MessagesState:
    query = state["messages"][-1].content
    entities = state["entities"]
    included_categories = subcategory_searcher.search(
        entities["category"], threshold=0.9
    )
    product_ids = get_products_in_subcategories(
        included_categories, price_range=entities.get("price_range")
    )
    returned_product_ids = retrieve_and_rerank(
        query=query,
        limit_rerank=10,
        product_ids=product_ids,
        searcher=summary_searcher,
        limit_retrieve=20,
    )
    pid_attr_keywords = collect_attributes_and_keywords_for_products(
        returned_product_ids
    )
    product_details = create_product_details(pid_attr_keywords)
    response = product_search_chain.invoke(
        {"query": create_user_query(entities), "product_details": product_details}
    )
    output_message = format_product_ranking_list(response)
    return {"product_ids": returned_product_ids, "messages": AIMessage(output_message)}
