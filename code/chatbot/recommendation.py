from langgraph.graph import MessagesState
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from utils2 import (
    OverallState,
    subcategory_searcher,
    usecase_searcher,
    rerank,
    keyword_searcher,
    retrieve_and_rerank,
    summary_searcher,
    format_product_ranking_list,
    graphdb,
    ProductRankingList,
    llm_precise,
)

recommendation_template = PromptTemplate(
    template="""
I want to find some products for my query: "{query}".
I have shortlisted some products from my initial search.
<products>{products}</products>
Please tell me whether each product is a good match for my query and a short explanation for your answer based on my query.
Return your results as a ranked list with the product ID, whether to keep the product, and the explanation. Ensure the explanation is concise (2-3 sentences max).
Good Explanation: This product is a facial serum that is a good option for sensitive skin and cruelty-free. It addresses common skincare concerns like redness and blemishes.
Bad Explanation: This product is a facial serum that is under $50, making it a good match for the query. It is designed to calm breakouts, heal blemishes, and reduce redness, which are common concerns for facial skin. The product is also fragrance-free and cruelty-free, which may be a plus for some users. It is suitable for all skin types, including oily and sensitive skin, which makes it a versatile option. It is made in the USA with high concentrations of active ingredients and minimal fillers or chemicals. Therefore, it is a good candidate to keep in the list of potential products.
**Important**: A concise explanation improves response efficiency and clarity.
""".strip(),
    input_variables=["query", "products"],
)
recommender_llm = llm_precise.with_structured_output(ProductRankingList)
recommendation_chain = recommendation_template | recommender_llm


def find_matching_products(
    subcategories, usecases, keywords, price_range=None, debug=False
):
    # Build the initial Cypher query to expand subcategories based on use cases
    cypher_query = f"""// Step 1: Expand the list of subcategories using the UseCase nodes
MATCH (u:UseCase)-[:USED_FOR]->(s:Subcategory)
WHERE u.title IN {usecases}
WITH COLLECT(DISTINCT s.name) + {subcategories} AS expanded_subcategories
// Step 2: Find products using the expanded list of subcategories and the keyword clause
MATCH (p:Product)-[:BELONGS_TO]->(s:Subcategory)
WHERE s.name IN expanded_subcategories
OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
WHERE k.name IN {keywords}
// Aggregation to count matches
WITH p, 
    COUNT(DISTINCT k) AS keyword_matches, 
    COUNT(DISTINCT s) AS subcategory_matches
    """

    # Add price range filter if provided
    if price_range:
        if "lt" in price_range:
            cypher_query += f" WHERE p.price < {price_range['lt']} "
        elif "around" in price_range:
            price = price_range["around"]
            cypher_query += f"""
            MATCH (p)-[:AROUND_PRICE]->(pr:PriceRange)
            WHERE {price} >= pr.lower_limit AND {price} <= pr.upper_limit
            """

    # Continue with the RETURN and ORDER BY clauses for ranking products
    cypher_query += f"""
RETURN p.product_id AS product_id, 
    keyword_matches, 
    subcategory_matches,
    (keyword_matches * 3 + subcategory_matches * 2) AS score
ORDER BY score DESC, keyword_matches DESC, subcategory_matches DESC
    """.strip()

    if debug:
        print(cypher_query)

    products = graphdb.run_query(cypher_query)
    return products


def format_product_details(product_ids):
    required_products = graphdb.run_query(
        f"MATCH (p:Product) WHERE p.product_id IN {product_ids} RETURN p.product_id AS product_id, p.summary AS summary"
    )
    formatted_products = [
        f"{product['product_id']}\n{product['summary']}"
        for product in required_products
    ]
    return "\n\n".join(formatted_products)


def recommendation(state: OverallState) -> MessagesState:
    query = state["messages"][-1].content
    entities = state["entities"]
    if "attributes" in entities and entities["attributes"]:
        for attribute in entities["attributes"]:
            entities["keywords"].append(
                f"{attribute}:{entities['attributes'][attribute]}"
            )

    price_range = None
    if "price_range" in entities and entities["price_range"]:
        price_range = entities["price_range"]

    included_categories = subcategory_searcher.search(
        entities["category"], limit=2, threshold=0.9
    )
    included_categories = [document["document"] for document in included_categories]

    included_usecases = usecase_searcher.search(query, threshold=0.9, limit=20)
    included_usecases = [document["document"] for document in included_usecases]
    included_usecases_reranked = rerank(
        query, included_usecases, limit=5, model="jina-reranker-v2-base-multilingual"
    )
    included_usecases_reranked = [
        document["document"]["text"] for document in included_usecases_reranked
    ]

    expanded_keywords = set()
    for keyword in entities["keywords"]:
        documents = keyword_searcher.search(keyword, threshold=0.9, limit=5)
        new_keywords = [document["document"] for document in documents]
        expanded_keywords.update(new_keywords)
    expanded_keywords = list(expanded_keywords)

    matching_products = find_matching_products(
        included_categories,
        included_usecases_reranked,
        expanded_keywords,
        price_range=price_range,
        debug=False,
    )
    min_score = min({entry["score"] for entry in matching_products})
    relevant_products = [
        entry["product_id"] for entry in matching_products if entry["score"] > min_score
    ]
    num_products_to_consider = 10
    relevant_products = relevant_products[:num_products_to_consider]

    other_products = []
    num_additional_products_required = max(
        num_products_to_consider - len(relevant_products), 0
    )

    additional_product_ids = []
    if num_additional_products_required != 0:
        other_products = [
            entry["product_id"]
            for entry in matching_products
            if entry["score"] == min_score
        ]
        additional_product_ids = retrieve_and_rerank(
            query,
            limit_rerank=num_additional_products_required,
            product_ids=other_products,
            searcher=summary_searcher,
        )

    final_product_ids = relevant_products + additional_product_ids
    product_details = format_product_details(final_product_ids)
    product_ranking = recommendation_chain.invoke(
        {"query": query, "products": product_details}
    )
    output_message = format_product_ranking_list(product_ranking)

    return {"product_ids": final_product_ids, "messages": AIMessage(output_message)}
