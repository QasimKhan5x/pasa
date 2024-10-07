from typing import List

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

from utils2 import llm_precise, graphdb

prompt_template = PromptTemplate(
    template="""
You are a product assistant helping users find information about products you have previously shown them. 
The conversation history contains the details of your interaction with the user, including a list of products you presented.

The user may refer to a product either by its title or by its position in the list (e.g., "the first product", "second item"). 
Your task is to determine which product the user is referring to in their query and return its corresponding integer index in the list of products you provided earlier.

**Instructions:**
- Analyze the conversation history to identify the list of products displayed to the user.
- Interpret the user's query to determine which product they are asking about.
- Return a single integer representing the 0-based index of the referred product in the original list.

<history>{history}</history>
<query>{query}</query>
""".strip(),
    input_variables=["history", "query"],
)


class ProductReference(BaseModel):
    product_index: int = Field(
        ...,
        title="Product Index",
        description="The integer index of the product the user is referring to based on the query. If none found, return -1.",
        examples=[1, 2, 3, -1],
    )

    @field_validator("product_index")
    @classmethod
    def validate_product_index(cls, value):
        if value < -1:
            raise ValueError("Product index must be >= -1")
        return value


product_reference_chain = prompt_template | llm_precise.with_structured_output(
    ProductReference
)


class ProductReferenceList(BaseModel):
    product_references: List[int] = Field(
        title="Product References",
        description="A list of product references based on a user query. If none found, return an empty list.",
        examples=[[0, 2], [2, 3], [0, 1, 2], []],
    )

    def __iter__(self):
        return iter(self.product_references)

    def __len__(self):
        return len(self.product_references)

    def __getitem__(self, index):
        return self.product_references[index]


prompt_template = PromptTemplate(
    template="""
You are a product assistant. The conversation history contains details about multiple products shown to the user. 
The user can refer to a product by mentioning the product title directly or by referring to the position of the product.
Based on the user's query, return the integer indices of the products they are referring to.
Return a list of atleast two or more integers representing the index of the products.

<history>{history}</history>
<query>{query}</query>
""".strip(),
    input_variables=["history", "query"],
)
product_reference_list_chain = prompt_template | llm_precise.with_structured_output(
    ProductReferenceList
)



def get_product_explanation(product_id):
    product = graphdb.run_query(f"MATCH (p:Product) WHERE p.product_id = '{product_id}' RETURN p")[0]['p']
    title = product["title"]
    rating_info = f"Rating: {product['average_rating']}/5 from {product['rating_number']} reviews"
    features = product["features"]
    description = product["description"]
    attribute_nodes = graphdb.run_query(f"MATCH (a:Attribute)<-[:HAS_ATTRIBUTE]-(p:Product) WHERE p.product_id = '{product_id}' RETURN a")
    attributes = [node["a"] for node in attribute_nodes]
    attributes = "\n".join([f"{attr['name']}: {attr['value']}" for attr in attributes])
    total_description = (
        f"{title}\n{rating_info}\n{features}\n{description}\n{attributes}"
    )
    return total_description


def explain_product(query, product_id):
    total_description = get_product_explanation(product_id)
    response = llm_precise(
        f"Answer the user query based on the product details provided.\n{total_description}\n{query}"
    )
    return response


def explain_reviews(query, product_id):
    product_reviews = graphdb.run_query(
        f"MATCH (p:Product {{product_id: '{product_id}'}})<-[:REVIEWS]-(r:Review) RETURN r.title as title, r.rating as rating, r.text as text"
    )
    formatted_reviews = []
    for review in product_reviews:
        formatted_reviews.append(
            f"{review['title']}\nRating: {review['rating']}\n{review['text']}"
        )
    reviews_str = "\n\n".join(formatted_reviews)
    response = llm_precise.invoke(
        f"Answer the user query based on the product reviews provided.\n{reviews_str}\n{query}"
    )
    return response


def compare_products(query, product_ids):
    product_explanations = [
        get_product_explanation(product_id) for product_id in product_ids
    ]
    product_descriptions = "\n\n".join(product_explanations)
    response = llm_precise(
        f"Compare the products based on the details provided and answer the user query. Format your answer as a Markdown table.\n{product_descriptions}\n{query}"
    )
    return response
