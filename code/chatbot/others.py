from typing import List
from ast import literal_eval

from langchain_core.prompts import PromptTemplate
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from utils2 import llm_precise, graphdb

prompt_template = PromptTemplate(
    template="""
You are a product assistant. The conversation history contains details about multiple products shown to the user. The user can refer to a product by mentioning the product title directly or by referring to the position of the product.
Based on the user's query, return the integer index of the product they are referring to.
Just return a single integer representing the index of the product.

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


def load_products(path):
    products_df = pd.read_csv(path)
    products_df["categories"] = products_df["categories"].apply(literal_eval)
    products_df["details"] = products_df["details"].apply(literal_eval)
    products_df["description"] = products_df["description"].apply(literal_eval)
    products_df["features"] = products_df["features"].apply(literal_eval)
    return products_df


products = load_products("/project/data/products_0.001.csv")


def get_product_explanation(product_id):
    product = products[products["parent_asin"] == product_id]
    title = product["title"].values[0]
    rating_info = f"Rating: {product['average_rating'].values[0]}/5 from {product['rating_number'].values[0]} reviews"
    features = "\n".join(product["features"].values[0])
    description = "\n".join(product["description"].values[0])
    attributes = ";".join({f"{k}={v}" for k, v in product["details"].items()})
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
    response = llm_precise(
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
