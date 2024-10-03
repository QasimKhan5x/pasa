import os

import requests
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from dotenv import load_dotenv, find_dotenv

from Neo4jConnection import Neo4jConnection

load_dotenv(find_dotenv())

# llm_precise = ChatNVIDIA(
#     model="meta/llama-3.1-70b-instruct",
#     api_key=os.getenv("NVIDIA_API_KEY"),
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=2048,
#     streaming=True,
# )

llm_precise = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
    top_p=0.7,
    max_tokens=2048,
    streaming=True,
)


graphdb = Neo4jConnection(
    uri=os.environ["NEO4J_URI"], user="neo4j", password=os.environ["NEO4J_PASSWORD"], db="neo4j"
)


class OverallState(MessagesState):
    intent: Literal[
        "greetings",
        "product_search",
        "information_retrieval",
        "reviews_ratings",
        "comparison",
        "recommendation",
        "noclass",
        "bye",
    ]
    entities: Optional[Dict[str, Any]]
    product_ids: Optional[List[str]]
    product_index: Optional[int]
    product_indices: Optional[List[int]]


class ProductRanking(BaseModel):
    product_id: str = Field(
        title="Product ID",
        description="The unique identifier for the product.",
        examples=["B07H8QMZWV"],
    )

    keep: bool = Field(
        title="Keep Product",
        description="A boolean value indicating whether the product should be kept or discarded based on the explanation provided.",
        examples=[True, False],
    )

    explanation: str = Field(
        title="Ranking Explanation",
        description="An explanation for the ranking of the product based on user criteria.",
        examples=[
            "This product is a great match for your query because it contains epsom salt, which is a key ingredient for muscle relaxation."
        ],
    )


class ProductRankingList(BaseModel):
    rankings: List[ProductRanking] = Field(
        title="Product Rankings",
        description="A list of product rankings with corresponding explanations.",
    )

    def __iter__(self):
        return iter(self.rankings)

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, index):
        return self.rankings[index]


def create_amazon_link(product_id):
    return f"https://www.amazon.com/dp/{product_id}"


def format_product_ranking_list(ranking) -> str:
    output_message = []
    for product in ranking:
        if product.keep:
            product_id = product.product_id
            explanation = product.explanation
            product_title = graphdb.run_query(
                f"MATCH (p:Product) WHERE p.product_id = '{product_id}' RETURN p.title"
            )[0]["p.title"]
            output_message.append(
                f"[{product_title}]({create_amazon_link(product_id)}): {explanation}"
            )
    output_message = "\n\n".join(output_message)
    return output_message


def rerank(query, documents, limit=5, model="jina-colbert-v2"):
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("JINA_API_KEY")}',
    }
    data = {"model": model, "query": query, "top_n": limit, "documents": documents}

    response = requests.post(url, headers=headers, json=data)
    data = response.json()["results"]
    return data


def retrieve_and_rerank(query, limit_rerank, product_ids, searcher, limit_retrieve=20):
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="product_id",
                match=models.MatchAny(any=product_ids),
            )
        ]
    )

    retrieved_documents = searcher.search(
        query, query_filter=query_filter, limit=limit_retrieve, threshold=0.9
    )
    documents = [document["document"] for document in retrieved_documents]
    sorted_product_ids = [document["product_id"] for document in retrieved_documents]

    reranked_documents = rerank(
        query, documents, limit_rerank, model="jina-reranker-v2-base-multilingual"
    )
    reranked_product_ids = [
        sorted_product_ids[entry["index"]] for entry in reranked_documents
    ]
    return reranked_product_ids


class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        self.qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2")
        self.qdrant_client.set_sparse_model("prithivida/Splade_PP_en_v1")

        vector_params = self.qdrant_client.get_fastembed_vector_params()
        vector_params["fast-all-minilm-l6-v2"].on_disk = True
        sparse_vector_params = self.qdrant_client.get_fastembed_sparse_vector_params()
        sparse_vector_params["fast-sparse-splade_pp_en_v1"].index.on_disk = True

    def search(self, text: str, query_filter=None, limit=3, threshold=0.5):
        # `search_result` contains found vector ids with similarity scores
        # along with the stored payload (metadata)
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold,
        )

        # Select and return metadata
        metadata = [hit.metadata for hit in search_result]
        return metadata


subcategory_searcher = HybridSearcher(collection_name="subcategories")
summary_searcher = HybridSearcher(collection_name="summaries")
usecase_searcher = HybridSearcher(collection_name="usecases")
keyword_searcher = HybridSearcher(collection_name="keywords")
subcategory_searcher = HybridSearcher(collection_name="subcategories")
