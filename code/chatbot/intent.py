import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from utils2 import llm_precise

class MessageClassification(BaseModel):
    
    category: Literal[
        "greetings",
        "product_search",
        "information_retrieval",
        "reviews",
        "comparison",
        "recommendation",
        "noclass",
        "bye"
    ] = Field(title="category", description="a set of valid categories representing user intents based on the message")

    @field_validator('category')
    @classmethod
    def validate_category(cls, value):
        valid_categories = [
            "greetings",
            "product_search",
            "information_retrieval",
            "reviews",
            "comparison",
            "recommendation",
            "noclass",
            "bye"
        ]
        if value not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return value
    
examples_path = "/project/data/examples-intent-classification.json"
with open(examples_path) as f:
    intent_examples = json.load(f)

prompt_prefix = """
<context>
You are an AI assistant designed to classify user messages into one of six categories based on their content.
</context
<classes>
Greetings: The user initiates the conversation with a greeting or seeks general assistance.
    Indicators:
        General salutations or questions about how you can help.
        Unrelated to any product and no specific information requested.
Product Search: The user wants to find specific products based on detailed criteria or filters.
    Indicators:
        Mentions of specific attributes of the product
        Direct requests to see products that meet certain specifications.
Information Retrieval: The user seeks detailed information about a particular product.
    Indicators:
        Questions about ingredients, features, or specifics of a product.
        Inquiries that require factual data or descriptions.
Reviews and Ratings: The user asks about customer feedback, reviews, or ratings of a product.
    Indicators:
        Requests for opinions, ratings, or what others think about a product.
        Interest in the product's reputation or user satisfaction.
Comparison: The user wants to compare multiple products or find alternatives.
    Indicators:
        Questions that involve comparing features, prices, or effectiveness.
        Seeking substitutes or similar products.
Recommendation: The user seeks personalized suggestions or explores broad product categories.
    Indicators:
        Open-ended requests for advice or suggestions.
        Attributes are not mentioned.
        Interest in popular, new, or suitable products without specific filters.
        Queries about gifts or products for special occasions.
</classes>
<examples>
""".strip()

prompt_suffix = """</examples>
<instructions>
Task: Classify the user's message into one of the six categories.
How to Classify:
    Read the user's message carefully.
    Identify intent based on the descriptions and indicators.
    Match the message to the most appropriate category.
Output Format: <output>category_name</output>
If No Match: return <output>noclass</output>.
</instructions>

<input>{input}</input>
"""

example_prompt_template = PromptTemplate(
    template="""
<input>{input}</input>
<output>{output}</output>
""".strip(),
    input_variables=["input"],
    output_variables=["output"]
) 

few_shot_template = FewShotPromptTemplate(
    examples=intent_examples,
    example_prompt=example_prompt_template,
    prefix=prompt_prefix,
    suffix=prompt_suffix,
)
intent_classifier = few_shot_template | llm_precise

async def async_get_intent(user_input: str) -> str:
    output = intent_classifier.invoke(user_input).content
    category = re.search(r"<output>(.*?)</output>", output).group(1)
    return MessageClassification(category=category).category

def get_intent(user_input: str) -> str:
    output = intent_classifier.invoke(user_input).content
    category = re.search(r"<output>(.*?)</output>", output).group(1)
    return MessageClassification(category=category).category