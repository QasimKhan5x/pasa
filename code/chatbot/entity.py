from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from utils2 import llm_precise

class Product(BaseModel):
    category: str = Field(
        ...,
        title="Product Category",
        description="The category of the product (head term).",
        examples=["moisturizer", "shampoo", "sunscreen"]
    )
    
    attributes: Optional[Dict[str, Any]] = Field(
        None,
        title="Product Attributes",
        description="A dictionary of product attributes, such as SPF, vegan, etc.",
        examples=[{"SPF": 30, "vegan": True}]
    )
    
    price_range: Optional[Dict[str, int]] = Field(
        None,
        title="Price Range",
        description=(
            "The price range of the product. Acceptable keys are 'lt' for less than or 'around' for approximate price."
        ),
        examples=[{"lt": 50}, {"around": 30}]
    )
    
    keywords: List[str] = Field(
        ...,
        title="Product Keywords",
        description="A list of keywords associated with the product, excluding the head term.",
        examples=[["hydrating", "SPF", "waterproof"]]
    )


# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Product)

prompt = PromptTemplate(
    template="Parse the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

entity_identification_chain = prompt | llm_precise | parser