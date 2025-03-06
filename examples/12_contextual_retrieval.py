"""
This example demonstrates how to implement Contextual Retrieval's context generation
as described by Anthropic (https://www.anthropic.com/news/contextual-retrieval).
It shows how to generate concise, contextual descriptions for document chunks.
"""

import asyncio

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class ContextGeneratorInput(BaseModel):
    """Input for generating context for a document chunk."""
    doc_content: str = Field(
        description="The full text content of the document",
    )
    chunk_content: str = Field(
        description="The specific chunk of text to generate context for",
    )


class ContextGeneratorOutput(BaseModel):
    """Output containing the generated context for a chunk."""
    context: str = Field(
        description="Generated contextual information for the chunk",
        examples=[
            "This chunk is from section 3.2 discussing revenue growth in Q2 2023",
            "This appears in the methodology section explaining the experimental setup",
        ],
    )


@workflowai.agent(
    id="context-generator",
    model=Model.CLAUDE_3_5_SONNET_LATEST,
)
async def generate_chunk_context(context_input: ContextGeneratorInput) -> ContextGeneratorOutput:
    """
    Here is the chunk we want to situate within the whole document.
    Please give a short succinct context to situate this chunk within the overall document
    for the purposes of improving search retrieval of the chunk.
    """
    ...


async def main():
    # Example: Generate context for a document chunk
    print("\nGenerating context for document chunk")
    print("-" * 50)

    # Example document content
    doc_content = """
    ACME Corporation (NASDAQ: ACME)
    Second Quarter 2023 Financial Results

    Executive Summary
    ACME Corp. delivered strong performance in Q2 2023, with revenue growth
    exceeding market expectations. The company's strategic initiatives in
    AI and cloud services continued to drive expansion.

    Financial Highlights
    The company's revenue grew by 3% over the previous quarter, reaching
    $323.4M. This growth was primarily driven by our enterprise segment,
    which saw a 15% increase in cloud service subscriptions.

    Operational Metrics
    - Customer base expanded to 15,000 enterprise clients
    - Cloud platform usage increased by 25%
    - AI solutions adoption rate reached 40%
    """

    # Example chunk from the Financial Highlights section
    chunk_content = "The company's revenue grew by 3% over the previous quarter, reaching $323.4M."

    context_input = ContextGeneratorInput(
        doc_content=doc_content,
        chunk_content=chunk_content,
    )

    run = await generate_chunk_context.run(context_input)
    print("\nGenerated Context:")
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
