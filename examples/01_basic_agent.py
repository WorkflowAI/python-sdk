"""
This example demonstrates how to create a basic WorkflowAI agent that takes a city name
and returns information about the capital of its country. It showcases:

1. Basic agent creation with input/output models
2. Field descriptions and examples
3. Cost and latency tracking
4. How to fetch and analyze completions after a run
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model, Run


class CityInput(BaseModel):
    """Input model for the city-to-capital agent."""

    # For simple input fields like 'city', descriptions and examples add token overhead
    # without providing additional context that a modern LLM wouldn't already understand.
    # Input fields never need examples since an actual value will be provided at runtime.
    city: str = Field()


class CapitalOutput(BaseModel):
    """Output model containing information about the capital city."""

    # Fields like country, capital, etc. are self-explanatory to LLMs
    # Omitting descriptions and examples for these would reduce token usage
    country: str = Field(
        description="The country where the input city is located",
        examples=["France", "United States", "Japan"],
    )
    capital: str = Field(
        description="The capital city of the country",
        examples=["Paris", "Washington D.C.", "Tokyo"],
    )
    fun_fact: str = Field(
        description="An interesting fact about the capital city",
        examples=["Paris has been the capital of France since 508 CE"],
    )


@workflowai.agent(
    id="city-to-capital",
    model=Model.CLAUDE_3_7_SONNET_LATEST,
)
async def get_capital_info(city_input: CityInput) -> Run[CapitalOutput]:
    """
    Find the capital city of the country where the input city is located.

    Guidelines:
    1. First identify the country where the input city is located
    2. Then provide the capital city of that country
    3. Include an interesting historical or cultural fact about the capital
    4. Be accurate and precise with geographical information
    5. If the input city is itself the capital, still provide the information
    """
    ...


async def display_completions(run: Run[Any]) -> None:
    """Helper function to display completions for a run."""
    try:
        completions = await run.fetch_completions()

        for completion in completions:
            print("\n--- Completion Details ---")

            # Use model_dump_json for clean serialization
            completion_json = completion.model_dump_json(indent=2)
            print(completion_json)
    except (ValueError, workflowai.WorkflowAIError) as e:
        print(f"Error: {e}")


async def main():
    # Example 1: Basic usage with Paris
    print("\nExample 1: Basic usage with Paris")
    print("-" * 50)
    try:
        run = await get_capital_info.run(CityInput(city="Paris"))
        print(run)
    except workflowai.WorkflowAIError as e:
        print(e)
        return

    # Example 2: Using Tokyo
    # print("\nExample 2: Using Tokyo")
    # print("-" * 50)
    # run = await get_capital_info.run(CityInput(city="Tokyo"))
    # print(run)

    # Fetch and display completions for the Tokyo example
    await display_completions(run)


if __name__ == "__main__":
    asyncio.run(main())
