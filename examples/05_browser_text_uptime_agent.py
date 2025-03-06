"""
This example demonstrates how to use the @browser-text tool to fetch and analyze
uptime data from API status pages. It shows how to:
1. Use @browser-text to fetch status page content
2. Extract uptime percentage for specific services
3. Handle different status page formats
"""

import asyncio

from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]

import workflowai
from workflowai import Model


class UptimeInput(BaseModel):
    """Input for checking API uptime."""
    status_page_url: str = Field(
        description="URL of the status page to check",
        examples=["https://status.openai.com", "https://status.anthropic.com"],
    )
    service_name: str = Field(
        description="Name of the specific API service to check",
        examples=["API", "Chat Completions", "Embeddings"],
    )


class UptimeOutput(BaseModel):
    """Output containing uptime percentage."""
    uptime_percentage: float = Field(
        description="The uptime percentage for the specified service over the last 30 days",
        examples=[99.99, 98.5],
        ge=0.0,
        le=100.0,
    )


@workflowai.agent(
    id="uptime-checker",
    model=Model.GPT_4O_MINI_LATEST,
)
async def check_uptime(uptime_input: UptimeInput, use_cache: str = "never") -> UptimeOutput:
    """
    Fetch and analyze uptime data from an API status page.
    Use @browser-text to get the page content.

    Guidelines:
    1. Visit the provided status page URL
    2. Find the specified service's uptime information
    3. Extract the most recent uptime percentage available
    4. Return just the percentage as a number between 0 and 100

    Focus on finding the most recent uptime data available.
    This could be daily, weekly, monthly or any other time period shown.
    """
    ...


async def main():
    # Example: Check OpenAI API uptime
    uptime_input = UptimeInput(
        status_page_url="https://status.openai.com",
        service_name="API",
    )

    print(f"\nChecking uptime for {uptime_input.service_name} at {uptime_input.status_page_url}...")
    print("-" * 50)

    # Get uptime data with caching disabled
    run = await check_uptime.run(uptime_input, use_cache="never")

    # Print the run
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
