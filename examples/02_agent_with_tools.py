"""
This example demonstrates how to create a WorkflowAI agent that uses tools to enhance its capabilities.
It showcases:

1. Using built-in WorkflowAI tools (@search-google)
2. Creating and using custom tools
3. Combining multiple tools in a single agent
"""

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


def get_current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)"""
    return datetime.now(tz=ZoneInfo("UTC")).date().isoformat()


def calculate_days_between(date1: str, date2: str) -> int:
    """Calculate the number of days between two dates in ISO format (YYYY-MM-DD)"""
    d1 = date.fromisoformat(date1)
    d2 = date.fromisoformat(date2)
    return abs((d2 - d1).days)


class HistoricalEventInput(BaseModel):
    """Input model for querying historical events."""
    query: str = Field(
        description="A query about a historical event",
        examples=[
            "When was the first moon landing?",
            "When did World War II end?",
            "When was the Declaration of Independence signed?",
        ],
    )


class HistoricalEventOutput(BaseModel):
    """Output model containing information about a historical event."""
    event_date: str = Field(
        description="The date of the event in ISO format (YYYY-MM-DD)",
        examples=["1969-07-20", "1945-09-02", "1776-07-04"],
    )
    event_description: str = Field(
        description="A brief description of the event",
        examples=[
            "Apollo 11 astronauts Neil Armstrong and Buzz Aldrin became the first humans to land on the Moon",
            "Japan formally surrendered to the Allied Powers aboard the USS Missouri in Tokyo Bay",
        ],
    )
    days_since_event: int = Field(
        description="Number of days between the event and today",
        examples=[19876, 28490, 90123],
    )


@workflowai.agent(
    id="historical-event-analyzer",
    model=Model.GEMINI_2_0_FLASH_LATEST,
    tools=[get_current_date, calculate_days_between],
)
async def analyze_historical_event(event_input: HistoricalEventInput) -> HistoricalEventOutput:
    """
    Find information about historical events and calculate days since they occurred.

    You have access to these tools:
    1. @search-google - Use this to find accurate information about events
    2. get_current_date - Use this to get today's date for calculating days since the event
    3. calculate_days_between - Calculate days between two dates (format: YYYY-MM-DD)

    Guidelines:
    1. Use @search-google to find accurate event information
    2. Use get_current_date to get today's date
    3. Use calculate_days_between to compute days since the event
    4. Return dates in ISO format (YYYY-MM-DD)
    5. Be precise with dates and descriptions
    """
    ...


async def main():
    # Example: Query about the moon landing
    print("\nExample: First Moon Landing")
    print("-" * 50)
    run = await analyze_historical_event.run(
        HistoricalEventInput(query="When was the first moon landing?"),
    )
    print(run)

    # Example: Query about World War II
    print("\nExample: End of World War II")
    print("-" * 50)
    run = await analyze_historical_event.run(
        HistoricalEventInput(query="When did World War II end?"),
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
