"""
This example demonstrates how to create a WorkflowAI agent that extracts flight information from emails.
It showcases:

1. Using Pydantic models for structured data extraction
2. Extracting specific details like flight numbers, dates, and times
"""

import asyncio
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class EmailInput(BaseModel):
    """Raw email content containing flight booking details.
    This could be a confirmation email, itinerary update, or e-ticket from any airline."""
    email_content: str


class FlightInfo(BaseModel):
    """Model for extracted flight information."""
    class Status(str, Enum):
        """Possible statuses for a flight booking."""
        CONFIRMED = "Confirmed"
        PENDING = "Pending"
        CANCELLED = "Cancelled"
        DELAYED = "Delayed"
        COMPLETED = "Completed"

    passenger: str
    airline: str
    flight_number: str
    from_airport: str = Field(description="Three-letter IATA airport code for departure")
    to_airport: str = Field(description="Three-letter IATA airport code for arrival")
    departure: datetime
    arrival: datetime
    status: Status

@workflowai.agent(
    id="flight-info-extractor",
    model=Model.GEMINI_2_0_FLASH_LATEST,
)
async def extract_flight_info(email_input: EmailInput) -> FlightInfo:
    """
    Extract flight information from an email containing booking details.
    """
    ...


async def main():
    email = """
    Dear Jane Smith,

    Your flight booking has been confirmed. Here are your flight details:

    Flight: UA789
    From: SFO
    To: JFK
    Departure: 2024-03-25 9:00 AM
    Arrival: 2024-03-25 5:15 PM
    Booking Reference: XYZ789

    Total Journey Time: 8 hours 15 minutes
    Status: Confirmed

    Thank you for choosing United Airlines!
    """
    run = await extract_flight_info.run(EmailInput(email_content=email))
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
