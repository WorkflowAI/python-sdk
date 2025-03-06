"""
This example demonstrates how to create an agent that extracts calendar event information
from email content or images. It shows how to:
1. Parse email content to identify event details
2. Extract structured calendar information (date, time, location)
3. Handle different email formats and writing styles
4. Process event information from images (posters, flyers)
5. Return a well-structured calendar event object
"""

import asyncio
import base64
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model
from workflowai.fields import File


class EmailInput(BaseModel):
    """Input model containing the email content to analyze."""
    subject: str = Field(
        description="The subject line of the email",
    )
    body: str = Field(
        description="The main content/body of the email",
    )


class ImageInput(BaseModel):
    """Input model containing an event poster or flyer image to analyze."""
    image: File = Field(
        description="An image of an event poster or flyer to analyze",
    )


class CalendarEvent(BaseModel):
    """Model containing the details of a calendar event."""
    title: Optional[str] = Field(
        default=None,
        description="The title/subject of the event",
        examples=["Team Meeting", "Product Review"],
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="The start time of the event in ISO format",
        examples=["2024-04-15T14:00:00-07:00"],
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="The end time of the event if specified",
        examples=["2024-04-15T15:00:00-07:00"],
    )
    location: Optional[str] = Field(
        default=None,
        description="The location or meeting room for the event",
        examples=["Conference Room A", "https://zoom.us/j/123456789"],
    )
    description: Optional[str] = Field(
        default=None,
        description="Additional details or description of the event",
        examples=["Monthly product review meeting to discuss Q2 progress"],
    )


class CalendarEventOutput(BaseModel):
    """Output model containing the extracted calendar event details if found."""
    calendar_event: Optional[CalendarEvent] = Field(
        default=None,
        description="The extracted calendar event details, or null if no event was found",
    )


@workflowai.agent(
    id="calendar-event-extractor",
    model=Model.GPT_4O_MINI_LATEST,
)
async def extract_calendar_event_from_email(email_input: EmailInput) -> CalendarEventOutput:
    """
    Extract calendar event details from email content.

    Guidelines:
    1. Analyze both subject and body to identify event information:
       - Look for date and time information in both fields
       - Extract location and description if available
       - Combine information from both fields when needed
       - Return null for calendar_event if no event information is found

    2. Handle different date/time formats:
       - Explicit dates ("April 15th, 2024")
       - Relative dates ("next Tuesday")
       - Various time formats ("3pm", "15:00", "3:00 PM PST")

    3. Make reasonable assumptions when information is ambiguous:
       - Default to 1 hour duration if end time not specified
       - Use UTC if timezone not specified
       - Extract location if mentioned

    4. Maintain accuracy and completeness:
       - Include all clearly stated information
       - Mark information as optional if not explicitly mentioned
       - Preserve timezone information when available
    """
    ...


@workflowai.agent(
    id="calendar-event-extractor",
    model=Model.GPT_4O_MINI_LATEST,
)
async def extract_calendar_event_from_image(image_input: ImageInput) -> CalendarEventOutput:
    """
    Extract calendar event details from an event poster or flyer image.

    Guidelines:
    1. Analyze the image content to identify event details:
       - Look for prominent text showing date and time
       - Identify event title and location
       - Extract any additional description or details
       - Return null for calendar_event if no event information is found

    2. Handle different date/time formats:
       - Explicit dates ("April 15th, 2024")
       - Various time formats ("3pm", "15:00", "3:00 PM PST")
       - Date ranges for multi-day events

    3. Make reasonable assumptions when information is ambiguous:
       - Default to 1 hour duration if end time not specified
       - Use UTC if timezone not specified
       - Extract location if mentioned

    4. Maintain accuracy and completeness:
       - Include all clearly visible information
       - Mark information as optional if not clearly visible
       - Preserve timezone information when available
    """
    ...


async def main():
    # Example 1: Basic meeting invitation
    print("\nExample 1: Basic meeting invitation")
    print("-" * 50)

    email1 = EmailInput(
        subject="Team Sync - Tomorrow at 2pm",
        body="""
        Hi team,

        Let's sync up tomorrow at 2pm in Conference Room A to discuss our Q2 roadmap.

        Please come prepared with your project updates.

        Best,
        Sarah
        """,
    )

    run = await extract_calendar_event_from_email.run(email1)
    print(run)

    # Example 2: Virtual meeting with more details
    print("\nExample 2: Virtual meeting with more details")
    print("-" * 50)

    email2 = EmailInput(
        subject="Invitation: Product Review Meeting (Apr 15, 2024)",
        body="""
        You are invited to our monthly product review meeting.

        Date: April 15th, 2024
        Time: 2:00 PM - 3:30 PM PST
        Location: https://zoom.us/j/123456789

        Agenda:
        1. Q1 metrics review
        2. Feature roadmap updates
        3. Customer feedback discussion
        """,
    )

    run = await extract_calendar_event_from_email(email2)
    print(run)

    # Example 3: Event poster image
    print("\nExample 3: Event poster image")
    print("-" * 50)

    # Load the image file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "assets", "poster.jpg")

    # Verify the file exists
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        return

    # Read and encode the image
    # Note: Using sync file read in example code is acceptable.
    # We avoid adding aiofiles dependency just for this example.
    with open(image_path, "rb") as f:  # noqa: ASYNC230
        image_data = f.read()

    # Create input with image
    image_input = ImageInput(
        image=File(
            content_type="image/jpeg",
            data=base64.b64encode(image_data).decode(),
        ),
    )

    run = await extract_calendar_event_from_image(image_input)
    print(run)

    # Example 4: Email without calendar event
    print("\nExample 4: Email without calendar event")
    print("-" * 50)

    email4 = EmailInput(
        subject="Weekly project status update",
        body="""
        Hi everyone,

        Here's a quick update on our project progress:
        - Frontend team completed the new dashboard UI
        - Backend API documentation is now up to date
        - QA team found 3 minor bugs that are being fixed

        Great work everyone!

        Regards,
        Alex
        """,
    )

    try:
        run = await extract_calendar_event_from_email.run(email4)
        print(run)
    except workflowai.WorkflowAIError as e:
        print(f"As expected, no calendar event found: {e!s}")


if __name__ == "__main__":
    asyncio.run(main())
