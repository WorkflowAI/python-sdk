"""
This example demonstrates how to create an agent that extracts and redacts Personal Identifiable
Information (PII) from text. It showcases:

1. Handling sensitive information with clear categorization
2. Structured output with both redacted text and extracted PII
3. Enum usage for PII categories
4. Comprehensive PII detection and redaction
"""

import asyncio
from enum import Enum

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class PIIType(str, Enum):
    """Categories of Personal Identifiable Information."""
    NAME = "NAME"  # Full names, first names, last names
    EMAIL = "EMAIL"  # Email addresses
    PHONE = "PHONE"  # Phone numbers, fax numbers
    ADDRESS = "ADDRESS"  # Physical addresses, postal codes
    SSN = "SSN"  # Social Security Numbers, National IDs
    DOB = "DOB"  # Date of birth, age
    FINANCIAL = "FINANCIAL"  # Credit card numbers, bank accounts
    LICENSE = "LICENSE"  # Driver's license, professional licenses
    URL = "URL"  # Personal URLs, social media profiles
    OTHER = "OTHER"  # Other types of PII not covered above


class PIIExtraction(BaseModel):
    """Represents an extracted piece of PII with its type."""
    text: str = Field(description="The extracted PII text")
    type: PIIType = Field(description="The category of PII")
    start_index: int = Field(description="Starting position in the original text")
    end_index: int = Field(description="Ending position in the original text")


class PIIInput(BaseModel):
    """Input model for PII extraction."""
    text: str = Field(
        description="The text to analyze for PII",
        examples=[
            "Hi, I'm John Doe. You can reach me at john.doe@email.com or call 555-0123. "
            "My SSN is 123-45-6789 and I live at 123 Main St, Springfield, IL 62701.",
        ],
    )


class PIIOutput(BaseModel):
    """Output model containing redacted text and extracted PII."""
    redacted_text: str = Field(
        description="The original text with all PII replaced by [REDACTED]",
        examples=[
            "Hi, I'm [REDACTED]. You can reach me at [REDACTED] or call [REDACTED]. "
            "My SSN is [REDACTED] and I live at [REDACTED].",
        ],
    )
    extracted_pii: list[PIIExtraction] = Field(
        description="List of extracted PII items with their types and positions",
        examples=[
            [
                {"text": "John Doe", "type": "NAME", "start_index": 8, "end_index": 16},
                {"text": "john.doe@email.com", "type": "EMAIL", "start_index": 30, "end_index": 47},
                {"text": "555-0123", "type": "PHONE", "start_index": 57, "end_index": 65},
            ],
        ],
    )


@workflowai.agent(
    id="pii-extractor",
    model=Model.LLAMA_4_SCOUT_BASIC,
)
async def extract_pii(input_data: PIIInput) -> PIIOutput:
    """
    Extract and redact Personal Identifiable Information (PII) from text.

    Guidelines:
    1. Identify all instances of PII in the input text
    2. Categorize each PII instance into one of the defined types
    3. Record the exact position (start and end indices) of each PII instance
    4. Replace all PII in the text with [REDACTED]
    5. Ensure no sensitive information is left unredacted
    6. Be thorough but avoid over-redacting non-PII information
    7. When in doubt about PII type, use the OTHER category
    8. Maintain the original text structure and formatting
    9. Handle overlapping PII appropriately (e.g., name within an email)
    10. Consider context when identifying PII (e.g., distinguish between company and personal emails)
    """
    ...


async def main():
    # Example 1: Basic PII extraction
    print("\nExample 1: Basic PII")
    print("-" * 50)
    text = (
        "Hello, my name is Sarah Johnson and my email is sarah.j@example.com. "
        "You can reach me at (555) 123-4567 or visit my blog at blog.sarahj.net. "
        "I was born on 03/15/1985."
    )
    result = await extract_pii.run(PIIInput(text=text))
    print("\nOriginal text:")
    print(text)
    print("\nRedacted text:")
    print(result.output.redacted_text)
    print("\nExtracted PII:")
    for pii in result.output.extracted_pii:
        print(f"- {pii.type}: {pii.text} (positions {pii.start_index}-{pii.end_index})")

    # Example 2: Complex PII with financial and address information
    print("\n\nExample 2: Complex PII")
    print("-" * 50)
    text = (
        "Customer: David Wilson\n"
        "Card: 4532-9678-1234-5678\n"
        "Address: 789 Oak Avenue, Apt 4B\n"
        "          Boston, MA 02108\n"
        "License: MA12-345-678\n"
        "SSN: 078-05-1120"
    )
    result = await extract_pii.run(PIIInput(text=text))
    print("\nOriginal text:")
    print(text)
    print("\nRedacted text:")
    print(result.output.redacted_text)
    print("\nExtracted PII:")
    for pii in result.output.extracted_pii:
        print(f"- {pii.type}: {pii.text} (positions {pii.start_index}-{pii.end_index})")


if __name__ == "__main__":
    asyncio.run(main())
