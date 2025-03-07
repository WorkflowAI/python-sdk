"""
This example demonstrates how to use the reply() method to have a conversation with the agent/LLM.
After getting an initial response, you can use reply() to ask follow-up questions or request
confirmation. The agent/LLM maintains context from the previous interaction, allowing it to:

1. Confirm its previous output
2. Correct mistakes if needed
3. Provide additional explanation
4. Refine its response based on new information

Example:
    run = await my_agent(input)  # Initial response
    run = await run.reply(user_message="Are you sure?")  # Ask for confirmation
    ...
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]

import workflowai
from workflowai import Model


class NameExtractionInput(BaseModel):
    """Input containing a sentence with a person's name."""

    sentence: str = Field(description="A sentence containing a person's name.")


class NameExtractionOutput(BaseModel):
    """Output containing the extracted first and last name."""

    first_name: str = Field(
        default="",
        description="The person's first name extracted from the sentence.",
    )
    last_name: str = Field(
        default="",
        description="The person's last name extracted from the sentence.",
    )


@workflowai.agent(id="name-extractor", model=Model.GPT_4O_MINI_LATEST)
async def extract_name(_: NameExtractionInput) -> NameExtractionOutput:
    """
    Extract a person's first and last name from a sentence.
    Be precise and consider cultural variations in name formats.
    If multiple names are present, focus on the most prominent one.
    """
    ...


async def main():
    # Example sentences to test
    sentences = [
        "My friend John Smith went to the store.",
        "Dr. Maria Garcia-Rodriguez presented her research.",
        "The report was written by James van der Beek last week.",
    ]

    for sentence in sentences:
        print(f"\nProcessing: {sentence}")

        # Initial extraction
        run = await extract_name.run(NameExtractionInput(sentence=sentence))

        print(f"Extracted: {run.output.first_name} {run.output.last_name}")

        # The reply() method allows you to continue the conversation with the LLM
        # by sending a follow-up message. The LLM will maintain context from the
        # previous interaction and can confirm or revise its previous output.
        # Here we ask it to double check its extraction.
        run = await run.reply(user_message="Are you sure?")

        print("\nAfter double-checking:")
        print(f"Final extraction: {run.output.first_name} {run.output.last_name}")


if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(main())
