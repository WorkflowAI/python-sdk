"""
This example demonstrates how to use images with WorkflowAI agents. It shows how to:
1. Pass image inputs to an agent
2. Analyze city photos for identification
3. Structure detailed visual analysis results
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]

import workflowai
from workflowai import WorkflowAIError
from workflowai.core.domain.model import Model
from workflowai.fields import Image


class ImageInput(BaseModel):
    image: Image = Field(description="The image to analyze")


class ImageOutput(BaseModel):
    city: str = Field(default="", description="Name of the city shown in the image")
    country: str = Field(default="", description="Name of the country where the city is located")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level in the identification (0-1)",
    )


@workflowai.agent(id="city-identifier", model=Model.GEMINI_2_0_FLASH_LATEST)
async def identify_city_from_image(image_input: ImageInput) -> ImageOutput:
    """
    Analyze the provided image and identify the city and country shown in it.
    If the image shows a recognizable landmark or cityscape, identify the city and country.
    If uncertain, indicate lower confidence or leave fields empty.

    Focus on:
    - Famous landmarks
    - Distinctive architecture
    - Recognizable skylines
    - Cultural elements that identify the location

    Return empty strings if the city/country cannot be determined with reasonable confidence.
    """
    ...


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "assets", "new-york-city.jpg")

    # With a properly async function you should use an async open
    # see https://github.com/Tinche/aiofiles for example
    with open(image_path, "rb") as image_file:  # noqa: ASYNC230
        import base64

        content = base64.b64encode(image_file.read()).decode("utf-8")

    image = Image(content_type="image/jpeg", data=content)
    try:
        agent_run = await identify_city_from_image.run(
            ImageInput(image=image),
        )
    except WorkflowAIError as e:
        print(f"Failed to run task. Code: {e.error.code}. Message: {e.error.message}")
        return

    print("\n--------\nAgent output:\n", agent_run.output, "\n--------\n")
    print(f"Cost: ${agent_run.cost_usd:.10f}")
    print(f"Latency: {agent_run.duration_seconds:.2f}s")

    # Example using URL for Image
    image_url = "https://workflowai.blob.core.windows.net/workflowai-public/fixtures/paris.jpg"
    image = Image(url=image_url)
    agent_run = await identify_city_from_image.run(
        ImageInput(image=image),
    )

    print("\n--------\nAgent output:\n", agent_run.output, "\n--------\n")
    print(f"Cost: ${agent_run.cost_usd:.10f}")
    print(f"Latency: {agent_run.duration_seconds:.2f}s")


if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(main())
