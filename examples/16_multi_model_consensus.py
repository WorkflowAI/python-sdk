"""
This example demonstrates how to ask the same question to multiple different LLMs
and then combine their responses into a single coherent answer using another LLM.

The example uses three different models for answering:
- GPT-4O Mini
- Gemini 2.0 Flash
- Llama 3.3 70B

Then uses O3 Mini (with medium reasoning effort) to analyze and combine their responses.
"""

import asyncio

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class MultiModelInput(BaseModel):
    """Input model containing the question to ask all models."""

    question: str = Field(
        description="The question to ask all models",
    )
    model_name: str = Field(
        description="Name of the model providing the response",
    )


class ModelResponse(BaseModel):
    """Response from an individual model."""

    model_name: str = Field(description="Name of the model that provided this response")
    response: str = Field(description="The model's response to the question")


class CombinerInput(BaseModel):
    """Input for the response combiner."""

    responses: list[ModelResponse] = Field(description="List of responses to combine")


class CombinedOutput(BaseModel):
    """Final output combining responses from all models."""

    combined_answer: str = Field(
        description="Synthesized answer combining insights from all models",
    )
    explanation: str = Field(
        description="Explanation of how the responses were combined and why",
    )


@workflowai.agent(
    id="question-answerer",
)
async def get_model_response(query: MultiModelInput) -> ModelResponse:
    """Get response from the specified model."""
    ...


@workflowai.agent(
    id="response-combiner",
    model=Model.O3_MINI_2025_01_31_MEDIUM_REASONING_EFFORT,
)
async def combine_responses(responses_input: CombinerInput) -> CombinedOutput:
    """
    Analyze and combine responses from multiple models into a single coherent answer.

    You are an expert at analyzing and synthesizing information from multiple sources.
    Your task is to:
    1. Review the responses from different models
    2. Identify key insights and unique perspectives from each
    3. Create a comprehensive answer that combines the best elements
    4. Explain your synthesis process

    Please ensure the combined answer is:
    - Accurate and well-reasoned
    - Incorporates unique insights from each model
    - Clear and coherent
    - Properly attributed when using specific insights
    """
    ...


async def main():
    # Example: Scientific explanation
    print("\nExample: Scientific Concept")
    print("-" * 50)
    question = "What is dark matter and why is it important for our understanding of the universe?"

    # Get responses from all models
    models = [
        (Model.GPT_4O_MINI_LATEST, "GPT-4O Mini"),
        (Model.GEMINI_2_0_FLASH_LATEST, "Gemini 2.0 Flash"),
        (Model.LLAMA_3_3_70B, "Llama 3.3 70B"),
    ]

    responses = []
    for model, model_name in models:
        run = await get_model_response.run(
            MultiModelInput(
                question=question,
                model_name=model_name,
            ),
            model=model,
        )
        responses.append(run.output)

    # Combine responses
    combined = await combine_responses.run(CombinerInput(responses=responses))
    print(combined)


if __name__ == "__main__":
    asyncio.run(main())
