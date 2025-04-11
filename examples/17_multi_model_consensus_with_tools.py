"""
This example builds upon example 16 (multi_model_consensus.py) but gives more agency to
the response-combiner. While example 16 uses a fixed set of models defined in the main
function, this version lets the response-combiner itself decide which models are most
appropriate for the question.

This example demonstrates agent delegation, where one agent (the response-combiner) can
dynamically invoke another agent (get_model_response) via tools. By providing the ask_model
function as a tool, the response-combiner can:

1. Choose which models to query based on the nature and complexity of the question
2. Adapt its strategy based on initial responses (e.g. asking specialized models for clarification)
3. Use its own reasoning to determine when it has enough perspectives to synthesize an answer

This hierarchical approach allows the response-combiner agent to orchestrate multiple model
queries by delegating to the get_model_response agent through tool calls. The response-combiner
acts as a "manager" agent that can strategically coordinate with "worker" agents to gather
the insights needed.
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class AskModelInput(BaseModel):
    """Input for asking a question to a specific model."""

    question: str = Field(description="The question to ask")
    model: Model = Field(description="The model to ask the question to")


class AskModelOutput(BaseModel):
    """Output from asking a question to a model."""

    response: str = Field(description="The model's response to the question")


# This function acts as a tool that allows one agent to delegate to another agent.
# The response-combiner agent can use this tool to dynamically query different models
# through the get_model_response agent. This creates a hierarchy where the
# response-combiner orchestrates multiple model queries by delegating to get_model_response.
async def ask_model(query_input: AskModelInput) -> AskModelOutput:
    """Ask a specific model a question and return its response."""
    run = await get_model_response.run(
        MultiModelInput(
            question=query_input.question,
        ),
        model=query_input.model,
    )
    # get_model_response.run() returns a Run[ModelResponse], so we need to access the output
    return AskModelOutput(response=run.output.response)


class MultiModelInput(BaseModel):
    """Input model containing the question to ask all models."""

    question: str = Field(
        description="The question to ask all models",
    )


class ModelResponse(BaseModel):
    """Response from an individual model."""

    response: str = Field(description="The model's response to the question")


class CombinerInput(BaseModel):
    """Input for the response combiner."""

    original_question: str = Field(description="The question to ask multiple models")


class CombinedOutput(BaseModel):
    """Final output combining responses from all models."""

    combined_answer: str = Field(
        description="Synthesized answer combining insights from all models",
    )
    explanation: str = Field(
        description="Explanation of how the responses were combined and why",
    )
    models_used: list[str] = Field(
        description="List of models whose responses were combined",
    )


@workflowai.agent(
    id="question-answerer",
)
async def get_model_response(query: MultiModelInput) -> ModelResponse:
    """
    Make sure to:
    1. Provide a clear and detailed response
    2. Stay focused on the question asked
    3. Be specific about any assumptions made
    4. Highlight areas of uncertainty
    """
    ...


@workflowai.agent(
    id="response-combiner",
    model=Model.GPT_4O_MINI_LATEST,
    tools=[ask_model],
)
async def combine_responses(responses_input: CombinerInput) -> CombinedOutput:
    """
    Analyze and combine responses from multiple models into a single coherent answer.
    You should ask at least 3 different models to get a diverse set of perspectives.

    You are an expert at analyzing and synthesizing information from multiple sources.
    Your task is to:
    1. Review the responses from different models (at least 3)
    2. Identify key insights and unique perspectives from each
    3. Create a comprehensive answer that combines the best elements
    4. Explain your synthesis process
    5. List all models whose responses were used in the synthesis

    Please ensure the combined answer is:
    - Accurate and well-reasoned
    - Incorporates unique insights from each model
    - Clear and coherent
    - Properly attributed when using specific insights
    - Based on responses from at least 3 different models
    """
    ...


@pytest.mark.xfail(reason="Example is flaky")
async def main():
    # Example: Scientific explanation
    print("\nExample: Scientific Concept")
    print("-" * 50)
    question = "What is dark matter and why is it important for our understanding of the universe?"

    # Let the response-combiner handle asking the models
    combined = await combine_responses.run(
        CombinerInput(
            original_question=question,
        ),
    )
    print(combined)


if __name__ == "__main__":
    asyncio.run(main())
