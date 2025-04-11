"""
This example demonstrates agent delegation, where one agent (the orchestrator) can dynamically
invoke other agents through tools. This pattern is useful when you want to:

1. Let an agent dynamically choose which specialized agents to use
2. Allow the orchestrator to adapt its strategy based on initial responses
3. Enable flexible workflows where the sequence of agent calls isn't fixed
4. Track which agents were used and why

The example shows how to:
1. Set up a tool that allows one agent to call another
2. Structure input/output types for delegation
3. Configure the orchestrator agent with the delegation tool
4. Handle responses and track agent usage
"""

import asyncio
from typing import Optional

import pytest
from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class DelegateInput(BaseModel):
    """Input for delegating a task to a specialized agent."""

    task: str = Field(description="The task to delegate")
    model: Model = Field(description="The model to use for this task")
    context: Optional[str] = Field(
        default=None,
        description="Additional context that might help the agent",
    )


class DelegateOutput(BaseModel):
    """Output from a delegated task."""

    response: str = Field(description="The agent's response to the task")
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0,
        le=1,
    )


class WorkerInput(BaseModel):
    """Input for the worker agent."""

    task: str = Field(description="The task to perform")
    context: Optional[str] = Field(
        default=None,
        description="Additional context that might help with the task",
    )


class WorkerOutput(BaseModel):
    """Output from the worker agent."""

    response: str = Field(description="The response to the task")
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0,
        le=1,
    )


class OrchestratorInput(BaseModel):
    """Input for the orchestrator agent."""

    objective: str = Field(description="The high-level objective to achieve")
    requirements: list[str] = Field(
        description="List of specific requirements or constraints",
        default_factory=list,
    )


class OrchestratorOutput(BaseModel):
    """Final output from the orchestrator."""

    solution: str = Field(description="The final solution that meets the objective")
    explanation: str = Field(description="Explanation of how the solution was derived")
    agents_used: list[str] = Field(
        description="List of agents/models used in the process",
        default_factory=list,
    )


@workflowai.agent()
async def worker_agent(agent_input: WorkerInput) -> WorkerOutput:
    """
    A specialized worker agent that handles specific tasks.

    Make sure to:
    1. Focus on the specific task assigned
    2. Provide detailed reasoning for your approach
    3. Include confidence level in your response
    """
    ...


async def delegate_task(agent_input: DelegateInput) -> DelegateOutput:
    """Delegate a task to a worker agent with a specific model."""
    # Run the worker agent with the specified model
    run = await worker_agent.run(
        WorkerInput(
            task=agent_input.task,
            context=agent_input.context,
        ),
        model=agent_input.model,
    )
    return DelegateOutput(
        response=run.output.response,
        confidence=run.output.confidence,
    )


@workflowai.agent(
    id="orchestrator",
    model=Model.GEMINI_2_0_FLASH_LATEST,
    tools=[delegate_task],
)
async def orchestrator_agent(agent_input: OrchestratorInput) -> OrchestratorOutput:
    """
    You are an expert orchestrator that breaks down complex objectives into smaller tasks
    and delegates them to specialized agents. You can use the delegate_task tool to assign
    work to other agents.

    Your responsibilities:
    1. Break down the objective into smaller, focused tasks
    2. Choose appropriate models for each task based on its nature:
       - GPT-4O for complex reasoning or creative tasks
       - Claude for analytical or structured tasks
       - Gemini for technical or scientific tasks
    3. Use the delegate_task tool to assign work
    4. Evaluate responses and confidence levels
    5. Request additional work if needed
    6. Synthesize all responses into a cohesive solution
    7. Track which models were used and why

    Make sure the final solution:
    - Meets all specified requirements
    - Is well-reasoned and explained
    - Acknowledges any limitations or uncertainties
    - Lists all models/agents used in the process
    """
    ...


@pytest.mark.xfail(reason="Example is flaky")
async def main():
    # Example: Software architecture task
    print("\nExample: Software Architecture Design")
    print("-" * 50)

    result = await orchestrator_agent.run(
        OrchestratorInput(
            objective="Design a scalable microservices architecture for an e-commerce platform",
            requirements=[
                "Must handle 10,000+ concurrent users",
                "Include payment processing and inventory management",
                "Ensure data consistency across services",
                "Provide real-time order tracking",
            ],
        ),
    )

    print("\nSolution:")
    print(result.output.solution)
    print("\nExplanation:")
    print(result.output.explanation)
    print("\nAgents Used:")
    for agent in result.output.agents_used:
        print(f"- {agent}")


if __name__ == "__main__":
    asyncio.run(main())
