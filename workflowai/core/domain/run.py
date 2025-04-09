import json
from collections.abc import Iterable
from typing import Any, Generic, Optional, Protocol

from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]
from typing_extensions import Unpack

from workflowai import env
from workflowai.core import _common_types
from workflowai.core.client import _types
from workflowai.core.domain.completion import Completion
from workflowai.core.domain.errors import BaseError
from workflowai.core.domain.task import AgentOutput
from workflowai.core.domain.tool_call import ToolCall, ToolCallRequest, ToolCallResult
from workflowai.core.domain.version import Version


class Run(BaseModel, Generic[AgentOutput]):
    """
    A run is an instance of a agent with a specific input and output.

    This class represent a run that already has been recorded and possibly
    been evaluated
    """

    id: str
    agent_id: str
    schema_id: int
    output: AgentOutput

    duration_seconds: Optional[float] = None
    cost_usd: Optional[float] = None

    version: Optional[Version] = Field(
        default=None,
        description="The version of the agent that was run. Only provided if the version differs from the version"
        " specified in the request, for example in case of a model fallback",
    )

    metadata: Optional[dict[str, Any]] = None

    tool_calls: Optional[list[ToolCall]] = None
    tool_call_requests: Optional[list[ToolCallRequest]] = None

    error: Optional[BaseError] = Field(
        default=None,
        description="An error that occurred during the run. Only provided if the run failed.",
    )

    feedback_token: Optional[str] = None

    _agent: Optional["_AgentBase[AgentOutput]"] = None

    def __eq__(self, other: object) -> bool:
        # Probably over simplistic but the object is not crazy complicated
        # We just need a way to ignore the agent object
        if not isinstance(other, Run):
            return False
        if self.__dict__ == other.__dict__:
            return True
        # Otherwise we check without the agent
        for field, value in self.__dict__.items():
            if field == "_agent":
                continue
            if not value == other.__dict__.get(field):
                return False
        return True

    async def reply(
        self,
        user_message: Optional[str] = None,
        tool_results: Optional[Iterable[ToolCallResult]] = None,
        **kwargs: Unpack["_common_types.RunParams[AgentOutput]"],
    ):
        if not self._agent:
            raise ValueError("Agent is not set")
        return await self._agent.reply(
            run_id=self.id,
            user_message=user_message,
            tool_results=tool_results,
            **kwargs,
        )

    @property
    def model(self):
        if self.version is None:
            return None
        return self.version.properties.model

    def format_output(self) -> str:
        """Format the run output as a string.

        Returns a formatted string containing:
        1. The output as a nicely formatted JSON object
        2. The cost with $ prefix (if available)
        3. The latency with 2 decimal places and 's' suffix (if available)
        4. The run URL for viewing in the web UI

        Example:
            Output:
            ==================================================
            {
              "message": "hello"
            }
            ==================================================
            Cost: $ 0.001
            Latency: 1.23s
            URL: https://workflowai.com/_/agents/agent-1/runs/test-id
        """
        # Format the output string
        output: list[str] = []
        # In case of partial validation, it is possible that the output is an empty model
        if self.output.model_dump():
            # Use model_dump_json which handles datetime serialization correctly
            output_json = self.output.model_dump_json(indent=2)
            output += [
                "\nOutput:",
                "=" * 50,
                output_json,  # Use the JSON string directly
                "=" * 50,
            ]
        if self.tool_call_requests:
            output += [
                "\nTool Call Requests:",
                "=" * 50,
                json.dumps(self.model_dump(include={"tool_call_requests"})["tool_call_requests"], indent=2),
                "=" * 50,
            ]

        # Add run information if available
        if self.cost_usd is not None:
            output.append(f"Cost: $ {self.cost_usd:.5f}")
        if self.duration_seconds is not None:
            output.append(f"Latency: {self.duration_seconds:.2f}s")

        # Always add the run URL
        output.append(f"URL: {self.run_url}")

        return "\n".join(output)

    def __str__(self) -> str:
        """Return a string representation of the run."""
        return self.format_output()

    @property
    def run_url(self):
        return f"{env.WORKFLOWAI_APP_URL}/_/agents/{self.agent_id}/runs/{self.id}"

    async def fetch_completions(self) -> list[Completion]:
        """Fetch the completions for this run.

        Returns:
            CompletionsResponse: The completions response containing a list of completions
            with their messages, responses and usage information.

        Raises:
            ValueError: If the agent is not set or if the run id is not set.
        """
        if not self._agent:
            raise ValueError("Agent is not set")
        if not self.id:
            raise ValueError("Run id is not set")

        return await self._agent.fetch_completions(self.id)


class _AgentBase(Protocol, Generic[AgentOutput]):
    async def reply(
        self,
        run_id: str,
        user_message: Optional[str] = None,
        tool_results: Optional[Iterable[ToolCallResult]] = None,
        **kwargs: Unpack["_types.RunParams[AgentOutput]"],
    ) -> "Run[AgentOutput]":
        """Reply to a run. Either a user_message or tool_results must be provided."""
        ...

    async def fetch_completions(self, run_id: str) -> list[Completion]: ...
