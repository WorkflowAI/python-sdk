from typing import Any, Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]
from typing_extensions import NotRequired, TypedDict

from workflowai.core._common_types import OutputValidator
from workflowai.core.domain.cache_usage import CacheUsage
from workflowai.core.domain.completion import Completion
from workflowai.core.domain.run import Run
from workflowai.core.domain.task import AgentOutput
from workflowai.core.domain.tool_call import ToolCall as DToolCall
from workflowai.core.domain.tool_call import ToolCallRequest as DToolCallRequest
from workflowai.core.domain.tool_call import ToolCallResult as DToolCallResult
from workflowai.core.domain.version import Version as DVersion
from workflowai.core.domain.version_properties import VersionProperties as DVersionProperties
from workflowai.core.utils._iter import safe_map_list

# TODO: we should likely only use typed dicts here to avoid validation issues
# We have some typed dicts but pydantic also validates them


class RunRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="A cliend defined ID. Must be a UUID7")

    task_input: dict[str, Any]

    version: Union[str, int, dict[str, Any]]

    use_cache: Optional[CacheUsage] = None

    metadata: Optional[dict[str, Any]] = None

    labels: Optional[set[str]] = None  # deprecated, to be included in metadata

    private_fields: Optional[set[str]] = None

    stream: Optional[bool] = None


class ReplyRequest(BaseModel):
    user_message: Optional[str] = None
    version: Union[str, int, dict[str, Any]]
    metadata: Optional[dict[str, Any]] = None

    class ToolResult(BaseModel):
        id: str
        output: Optional[Any]
        error: Optional[str]

        @classmethod
        def from_domain(cls, tool_result: DToolCallResult):
            return cls(
                id=tool_result.id,
                output=tool_result.output,
                error=tool_result.error,
            )

    tool_results: Optional[list[ToolResult]] = None

    stream: Optional[bool] = None


class VersionProperties(TypedDict):
    model: NotRequired[Optional[str]]
    provider: NotRequired[Optional[str]]
    temperature: NotRequired[Optional[float]]
    instructions: NotRequired[Optional[str]]


def version_properties_to_domain(properties: VersionProperties) -> DVersionProperties:
    return DVersionProperties.model_construct(
        None,
        **properties,
    )


class Version(BaseModel):
    properties: VersionProperties

    def to_domain(self) -> DVersion:
        return DVersion(
            properties=version_properties_to_domain(self.properties),
        )


class ToolCall(TypedDict):
    id: str
    name: str
    input_preview: str
    output_preview: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    status: NotRequired[Optional[Literal["success", "failed", "in_progress"]]]


def tool_call_to_domain(tool_call: ToolCall) -> DToolCall:
    return DToolCall(
        id=tool_call["id"],
        name=tool_call["name"],
        input_preview=tool_call["input_preview"],
        output_preview=tool_call.get("output_preview"),
        error=tool_call.get("error"),
        status=tool_call.get("status"),
    )


class ToolCallRequest(TypedDict):
    id: str
    name: str
    input: dict[str, Any]


def tool_call_request_to_domain(tool_call_request: ToolCallRequest) -> DToolCallRequest:
    return DToolCallRequest(
        id=tool_call_request["id"],
        name=tool_call_request["name"],
        input=tool_call_request["input"],
    )


class RunResponse(BaseModel):
    id: str
    task_output: dict[str, Any]

    version: Optional[Version] = None
    duration_seconds: Optional[float] = None
    cost_usd: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None

    tool_calls: Optional[list[ToolCall]] = None
    tool_call_requests: Optional[list[ToolCallRequest]] = None

    def to_domain(
        self,
        task_id: str,
        task_schema_id: int,
        validator: OutputValidator[AgentOutput],
        partial: Optional[bool] = None,
    ) -> Run[AgentOutput]:
        # We do partial validation if either:
        # - there are tool call requests, which means that the output can be empty
        # - the run has not yet finished, for exmaple when streaming, in which case the duration_seconds is None
        if partial is None:
            partial = bool(self.tool_call_requests) or self.duration_seconds is None
        return Run(
            id=self.id,
            agent_id=task_id,
            schema_id=task_schema_id,
            output=validator(self.task_output, partial),
            version=self.version and self.version.to_domain(),
            duration_seconds=self.duration_seconds,
            cost_usd=self.cost_usd,
            tool_calls=safe_map_list(self.tool_calls, tool_call_to_domain),
            tool_call_requests=safe_map_list(self.tool_call_requests, tool_call_request_to_domain),
        )


class CreateAgentRequest(BaseModel):
    id: str = Field(description="The agent id, must be unique per tenant and URL safe")
    input_schema: dict[str, Any] = Field(description="The input schema for the agent")
    output_schema: dict[str, Any] = Field(description="The output schema for the agent")


class CreateAgentResponse(BaseModel):
    id: str
    schema_id: int


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    provider_name: str = Field(description="Name of the model provider")
    price_per_input_token_usd: Optional[float] = Field(None, description="Cost per input token in USD")
    price_per_output_token_usd: Optional[float] = Field(None, description="Cost per output token in USD")
    release_date: Optional[str] = Field(None, description="Release date of the model")
    context_window_tokens: Optional[int] = Field(None, description="Size of the context window in tokens")
    quality_index: Optional[float] = Field(None, description="Quality index of the model")


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str = Field(description="Unique identifier for the model")
    name: str = Field(description="Display name of the model")
    icon_url: Optional[str] = Field(None, description="URL for the model's icon")
    modes: list[str] = Field(default_factory=list, description="Supported modes for this model")
    is_not_supported_reason: Optional[str] = Field(
        None,
        description="Reason why the model is not supported, if applicable",
    )
    average_cost_per_run_usd: Optional[float] = Field(None, description="Average cost per run in USD")
    is_latest: bool = Field(default=False, description="Whether this is the latest version of the model")
    metadata: Optional[ModelMetadata] = Field(None, description="Additional metadata about the model")
    is_default: bool = Field(default=False, description="Whether this is the default model")
    providers: list[str] = Field(default_factory=list, description="List of providers that offer this model")


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """A generic paginated response."""

    items: list[T] = Field(description="List of items in this page")
    count: Optional[int] = Field(None, description="Total number of items available")


class ListModelsResponse(Page[ModelInfo]):
    """Response from the list models API endpoint."""


class ListModelsRequest(BaseModel):
    instructions: Optional[str] = Field(default=None, description="Used to detect internal tools")
    requires_tools: Optional[bool] = Field(default=None, description="Whether the agent uses external tools")


class CompletionsResponse(BaseModel):
    """Response from the completions API endpoint."""

    completions: list[Completion]
