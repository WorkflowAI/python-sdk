from typing import (
    Annotated,
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
)

from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from workflowai.core.domain.cache_usage import CacheUsage
from workflowai.core.domain.task import AgentOutput
from workflowai.core.domain.version_reference import VersionReference

AgentInputContra = TypeVar("AgentInputContra", bound=BaseModel, contravariant=True)
AgentOutputCov = TypeVar("AgentOutputCov", bound=BaseModel, covariant=True)


class OutputValidator(Protocol, Generic[AgentOutputCov]):
    def __call__(self, data: dict[str, Any], partial: bool) -> AgentOutputCov:
        """A way to convert a json object into an AgentOutput

        Args:
            data (dict[str, Any]): The json object to convert
            partial (bool): Whether the json is partial, meaning that
            it may not contain all the fields required by the AgentOutput model.

        Returns:
            AgentOutputCov: The converted AgentOutput
        """
        ...


class VersionRunParams(TypedDict):
    model: NotRequired[Optional[str]]
    version: NotRequired[Optional["VersionReference"]]
    instructions: NotRequired[Optional[str]]
    temperature: NotRequired[Optional[float]]


class OtherRunParams(TypedDict):
    use_cache: NotRequired["CacheUsage"]

    max_retry_delay: NotRequired[float]
    max_retry_count: NotRequired[float]

    max_turns: NotRequired[int]  # 10 by default
    max_turns_raises: NotRequired[bool]  # True by default


class BaseRunParams(VersionRunParams, OtherRunParams):
    metadata: NotRequired[Optional[dict[str, Any]]]


class RunParams(BaseRunParams, Generic[AgentOutput]):
    id: Annotated[
        NotRequired[str],
        "A user defined ID for the run. The ID must be a UUID7, ordered by creation time."
        "If not provided, a UUID7 will be assigned by the server",
    ]
    validator: NotRequired[OutputValidator["AgentOutput"]]
