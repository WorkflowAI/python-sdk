from collections.abc import AsyncIterator, Iterable
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    Union,
    overload,
)

from typing_extensions import Unpack

from workflowai.core._common_types import AgentInputContra, AgentOutputCov, RunParams
from workflowai.core.client._models import ModelInfo
from workflowai.core.domain.run import Run
from workflowai.core.domain.task import AgentInput, AgentOutput
from workflowai.core.domain.tool_call import ToolCallResult


class _BaseObject(Protocol):
    __name__: str
    __doc__: Optional[str]
    __module__: str
    __qualname__: str
    __annotations__: dict[str, Any]
    __defaults__: Optional[tuple[Any, ...]]


class RunFn(_BaseObject, Generic[AgentInputContra, AgentOutput], Protocol):
    async def __call__(self, _: AgentInputContra, /) -> Run[AgentOutput]: ...


class RunFnOutputOnly(_BaseObject, Generic[AgentInputContra, AgentOutputCov], Protocol):
    async def __call__(self, _: AgentInputContra, /) -> AgentOutputCov: ...


class StreamRunFn(_BaseObject, Generic[AgentInputContra, AgentOutput], Protocol):
    def __call__(self, _: AgentInputContra, /) -> AsyncIterator[Run[AgentOutput]]: ...


class StreamRunFnOutputOnly(_BaseObject, Generic[AgentInputContra, AgentOutputCov], Protocol):
    def __call__(self, _: AgentInputContra, /) -> AsyncIterator[AgentOutputCov]: ...


RunTemplate = Union[
    RunFn[AgentInput, AgentOutput],
    RunFnOutputOnly[AgentInput, AgentOutput],
    StreamRunFn[AgentInput, AgentOutput],
    StreamRunFnOutputOnly[AgentInput, AgentOutput],
]


class AgentInterface(_BaseObject, Generic[AgentInputContra, AgentOutput], Protocol):
    __kwdefaults__: Optional[dict[str, Any]]
    __code__: Any

    @property
    def agent_uid(self) -> int: ...

    async def run(
        self,
        agent_input: AgentInputContra,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> Run[AgentOutput]: ...

    def stream(
        self,
        agent_input: AgentInputContra,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> AsyncIterator[Run[AgentOutput]]: ...

    async def register(self): ...

    async def reply(
        self,
        run_id: str,
        user_message: Optional[str] = None,
        tool_results: Optional[Iterable[ToolCallResult]] = None,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ): ...

    async def list_models(self) -> list[ModelInfo]: ...


class RunnableAgent(AgentInterface[AgentInputContra, AgentOutput], Protocol):
    async def __call__(
        self,
        _: AgentInputContra,
        /,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> Run[AgentOutput]: ...


class RunnableOutputAgent(AgentInterface[AgentInputContra, AgentOutput], Protocol):
    async def __call__(
        self,
        _: AgentInputContra,
        /,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> AgentOutput: ...


class StreamableAgent(AgentInterface[AgentInputContra, AgentOutput], Protocol):
    def __call__(
        self,
        _: AgentInputContra,
        /,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> AsyncIterator[Run[AgentOutput]]: ...


class StreamableOutputAgent(AgentInterface[AgentInputContra, AgentOutput], Protocol):
    def __call__(
        self,
        _: AgentInputContra,
        /,
        **kwargs: Unpack[RunParams[AgentOutput]],
    ) -> AsyncIterator[AgentOutput]: ...


FinalRunTemplate = Union[
    RunnableAgent[AgentInput, AgentOutput],
    RunnableOutputAgent[AgentInput, AgentOutput],
    StreamableAgent[AgentInput, AgentOutput],
    StreamableOutputAgent[AgentInput, AgentOutput],
]


class AgentDecorator(Protocol):
    @overload
    def __call__(self, fn: RunFn[AgentInput, AgentOutput]) -> RunnableAgent[AgentInput, AgentOutput]: ...

    @overload
    def __call__(
        self,
        fn: RunFnOutputOnly[AgentInput, AgentOutput],
    ) -> RunnableOutputAgent[AgentInput, AgentOutput]: ...

    @overload
    def __call__(self, fn: StreamRunFn[AgentInput, AgentOutput]) -> StreamableAgent[AgentInput, AgentOutput]: ...

    @overload
    def __call__(
        self,
        fn: StreamRunFnOutputOnly[AgentInput, AgentOutput],
    ) -> StreamableOutputAgent[AgentInput, AgentOutput]: ...

    def __call__(self, fn: RunTemplate[AgentInput, AgentOutput]) -> FinalRunTemplate[AgentInput, AgentOutput]: ...
