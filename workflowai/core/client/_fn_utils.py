import inspect
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import (
    Any,
    Generic,
    NamedTuple,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ValidationError
from typing_extensions import Unpack

from workflowai.core.client._api import APIClient
from workflowai.core.client._models import RunResponse
from workflowai.core.client._types import (
    AgentDecorator,
    FinalRunTemplate,
    RunParams,
    RunTemplate,
)
from workflowai.core.client._utils import default_validator
from workflowai.core.client.agent import Agent
from workflowai.core.domain.errors import InvalidGenerationError
from workflowai.core.domain.model import ModelOrStr
from workflowai.core.domain.run import Run
from workflowai.core.domain.task import AgentInput, AgentOutput
from workflowai.core.domain.version_properties import VersionProperties
from workflowai.core.domain.version_reference import VersionReference

# TODO: add sync support


def get_generic_args(t: type[BaseModel]) -> Union[Sequence[type], None]:
    return t.__pydantic_generic_metadata__.get("args")


def check_return_type(return_type_hint: type[Any]) -> tuple[bool, type[BaseModel]]:
    if issubclass(return_type_hint, Run):
        args = get_generic_args(return_type_hint)  # pyright: ignore [reportUnknownArgumentType]
        if not args:
            raise ValueError("Run must have a generic argument")
        output_cls = args[0]
        if not issubclass(output_cls, BaseModel):
            raise ValueError("Run generic argument must be a subclass of BaseModel")
        return False, output_cls
    if issubclass(return_type_hint, BaseModel):
        return True, return_type_hint
    raise ValueError("Function must have a return type hint that is a subclass of Pydantic's 'BaseModel' or 'Run'")


class RunFunctionSpec(NamedTuple):
    stream: bool
    output_only: bool
    input_cls: type[BaseModel]
    output_cls: type[BaseModel]


def is_async_iterator(t: type[Any]) -> bool:
    ori: Any = get_origin(t)
    if not ori:
        return False
    return issubclass(ori, AsyncIterator)


def _first_arg_name(fn: Callable[..., Any]) -> Optional[str]:
    sig = inspect.signature(fn)
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            return param.name
    return None


def extract_fn_spec(fn: RunTemplate[AgentInput, AgentOutput]) -> RunFunctionSpec:
    first_arg_name = _first_arg_name(fn)
    if not first_arg_name:
        raise ValueError("Function must have a first positional argument")
    hints = get_type_hints(fn)
    if "return" not in hints:
        raise ValueError("Function must have a return type hint")
    if first_arg_name not in hints:
        raise ValueError("Function must have a first positional parameter")

    return_type_hint = hints["return"]
    input_cls = hints[first_arg_name]
    if not issubclass(input_cls, BaseModel):
        raise ValueError("First positional parameter must be a subclass of BaseModel")

    output_cls = None

    if is_async_iterator(return_type_hint):
        stream = True
        output_only, output_cls = check_return_type(get_args(return_type_hint)[0])
    else:
        stream = False
        output_only, output_cls = check_return_type(return_type_hint)

    return RunFunctionSpec(stream, output_only, input_cls, output_cls)


class _RunnableAgent(Agent[AgentInput, AgentOutput], Generic[AgentInput, AgentOutput]):
    async def __call__(self, input: AgentInput, **kwargs: Unpack[RunParams[AgentOutput]]):  # noqa: A002
        """Run the agent and return the full run object. Handles recoverable errors when possible

        Args:
            _ (AgentInput): The input to the task.
            id (Optional[str]): A user defined ID for the run. The ID must be a UUID7, ordered by
                creation time. If not provided, a UUID7 will be assigned by the server.
            model (Optional[str]): The model to use for this run. Overrides the version's model if
                provided.
            version (Optional[VersionReference]): The version of the task to run. If not provided,
                the version defined in the task is used.
            instructions (Optional[str]): Custom instructions for this run. Overrides the version's
                instructions if provided.
            temperature (Optional[float]): The temperature to use for this run. Overrides the
                version's temperature if provided.
            use_cache (CacheUsage, optional): How to use the cache. Defaults to "auto".
                "auto" (default): if a previous run exists with the same version and input, and if
                    the temperature is 0, the cached output is returned
                "always": the cached output is returned when available, regardless
                    of the temperature value
                "never": the cache is never used
            labels (Optional[set[str]], optional): Labels are deprecated, please use metadata instead.
            metadata (Optional[dict[str, Any]], optional): A dictionary of metadata to attach to the
                run.
            max_retry_delay (Optional[float], optional): The maximum delay between retries in
                milliseconds. Defaults to 60000.
            max_retry_count (Optional[float], optional): The maximum number of retry attempts.
                Defaults to 1.
            max_turns (Optional[int], optional): Maximum number of tool iteration cycles.
                Defaults to 10.
            validator (Optional[OutputValidator[AgentOutput]], optional): Custom validator for the
                output.

        Returns:
            Run[AgentOutput]: The task run object.
        """
        try:
            return await self.run(input, **kwargs)
        except InvalidGenerationError as e:
            if e.partial_output and e.run_id:
                try:
                    validator, _ = self._sanitize_validator(kwargs, default_validator(self.output_cls))
                    run = self._build_run_no_tools(
                        chunk=RunResponse(
                            id=e.run_id,
                            task_output=e.partial_output,
                        ),
                        schema_id=self.schema_id or 0,
                        validator=validator,
                        partial=False,
                    )
                    run.error = e.error
                    return run

                except ValidationError:
                    # Error is not recoverable so not returning anything
                    pass
            raise e


class _RunnableOutputOnlyAgent(Agent[AgentInput, AgentOutput], Generic[AgentInput, AgentOutput]):
    async def __call__(self, input: AgentInput, **kwargs: Unpack[RunParams[AgentOutput]]):  # noqa: A002
        """Run the agent

        This variant returns only the output, without the run metadata.

        Args:
            _ (AgentInput): The input to the task.
            id (Optional[str]): A user defined ID for the run. The ID must be a UUID7, ordered by
                creation time. If not provided, a UUID7 will be assigned by the server.
            model (Optional[str]): The model to use for this run. Overrides the version's model if
                provided.
            version (Optional[VersionReference]): The version of the task to run. If not provided,
                the version defined in the task is used.
            instructions (Optional[str]): Custom instructions for this run. Overrides the version's
                instructions if provided.
            temperature (Optional[float]): The temperature to use for this run. Overrides the
                version's temperature if provided.
            use_cache (CacheUsage, optional): How to use the cache. Defaults to "auto".
                "auto" (default): if a previous run exists with the same version and input, and if
                    the temperature is 0, the cached output is returned
                "always": the cached output is returned when available, regardless
                    of the temperature value
                "never": the cache is never used
            labels (Optional[set[str]], optional): Labels are deprecated, please use metadata instead.
            metadata (Optional[dict[str, Any]], optional): A dictionary of metadata to attach to the
                run.
            max_retry_delay (Optional[float], optional): The maximum delay between retries in
                milliseconds. Defaults to 60000.
            max_retry_count (Optional[float], optional): The maximum number of retry attempts.
                Defaults to 1.
            max_turns (Optional[int], optional): Maximum number of tool iteration cycles.
                Defaults to 10.
            validator (Optional[OutputValidator[AgentOutput]], optional): Custom validator for the
                output.

        Returns:
            AgentOutput: The output of the task.
        """
        return (await self.run(input, **kwargs)).output


class _RunnableStreamAgent(Agent[AgentInput, AgentOutput], Generic[AgentInput, AgentOutput]):
    def __call__(self, input: AgentInput, **kwargs: Unpack[RunParams[AgentOutput]]):  # noqa: A002
        """Stream the output of the agent

        Args:
            _ (AgentInput): The input to the task.
            id (Optional[str]): A user defined ID for the run. The ID must be a UUID7, ordered by
                creation time. If not provided, a UUID7 will be assigned by the server.
            model (Optional[str]): The model to use for this run. Overrides the version's model if
                provided.
            version (Optional[VersionReference]): The version of the task to run. If not provided,
                the version defined in the task is used.
            instructions (Optional[str]): Custom instructions for this run. Overrides the version's
                instructions if provided.
            temperature (Optional[float]): The temperature to use for this run. Overrides the
                version's temperature if provided.
            use_cache (CacheUsage, optional): How to use the cache. Defaults to "auto".
                "auto" (default): if a previous run exists with the same version and input, and if
                    the temperature is 0, the cached output is returned
                "always": the cached output is returned when available, regardless
                    of the temperature value
                "never": the cache is never used
            labels (Optional[set[str]], optional): Labels are deprecated, please use metadata instead.
            metadata (Optional[dict[str, Any]], optional): A dictionary of metadata to attach to the
                run.
            max_retry_delay (Optional[float], optional): The maximum delay between retries in
                milliseconds. Defaults to 60000.
            max_retry_count (Optional[float], optional): The maximum number of retry attempts.
                Defaults to 1.
            max_turns (Optional[int], optional): Maximum number of tool iteration cycles.
                Defaults to 10.
            validator (Optional[OutputValidator[AgentOutput]], optional): Custom validator for the
                output.

        Returns:
            AsyncIterator[Run[AgentOutput]]: An async iterator yielding task run objects.
        """
        return self.stream(input, **kwargs)


class _RunnableStreamOutputOnlyAgent(Agent[AgentInput, AgentOutput], Generic[AgentInput, AgentOutput]):
    async def __call__(self, input: AgentInput, **kwargs: Unpack[RunParams[AgentOutput]]):  # noqa: A002
        """Stream the output of the agent

        This variant yields only the output, without the run metadata.

        Args:
            _ (AgentInput): The input to the task.
            id (Optional[str]): A user defined ID for the run. The ID must be a UUID7, ordered by
                creation time. If not provided, a UUID7 will be assigned by the server.
            model (Optional[str]): The model to use for this run. Overrides the version's model if
                provided.
            version (Optional[VersionReference]): The version of the task to run. If not provided,
                the version defined in the task is used.
            instructions (Optional[str]): Custom instructions for this run. Overrides the version's
                instructions if provided.
            temperature (Optional[float]): The temperature to use for this run. Overrides the
                version's temperature if provided.
            use_cache (CacheUsage, optional): How to use the cache. Defaults to "auto".
                "auto" (default): if a previous run exists with the same version and input, and if
                    the temperature is 0, the cached output is returned
                "always": the cached output is returned when available, regardless
                    of the temperature value
                "never": the cache is never used
            labels (Optional[set[str]], optional): Labels are deprecated, please use metadata instead.
            metadata (Optional[dict[str, Any]], optional): A dictionary of metadata to attach to the
                run.
            max_retry_delay (Optional[float], optional): The maximum delay between retries in
                milliseconds. Defaults to 60000.
            max_retry_count (Optional[float], optional): The maximum number of retry attempts.
                Defaults to 1.
            max_turns (Optional[int], optional): Maximum number of tool iteration cycles.
                Defaults to 10.
            validator (Optional[OutputValidator[AgentOutput]], optional): Custom validator for the
                output.

        Returns:
            AsyncIterator[AgentOutput]: An async iterator yielding task outputs.
        """
        async for chunk in self.stream(input, **kwargs):
            yield chunk.output


def clean_docstring(docstring: Optional[str]) -> str:
    """Clean a docstring by removing empty lines at start/end and normalizing indentation."""
    if not docstring:
        return ""

    # Split into lines and remove empty lines at start/end
    lines = [line.rstrip() for line in docstring.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    # Find and remove common indentation
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    lines = [line[indent:] if line.strip() else "" for line in lines]

    return "\n".join(lines)


def wrap_run_template(
    client: Callable[[], APIClient],
    agent_id: str,
    schema_id: Optional[int],
    version: Optional[VersionReference],
    model: Optional[ModelOrStr],
    fn: RunTemplate[AgentInput, AgentOutput],
    tools: Optional[Iterable[Callable[..., Any]]] = None,
) -> Union[
    _RunnableAgent[AgentInput, AgentOutput],
    _RunnableOutputOnlyAgent[AgentInput, AgentOutput],
    _RunnableStreamAgent[AgentInput, AgentOutput],
    _RunnableStreamOutputOnlyAgent[AgentInput, AgentOutput],
]:
    stream, output_only, input_cls, output_cls = extract_fn_spec(fn)

    if not version and (fn.__doc__ or model):
        version = VersionProperties(
            instructions=clean_docstring(fn.__doc__),
            model=model,
        )

    if stream:
        task_cls = _RunnableStreamOutputOnlyAgent if output_only else _RunnableStreamAgent
    else:
        task_cls = _RunnableOutputOnlyAgent if output_only else _RunnableAgent
    return task_cls(  # pyright: ignore [reportUnknownVariableType]
        agent_id=agent_id,
        input_cls=input_cls,
        output_cls=output_cls,
        api=client,
        schema_id=schema_id,
        version=version,
        tools=tools,
    )


def agent_id_from_fn_name(fn: Any) -> str:
    return fn.__name__.replace("_", "-").lower()


def agent_wrapper(
    client: Callable[[], APIClient],
    schema_id: Optional[int] = None,
    agent_id: Optional[str] = None,
    version: Optional[VersionReference] = None,
    model: Optional[ModelOrStr] = None,
    tools: Optional[Iterable[Callable[..., Any]]] = None,
) -> AgentDecorator:
    def wrap(fn: RunTemplate[AgentInput, AgentOutput]):
        tid = agent_id or agent_id_from_fn_name(fn)
        # TODO[types]: Not sure why a cast is needed here
        agent = cast(
            FinalRunTemplate[AgentInput, AgentOutput],
            wrap_run_template(client, tid, schema_id, version, model, fn, tools),
        )

        agent.__doc__ = """A class representing an AI agent that can process inputs and generate outputs.

    The Agent class provides functionality to run AI-powered tasks with support for streaming,
    tool execution, and version management.
"""
        agent.__name__ = fn.__name__

        return agent  # pyright: ignore [reportReturnType]

    # TODO[types]: pyright is unhappy with generics
    return wrap  # pyright: ignore [reportReturnType]
