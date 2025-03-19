from collections.abc import AsyncIterator
from typing import Union
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from tests.models.hello_task import HelloTaskInput, HelloTaskOutput
from tests.utils import mock_aiter
from workflowai.core.client._api import APIClient
from workflowai.core.client._fn_utils import (
    _RunnableAgent,  # pyright: ignore [reportPrivateUsage]
    _RunnableOutputOnlyAgent,  # pyright: ignore [reportPrivateUsage]
    _RunnableStreamAgent,  # pyright: ignore [reportPrivateUsage]
    _RunnableStreamOutputOnlyAgent,  # pyright: ignore [reportPrivateUsage]
    agent_wrapper,
    clean_docstring,
    extract_fn_spec,
    get_generic_args,
    is_async_iterator,
)
from workflowai.core.client._models import RunRequest, RunResponse
from workflowai.core.domain.run import Run


async def say_hello(_: HelloTaskInput) -> HelloTaskOutput: ...


async def say_hello_run(bla: HelloTaskInput) -> Run[HelloTaskOutput]: ...


def stream_hello(_: HelloTaskInput) -> AsyncIterator[HelloTaskOutput]: ...


def stream_hello_run(_: HelloTaskInput) -> AsyncIterator[Run[HelloTaskOutput]]: ...


class TestGetGenericArgs:
    def test_get_generic_arg(self):
        assert get_generic_args(Run[HelloTaskOutput]) == (HelloTaskOutput,)


class TestIsAsyncIterator:
    def test_is_async_iterator(self):
        assert is_async_iterator(AsyncIterator[HelloTaskOutput])
        assert not is_async_iterator(HelloTaskOutput)
        assert not is_async_iterator(BaseModel)


class TestExtractFnData:
    def test_run_output_only(self):
        assert extract_fn_spec(say_hello) == (False, True, HelloTaskInput, HelloTaskOutput)

    def test_run(self):
        assert extract_fn_spec(say_hello_run) == (False, False, HelloTaskInput, HelloTaskOutput)

    def test_stream_output_only(self):
        assert extract_fn_spec(stream_hello) == (True, True, HelloTaskInput, HelloTaskOutput)

    def test_stream(self):
        assert extract_fn_spec(stream_hello_run) == (True, False, HelloTaskInput, HelloTaskOutput)


class TestAgentWrapper:
    """Check that the agent wrapper returns the correct types, and checks the implementation of the __call__ fn"""

    @pytest.fixture
    def mock_api_client(self):
        return Mock(spec=APIClient)

    async def fn_run(self, task_input: HelloTaskInput) -> Run[HelloTaskOutput]: ...

    async def test_fn_run(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello")(self.fn_run)
        assert isinstance(wrapped, _RunnableAgent)

        mock_api_client.post.return_value = RunResponse(id="1", task_output={"message": "Hello, World!"})
        run = await wrapped(HelloTaskInput(name="World"))
        assert isinstance(run, Run)
        assert run.id == "1"
        assert run.output == HelloTaskOutput(message="Hello, World!")

    async def test_fn_run_with_default_cache(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello", use_cache="never")(self.fn_run)
        assert isinstance(wrapped, _RunnableAgent)

        mock_api_client.post.return_value = RunResponse(id="1", task_output={"message": "Hello, World!"})
        run = await wrapped(HelloTaskInput(name="World"))
        assert isinstance(run, Run)

        mock_api_client.post.assert_called_once()
        req = mock_api_client.post.call_args.args[1]
        assert isinstance(req, RunRequest)
        assert req.use_cache == "never"

        mock_api_client.post.reset_mock()

        # Check that it can be overridden
        _ = await wrapped(HelloTaskInput(name="World"), use_cache="always")

        mock_api_client.post.assert_called_once()
        req = mock_api_client.post.call_args.args[1]
        assert isinstance(req, RunRequest)
        assert req.use_cache == "always"

    def fn_stream(self, task_input: HelloTaskInput) -> AsyncIterator[Run[HelloTaskOutput]]: ...

    async def test_fn_stream(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello")(self.fn_stream)
        assert isinstance(wrapped, _RunnableStreamAgent)

        mock_api_client.stream.return_value = mock_aiter(RunResponse(id="1", task_output={"message": "Hello, World!"}))
        chunks = [c async for c in wrapped(HelloTaskInput(name="World"))]
        assert len(chunks) == 1
        assert isinstance(chunks[0], Run)
        assert chunks[0].id == "1"
        assert chunks[0].output == HelloTaskOutput(message="Hello, World!")

    async def fn_run_output_only(self, task_input: HelloTaskInput) -> HelloTaskOutput: ...

    async def test_fn_run_output_only(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello")(self.fn_run_output_only)
        assert isinstance(wrapped, _RunnableOutputOnlyAgent)

        mock_api_client.post.return_value = RunResponse(id="1", task_output={"message": "Hello, World!"})
        run = await wrapped(HelloTaskInput(name="World"))
        assert isinstance(run, HelloTaskOutput)
        assert run == HelloTaskOutput(message="Hello, World!")

    def fn_stream_output_only(self, task_input: HelloTaskInput) -> AsyncIterator[HelloTaskOutput]: ...

    async def test_fn_stream_output_only(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello")(self.fn_stream_output_only)
        assert isinstance(wrapped, _RunnableStreamOutputOnlyAgent)

        mock_api_client.stream.return_value = mock_aiter(RunResponse(id="1", task_output={"message": "Hello, World!"}))
        chunks = [c async for c in wrapped(HelloTaskInput(name="World"))]
        assert len(chunks) == 1
        assert isinstance(chunks[0], HelloTaskOutput)
        assert chunks[0] == HelloTaskOutput(message="Hello, World!")

    async def test_agent_functions_and_doc(self, mock_api_client: Mock):
        wrapped = agent_wrapper(lambda: mock_api_client, schema_id=1, agent_id="hello")(self.fn_run_output_only)
        assert wrapped.__doc__

        mock_api_client.post.return_value = RunResponse(id="1", task_output={"message": "Hello, World!"})
        output = await wrapped(HelloTaskInput(name="World"))
        assert isinstance(output, HelloTaskOutput)

        mock_api_client.post.return_value = RunResponse(id="1", task_output={"message": "Hello, World!"})
        run = await wrapped.run(HelloTaskInput(name="World"), model="gpt-4o")
        assert isinstance(run, Run)

        mock_api_client.stream.return_value = mock_aiter(RunResponse(id="1", task_output={"message": "Hello, World!"}))
        chunks = [c async for c in wrapped.stream(HelloTaskInput(name="World"))]
        assert len(chunks) == 1
        assert isinstance(chunks[0], Run)

        assert wrapped.run.__doc__
        assert wrapped.stream.__doc__
        assert wrapped.reply.__doc__
        assert wrapped.register.__doc__
        assert wrapped.__call__.__doc__


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Empty docstrings
        ("", ""),
        (None, ""),
        # Single line docstrings
        ("Hello world", "Hello world"),
        ("  Hello world  ", "Hello world"),
        # Docstring with empty lines at start/end
        (
            """

        Hello world

        """,
            "Hello world",
        ),
        # Multi-line docstring with indentation
        (
            """
        First line
        Second line
            Indented line
        Last line
        """,
            "First line\nSecond line\n    Indented line\nLast line",
        ),
        # Docstring with empty lines in between
        (
            """
        First line

        Second line

        Third line
        """,
            "First line\n\nSecond line\n\nThird line",
        ),
        # Real-world example
        (
            """
        Find the capital city of the country where the input city is located.

        Guidelines:
        1. First identify the country where the input city is located
        2. Then provide the capital city of that country
        3. Include an interesting historical or cultural fact about the capital
        4. Be accurate and precise with geographical information
        5. If the input city is itself the capital, still provide the information
        """,
            "Find the capital city of the country where the input city is located.\n\n"
            "Guidelines:\n"
            "1. First identify the country where the input city is located\n"
            "2. Then provide the capital city of that country\n"
            "3. Include an interesting historical or cultural fact about the capital\n"
            "4. Be accurate and precise with geographical information\n"
            "5. If the input city is itself the capital, still provide the information",
        ),
    ],
)
def test_clean_docstring(value: Union[str, None], expected: str):
    assert clean_docstring(value) == expected
