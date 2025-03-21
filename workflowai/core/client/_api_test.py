from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import BaseModel
from pytest_httpx import HTTPXMock, IteratorStream

from workflowai.core.client._api import APIClient
from workflowai.core.domain.errors import WorkflowAIError


class TestAPIClientExtractError:
    def test_extract_error(self):
        client = APIClient(url="test_url", api_key="test_api_key")

        # Test valid JSON error response
        response = httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Test error message",
                    "details": {"key": "value"},
                },
                "id": "test_task_123",
            },
        )

        error = client._extract_error(response, response.content)  # pyright:ignore[reportPrivateUsage]
        assert isinstance(error, WorkflowAIError)
        assert error.error.message == "Test error message"
        assert error.error.details == {"key": "value"}
        assert error.run_id == "test_task_123"
        assert error.response == response

    def test_extract_partial_output(self):
        client = APIClient(url="test_url", api_key="test_api_key")

        # Test valid JSON error response
        response = httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Test error message",
                    "details": {"key": "value"},
                },
                "id": "test_task_123",
                "task_output": {"key": "value"},
            },
        )

        error = client._extract_error(response, response.content)  # pyright:ignore[reportPrivateUsage]
        assert isinstance(error, WorkflowAIError)
        assert error.error.message == "Test error message"
        assert error.error.details == {"key": "value"}
        assert error.run_id == "test_task_123"
        assert error.partial_output == {"key": "value"}
        assert error.response == response

    def test_extract_error_invalid_json(self):
        client = APIClient(url="test_url", api_key="test_api_key")

        # Test invalid JSON response
        invalid_data = b"Invalid JSON data"
        response = httpx.Response(status_code=400, content=invalid_data)

        with pytest.raises(WorkflowAIError) as e:
            client._extract_error(response, invalid_data)  # pyright:ignore[reportPrivateUsage]
        assert isinstance(e.value, WorkflowAIError)
        assert e.value.error.message == "Unknown error"
        assert e.value.error.details == {"raw": "b'Invalid JSON data'"}
        assert e.value.response == response

    def test_extract_error_with_custom_error(self):
        client = APIClient(url="test_url", api_key="test_api_key")

        # Test with provided exception
        invalid_data = "{'detail': 'Not Found'}"
        response = httpx.Response(status_code=404, content=invalid_data)
        exception = ValueError("Custom error")

        with pytest.raises(WorkflowAIError) as e:
            client._extract_error(response, invalid_data, exception)  # pyright:ignore[reportPrivateUsage]
        assert isinstance(e.value, WorkflowAIError)
        assert e.value.error.message == "Custom error"
        assert e.value.error.details == {"raw": "{'detail': 'Not Found'}"}
        assert e.value.response == response


@pytest.fixture
def client() -> APIClient:
    return APIClient(url="https://blabla.com", api_key="test_api_key")


class _TestInputModel(BaseModel):
    bla: str = "bla"


class _TestOutputModel(BaseModel):
    a: str


class TestAPIClientStream:
    async def test_stream_404(self, httpx_mock: HTTPXMock, client: APIClient):
        class _TestInputModel(BaseModel):
            test_input: str

        class _TestOutputModel(BaseModel):
            test_output: str

        httpx_mock.add_response(status_code=404)

        with pytest.raises(WorkflowAIError) as e:  # noqa: PT012
            async for _ in client.stream(
                method="GET",
                path="test_path",
                data=_TestInputModel(test_input="test"),
                returns=_TestOutputModel,
            ):
                pass

        assert e.value.response
        assert e.value.response.status_code == 404
        assert e.value.response.reason_phrase == "Not Found"

    @pytest.fixture
    async def stream_fn(self, client: APIClient):
        async def _stm():
            return [
                chunk
                async for chunk in client.stream(
                    method="GET",
                    path="test_path",
                    data=_TestInputModel(),
                    returns=_TestOutputModel,
                )
            ]

        return _stm

    async def test_stream_with_single_chunk(
        self,
        stream_fn: Callable[[], Awaitable[list[_TestOutputModel]]],
        httpx_mock: HTTPXMock,
    ):
        httpx_mock.add_response(
            stream=IteratorStream(
                [
                    b'data: {"a":"test"}\n\n',
                ],
            ),
        )

        chunks = await stream_fn()
        assert chunks == [_TestOutputModel(a="test")]

    @pytest.mark.parametrize(
        "streamed_chunks",
        [
            # 2 perfect chunks([b'data: {"a":"test"}\n\n', b'data: {"a":"test2"}\n\n'],),
            [b'data: {"a":"test"}\n\n', b'data: {"a":"test2"}\n\n'],
            # 2 chunks in one
            [b'data: {"a":"test"}\n\ndata: {"a":"test2"}\n\n'],
            # Split not at the end
            [b'data: {"a":"test"}', b'\n\ndata: {"a":"test2"}\n\n'],
            # Really messy
            [b"dat", b'a: {"a":"', b'test"}', b"\n", b"\ndata", b': {"a":"test2"}\n\n'],
        ],
    )
    async def test_stream_with_multiple_chunks(
        self,
        stream_fn: Callable[[], Awaitable[list[_TestOutputModel]]],
        httpx_mock: HTTPXMock,
        streamed_chunks: list[bytes],
    ):
        assert isinstance(streamed_chunks, list), "sanity check"
        assert all(isinstance(chunk, bytes) for chunk in streamed_chunks), "sanity check"

        httpx_mock.add_response(stream=IteratorStream(streamed_chunks))
        chunks = await stream_fn()
        assert chunks == [_TestOutputModel(a="test"), _TestOutputModel(a="test2")]


class TestReadAndConnectError:
    @pytest.mark.parametrize("exception", [httpx.ReadError("arg"), httpx.ConnectError("arg")])
    async def test_get(self, httpx_mock: HTTPXMock, client: APIClient, exception: Exception):
        httpx_mock.add_exception(exception)

        with pytest.raises(WorkflowAIError) as e:
            await client.get(path="test_path", returns=_TestOutputModel)

        assert e.value.error.code == "connection_error"

    @pytest.mark.parametrize("exception", [httpx.ReadError("arg"), httpx.ConnectError("arg")])
    async def test_post(self, httpx_mock: HTTPXMock, client: APIClient, exception: Exception):
        httpx_mock.add_exception(exception)

        with pytest.raises(WorkflowAIError) as e:
            await client.post(path="test_path", data=_TestInputModel(), returns=_TestOutputModel)

        assert e.value.error.code == "connection_error"

    @pytest.mark.parametrize("exception", [httpx.ReadError("arg"), httpx.ConnectError("arg")])
    async def test_stream(self, httpx_mock: HTTPXMock, client: APIClient, exception: Exception):
        httpx_mock.add_exception(exception)

        with pytest.raises(WorkflowAIError) as e:  # noqa: PT012
            async for _ in client.stream(
                method="GET",
                path="test_path",
                data=_TestInputModel(),
                returns=_TestOutputModel,
            ):
                pass

        assert e.value.error.code == "connection_error"
