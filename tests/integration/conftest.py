import json
from collections.abc import Callable
from typing import Any, Optional, Union
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pytest_httpx import HTTPXMock, IteratorStream

from workflowai.core.client.client import WorkflowAI


@pytest.fixture(scope="module", autouse=True)
def init_client():
    with patch("workflowai.shared_client", new=WorkflowAI(api_key="test", url="https://run.workflowai.dev")):
        yield


class CityToCapitalTaskInput(BaseModel):
    city: str


class CityToCapitalTaskOutput(BaseModel):
    capital: str


class IntTestClient:
    REGISTER_URL = "https://api.workflowai.dev/v1/_/agents"

    def __init__(self, httpx_mock: HTTPXMock):
        self.httpx_mock = httpx_mock

    def mock_register(self, schema_id: int = 1, task_id: str = "city-to-capital", variant_id: str = "1"):
        self.httpx_mock.add_response(
            method="POST",
            url=self.REGISTER_URL,
            json={"schema_id": schema_id, "variant_id": variant_id, "id": task_id, "uid": 123},
        )

    def mock_response(
        self,
        task_id: str = "city-to-capital",
        capital: str = "Tokyo",
        json: Optional[dict[str, Any]] = None,
        url: Optional[str] = None,
        status_code: int = 200,
    ):
        self.httpx_mock.add_response(
            method="POST",
            url=url or f"https://run.workflowai.dev/v1/_/agents/{task_id}/schemas/1/run",
            json=json or {"id": "123", "task_output": {"capital": capital}},
            status_code=status_code,
        )

    def mock_stream(
        self,
        task_id: str = "city-to-capital",
        outputs: Optional[list[dict[str, Any]]] = None,
        run_id: str = "1",
        metadata: Optional[dict[str, Any]] = None,
    ):
        outputs = outputs or [
            {"capital": ""},
            {"capital": "Tok"},
            {"capital": "Tokyo"},
        ]
        if metadata is None:
            metadata = {"cost_usd": 0.01, "duration_seconds": 10.1}

        payloads = [{"id": run_id, "task_output": o} for o in outputs]

        final_payload = {**payloads[-1], **metadata}
        payloads.append(final_payload)
        streams = [f"data: {json.dumps(p)}\n\n".encode() for p in payloads]

        self.httpx_mock.add_response(
            url=f"https://run.workflowai.dev/v1/_/agents/{task_id}/schemas/1/run",
            stream=IteratorStream(streams),
        )

    def check_register(
        self,
        task_id: str = "city-to-capital",
        input_schema: Optional[Union[dict[str, Any], Callable[[dict[str, Any]], None]]] = None,
        output_schema: Optional[Union[dict[str, Any], Callable[[dict[str, Any]], None]]] = None,
    ):
        request = self.httpx_mock.get_request(url=self.REGISTER_URL)
        assert request is not None
        assert request.headers["Authorization"] == "Bearer test"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["x-workflowai-source"] == "sdk"
        assert request.headers["x-workflowai-language"] == "python"

        body = json.loads(request.content)
        assert body["id"] == task_id
        if callable(input_schema):
            input_schema(body["input_schema"])
        else:
            assert body["input_schema"] == input_schema or {"city": {"type": "string"}}
        if callable(output_schema):
            output_schema(body["output_schema"])
        else:
            assert body["output_schema"] == output_schema or {"capital": {"type": "string"}}

    def check_request(
        self,
        version: Any = "production",
        task_id: str = "city-to-capital",
        task_input: Optional[dict[str, Any]] = None,
        **matchers: Any,
    ):
        if not matchers:
            matchers = {"url": f"https://run.workflowai.dev/v1/_/agents/{task_id}/schemas/1/run"}
        request = self.httpx_mock.get_request(**matchers)
        assert request is not None
        body = json.loads(request.content)
        assert body["task_input"] == task_input or {"city": "Hello"}
        assert body["version"] == version
        assert request.headers["Authorization"] == "Bearer test"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["x-workflowai-source"] == "sdk"
        assert request.headers["x-workflowai-language"] == "python"


@pytest.fixture
def test_client(httpx_mock: HTTPXMock) -> IntTestClient:
    return IntTestClient(httpx_mock)
