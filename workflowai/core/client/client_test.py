import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pytest_httpx import HTTPXMock, IteratorStream

from tests.models.hello_task import HelloTaskInput, HelloTaskOutput
from workflowai.core.client.client import WorkflowAI
from workflowai.core.domain.run import Run


@pytest.fixture
def workflowai():
    # using httpx_mock to make sure we don't actually call the api
    return WorkflowAI(api_key="test")


class TestAgentDecorator:
    @pytest.fixture
    def mock_run_fn(self):
        with patch("workflowai.core.client.agent.Agent.run", autospec=True) as run_mock:
            yield run_mock

    @pytest.fixture
    def mock_stream_fn(self):
        with patch("workflowai.core.client.agent.Agent.stream", autospec=True) as run_mock:
            yield run_mock

    def test_fn_name(self, workflowai: WorkflowAI):
        @workflowai.task(schema_id=1, task_id="123")
        async def fn(task_input: HelloTaskInput) -> HelloTaskOutput: ...

        assert fn.__name__ == "fn"
        assert fn.__doc__ is not None
        assert callable(fn)

    async def test_run_output_only(self, workflowai: WorkflowAI, mock_run_fn: Mock):
        @workflowai.task(schema_id=1, task_id="123")
        async def fn(task_input: HelloTaskInput) -> HelloTaskOutput: ...

        mock_run_fn.return_value = Run(
            id="1",
            output=HelloTaskOutput(message="hello"),
            agent_id="123",
            schema_id=1,
        )

        output = await fn(HelloTaskInput(name="Alice"))

        assert output == HelloTaskOutput(message="hello")

    async def test_run_with_version(self, workflowai: WorkflowAI, mock_run_fn: Mock):
        @workflowai.task(schema_id=1, task_id="123")
        async def fn(task_input: HelloTaskInput) -> Run[HelloTaskOutput]: ...

        mock_run_fn.return_value = Run(
            id="1",
            output=HelloTaskOutput(message="hello"),
            agent_id="123",
            schema_id=1,
        )

        run = await fn(HelloTaskInput(name="Alice"))

        assert run.id == "1"
        assert run.output == HelloTaskOutput(message="hello")
        assert isinstance(run, Run)

    async def test_stream(self, workflowai: WorkflowAI, httpx_mock: HTTPXMock):
        # We avoid mocking the run fn directly here, python does weird things with
        # having to await async iterators depending on how they are defined so instead we mock
        # the underlying api call to check that we don't need the extra await

        @workflowai.task(schema_id=1, task_id="123")
        def fn(task_input: HelloTaskInput) -> AsyncIterator[Run[HelloTaskOutput]]: ...

        httpx_mock.add_response(
            stream=IteratorStream(
                [
                    b'data: {"id":"1","task_output":{"message":""}}\n\n',
                    b'data: {"id":"1","task_output":{"message":"hel"}}\n\ndata: {"id":"1","task_output":{"message":"hello"}}\n\n',  # noqa: E501
                    b'data: {"id":"1","task_output":{"message":"hello"},"cost_usd":0.01,"duration_seconds":10.1}\n\n',
                ],
            ),
        )

        chunks = [chunk async for chunk in fn(HelloTaskInput(name="Alice"))]

        def _run(output: HelloTaskOutput, **kwargs: Any) -> Run[HelloTaskOutput]:
            return Run(id="1", agent_id="123", schema_id=1, output=output, **kwargs)

        assert chunks == [
            _run(HelloTaskOutput(message="")),
            _run(HelloTaskOutput(message="hel")),
            _run(HelloTaskOutput(message="hello")),
            _run(HelloTaskOutput(message="hello"), duration_seconds=10.1, cost_usd=0.01),
        ]

    async def test_stream_output_only(self, workflowai: WorkflowAI, httpx_mock: HTTPXMock):
        @workflowai.task(schema_id=1)
        def fn(task_input: HelloTaskInput) -> AsyncIterator[HelloTaskOutput]: ...

        httpx_mock.add_response(
            stream=IteratorStream(
                [
                    b'data: {"id":"1","task_output":{"message":""}}\n\n',
                    b'data: {"id":"1","task_output":{"message":"hel"}}\n\ndata: {"id":"1","task_output":{"message":"hello"}}\n\n',  # noqa: E501
                    b'data: {"id":"1","task_output":{"message":"hello"},"cost_usd":0.01,"duration_seconds":10.1}\n\n',
                ],
            ),
        )

        chunks = [chunk async for chunk in fn(HelloTaskInput(name="Alice"))]

        # We could remove duplicates but it would add a condition for everyone and every chunk
        # that might not be useful.
        assert chunks == [
            HelloTaskOutput(message=""),
            HelloTaskOutput(message="hel"),
            HelloTaskOutput(message="hello"),
            HelloTaskOutput(message="hello"),
        ]


class TestSendFeedback:
    async def test_send_feedback(self, workflowai: WorkflowAI, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.workflowai.com/v1/feedback",
            method="POST",
            status_code=200,
        )

        await workflowai.send_feedback(feedback_token="bliblu", outcome="positive")

        reqs = httpx_mock.get_requests()
        assert len(reqs) == 1
        req = reqs[0]
        assert req.method == "POST"
        assert req.url == "https://api.workflowai.com/v1/feedback"
        assert json.loads(req.content) == {
            "feedback_token": "bliblu",
            "outcome": "positive",
        }
