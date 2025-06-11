import json
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

import workflowai
from tests.integration.conftest import CityToCapitalTaskInput, CityToCapitalTaskOutput, IntTestClient
from workflowai.core.domain.run import Run


async def test_run_task(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1)
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> CityToCapitalTaskOutput: ...

    test_client.mock_response()

    task_input = CityToCapitalTaskInput(city="Hello")
    output = await city_to_capital(task_input)

    assert output.capital == "Tokyo"

    test_client.check_request()


async def test_run_task_run(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1)
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> Run[CityToCapitalTaskOutput]: ...

    test_client.mock_response()

    task_input = CityToCapitalTaskInput(city="Hello")
    with_run = await city_to_capital(task_input)

    assert with_run.id == "123"
    assert with_run.output.capital == "Tokyo"

    test_client.check_request()


async def test_run_task_run_with_fallback(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1)
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> Run[CityToCapitalTaskOutput]: ...

    test_client.mock_response()

    task_input = CityToCapitalTaskInput(city="Hello")
    with_run = await city_to_capital(task_input, use_fallback="never")

    assert with_run.id == "123"
    assert with_run.output.capital == "Tokyo"

    body = test_client.check_request()
    assert body["use_fallback"] == "never"


async def test_run_task_run_version(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1, version="staging")
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> Run[CityToCapitalTaskOutput]: ...

    test_client.mock_response()

    task_input = CityToCapitalTaskInput(city="Hello")
    with_run = await city_to_capital(task_input)

    assert with_run.id == "123"
    assert with_run.output.capital == "Tokyo"

    test_client.check_request(version="staging")


async def test_run_task_run_with_model(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1, version="staging")
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> Run[CityToCapitalTaskOutput]: ...

    test_client.mock_response()

    task_input = CityToCapitalTaskInput(city="Hello")
    with_run = await city_to_capital(task_input, model="gpt-4o-latest")

    assert with_run.id == "123"
    assert with_run.output.capital == "Tokyo"

    test_client.check_request(version={"model": "gpt-4o-latest"})


async def test_stream_task_run(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1)
    def city_to_capital(task_input: CityToCapitalTaskInput) -> AsyncIterator[CityToCapitalTaskOutput]: ...

    test_client.mock_stream()

    task_input = CityToCapitalTaskInput(city="Hello")
    chunks = [chunk async for chunk in city_to_capital(task_input)]

    assert [chunk.capital for chunk in chunks] == [
        "",
        "Tok",
        "Tokyo",
        "Tokyo",
    ]


async def test_stream_task_run_custom_id(test_client: IntTestClient) -> None:
    @workflowai.agent(schema_id=1, id="custom-id")
    def city_to_capital(task_input: CityToCapitalTaskInput) -> AsyncIterator[CityToCapitalTaskOutput]: ...

    test_client.mock_stream(task_id="custom-id")

    task_input = CityToCapitalTaskInput(city="Hello")
    chunks = [chunk async for chunk in city_to_capital(task_input)]

    assert [chunk.capital for chunk in chunks] == [
        "",
        "Tok",
        "Tokyo",
        "Tokyo",
    ]


async def test_auto_register(test_client: IntTestClient):
    @workflowai.agent()
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> CityToCapitalTaskOutput: ...

    test_client.mock_register()

    test_client.mock_response()

    res = await city_to_capital(CityToCapitalTaskInput(city="Hello"))
    assert res.capital == "Tokyo"

    test_client.mock_response(capital="Paris")
    # Run it a second time
    res = await city_to_capital(CityToCapitalTaskInput(city="Hello"), use_cache="never")
    assert res.capital == "Paris"

    req = test_client.httpx_mock.get_requests()
    assert len(req) == 3
    assert req[0].url == test_client.REGISTER_URL

    req_body = json.loads(req[0].read())
    assert req_body == {
        "id": "city-to-capital",
        "input_schema": {
            "properties": {
                "city": {
                    "type": "string",
                },
            },
            "required": [
                "city",
            ],
            "type": "object",
        },
        "output_schema": {
            "properties": {
                "capital": {
                    "type": "string",
                },
            },
            "required": [
                "capital",
            ],
            "type": "object",
        },
    }


async def test_run_with_tool(test_client: IntTestClient):
    """Test a round trip with a tool call request and a reply."""

    class _SayHelloToolInput(BaseModel):
        name: str

    class _SayHelloToolOutput(BaseModel):
        message: str

    def say_hello(tool_input: _SayHelloToolInput) -> _SayHelloToolOutput:
        return _SayHelloToolOutput(message=f"Hello {tool_input.name}")

    # Sanity check to make sure that CityToCapitalTaskOutput.capital is a field required by pydantic
    with pytest.raises(AttributeError):
        _ = CityToCapitalTaskOutput.model_construct(None).capital

    @workflowai.agent(id="city-to-capital", tools=[say_hello])
    async def city_to_capital(task_input: CityToCapitalTaskInput) -> CityToCapitalTaskOutput:
        """Say hello to the user"""
        ...

    test_client.mock_register()

    # First response will respond a tool call request
    test_client.mock_response(
        json={
            "id": "1234",
            "task_output": {},
            "tool_call_requests": [
                {
                    "id": "say_hello_1",
                    "name": "say_hello",
                    "input": {"tool_input": {"name": "john"}},
                },
            ],
        },
    )

    # Second response will respond the final output
    test_client.mock_response(url="https://run.workflowai.dev/v1/_/agents/city-to-capital/runs/1234/reply")

    task_input = CityToCapitalTaskInput(city="Hello")
    output = await city_to_capital(task_input)
    assert output.capital == "Tokyo"

    assert len(test_client.httpx_mock.get_requests()) == 3

    run_req = test_client.httpx_mock.get_request(
        url="https://run.workflowai.dev/v1/_/agents/city-to-capital/schemas/1/run",
    )
    assert run_req is not None
    run_req_body = json.loads(run_req.content)
    assert run_req_body["task_input"] == {"city": "Hello"}
    assert set(run_req_body["version"].keys()) == {"enabled_tools", "instructions", "model"}
    assert len(run_req_body["version"]["enabled_tools"]) == 1
    assert run_req_body["version"]["enabled_tools"][0]["name"] == "say_hello"

    reply_req = test_client.httpx_mock.get_request(
        url="https://run.workflowai.dev/v1/_/agents/city-to-capital/runs/1234/reply",
    )
    assert reply_req is not None
    reply_req_body = json.loads(reply_req.content)
    assert reply_req_body["tool_results"] == [
        {
            "id": "say_hello_1",
            "output": {"message": "Hello john"},
        },
    ]
