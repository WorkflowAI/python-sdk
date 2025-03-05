from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from tests.utils import fixture_text
from workflowai.core.client._models import RunResponse
from workflowai.core.client._utils import default_validator
from workflowai.core.domain.run import Run
from workflowai.core.domain.tool_call import ToolCallRequest


@pytest.mark.parametrize(
    "fixture",
    [
        "task_run.json",
    ],
)
def test_task_run_response(fixture: str):
    txt = fixture_text(fixture)
    task_run = RunResponse.model_validate_json(txt)
    assert task_run


class _TaskOutput(BaseModel):
    a: int
    b: str


class _TaskOutputOpt(BaseModel):
    a: Optional[int] = None
    b: Optional[str] = None


class TestRunResponseToDomain:
    def test_no_version_not_optional(self):
        # Check that partial model is ok
        chunk = RunResponse.model_validate_json('{"id": "1", "task_output": {"a": 1}}')
        assert chunk.task_output == {"a": 1}

        with pytest.raises(ValidationError):  # sanity
            _TaskOutput.model_validate({"a": 1})

        parsed = chunk.to_domain(task_id="1", task_schema_id=1, validator=default_validator(_TaskOutput))
        assert isinstance(parsed, Run)
        assert parsed.output.a == 1
        # b is not defined

        assert parsed.output.b == ""

    def test_no_version_optional(self):
        chunk = RunResponse.model_validate_json('{"id": "1", "task_output": {"a": 1}}')
        assert chunk

        parsed = chunk.to_domain(task_id="1", task_schema_id=1, validator=default_validator(_TaskOutputOpt))
        assert isinstance(parsed, Run)
        assert parsed.output.a == 1
        assert parsed.output.b is None

    def test_with_version(self):
        """Full output is validated since the duration is passed and there are no tool calls"""
        chunk = RunResponse.model_validate_json(
            '{"id": "1", "task_output": {"a": 1, "b": "test"}, "cost_usd": 0.1, "duration_seconds": 1, "version": {"properties": {"a": 1, "b": "test"}}}',  # noqa: E501
        )
        assert chunk

        parsed = chunk.to_domain(task_id="1", task_schema_id=1, validator=default_validator(_TaskOutput))
        assert isinstance(parsed, Run)
        assert parsed.output.a == 1
        assert parsed.output.b == "test"

        assert parsed.cost_usd == 0.1
        assert parsed.duration_seconds == 1

    def test_with_version_validation_fails(self):
        chunk = RunResponse.model_validate_json(
            """{"id": "1", "task_output": {"a": 1},
            "version": {"properties": {"a": 1, "b": "test"}}, "duration_seconds": 1}""",
        )
        with pytest.raises(ValidationError):
            chunk.to_domain(task_id="1", task_schema_id=1, validator=default_validator(_TaskOutput))

    def test_with_tool_calls(self):
        chunk = RunResponse.model_validate_json(
            """{"id": "1", "task_output": {},
            "tool_call_requests": [{"id": "1", "name": "test", "input": {"a": 1}}], "duration_seconds": 1}""",
        )
        assert chunk

        parsed = chunk.to_domain(task_id="1", task_schema_id=1, validator=default_validator(_TaskOutput))
        assert isinstance(parsed, Run)
        assert parsed.tool_call_requests == [ToolCallRequest(id="1", name="test", input={"a": 1})]
