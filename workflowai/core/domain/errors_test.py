from typing import Optional
from unittest.mock import Mock

import httpx
import pytest
from freezegun import freeze_time

from workflowai.core.domain.errors import (
    BaseError,
    InvalidAPIKeyError,
    WorkflowAIError,
    _retry_after_to_delay_seconds,  # pyright: ignore [reportPrivateUsage]
)


class TestErrorFromResponse:
    def test_extract_error(self):
        # Test valid JSON error response
        response = httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Test error message",
                    "details": {"key": "value"},
                    "code": "object_not_found",
                },
                "id": "test_task_123",
            },
        )

        error = WorkflowAIError.from_response(response, response.content)
        assert isinstance(error, WorkflowAIError)
        assert error.error.message == "Test error message"
        assert error.error.details == {"key": "value"}
        assert error.run_id == "test_task_123"
        assert error.response == response
        assert error.code == "object_not_found"

    def test_extract_partial_output(self):
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

        error = WorkflowAIError.from_response(response, response.content)
        assert isinstance(error, WorkflowAIError)
        assert error.error.message == "Test error message"
        assert error.error.details == {"key": "value"}
        assert error.run_id == "test_task_123"
        assert error.partial_output == {"key": "value"}
        assert error.response == response

    def test_extract_error_invalid_json(self):
        # Test invalid JSON response
        invalid_data = b"Invalid JSON data"
        response = httpx.Response(status_code=400, content=invalid_data)

        error = WorkflowAIError.from_response(response, invalid_data)
        assert isinstance(error, WorkflowAIError)
        assert error.error.message == "Unknown error"
        assert error.error.details == {"raw": "b'Invalid JSON data'"}
        assert error.response == response


@freeze_time("2024-01-01T00:00:00Z")
@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (None, None),
        ("10", 10),
        ("Wed, 01 Jan 2024 00:00:10 UTC", 10),
    ],
)
def test_retry_after_to_delay_seconds(retry_after: Optional[str], expected: Optional[float]):
    assert _retry_after_to_delay_seconds(retry_after) == expected


def test_workflow_ai_error_code():
    error = WorkflowAIError(
        response=Mock(),
        error=BaseError(
            message="test",
            status_code=404,
            code="object_not_found",
        ),
    )
    assert error.code == "object_not_found"


def test_workflow_ai_error_status_code():
    error = WorkflowAIError(
        response=Mock(),
        error=BaseError(
            message="test",
            status_code=404,
            code="object_not_found",
        ),
    )
    assert error.status_code == 404


def test_workflow_ai_error_message():
    error = WorkflowAIError(
        response=Mock(),
        error=BaseError(
            message="test",
            status_code=404,
            code="object_not_found",
        ),
    )
    assert error.message == "test"


def test_workflow_ai_error_details():
    error = WorkflowAIError(
        response=Mock(),
        error=BaseError(
            message="test",
            status_code=404,
            code="object_not_found",
            details={"test": "test"},
        ),
    )
    assert error.details == {"test": "test"}


def test_invalid_api_key_error():
    error = InvalidAPIKeyError(
        response=Mock(),
        error=BaseError(
            message="test",
            status_code=404,
            code="object_not_found",
        ),
    )
    assert str(error).startswith("❌ Your API key is invalid")
