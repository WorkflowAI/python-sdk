from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from workflowai.core.client._utils import (
    build_retryable_wait,
    global_default_version_reference,
    split_chunks,
    tolerant_validator,
)
from workflowai.core.domain.errors import BaseError, WorkflowAIError


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (b'data: {"foo": "bar"}\n\ndata: {"foo": "baz"}', ['{"foo": "bar"}', '{"foo": "baz"}']),
        (
            b'data: {"foo": "bar"}\n\ndata: {"foo": "baz"}\n\ndata: {"foo": "qux"}',
            ['{"foo": "bar"}', '{"foo": "baz"}', '{"foo": "qux"}'],
        ),
    ],
)
def test_split_chunks(chunk: bytes, expected: list[bytes]):
    assert list(split_chunks(chunk)) == expected


class TestBuildRetryableWait:
    @pytest.fixture
    def request_error(self):
        response = Mock()
        response.headers = {"Retry-After": "0.01"}
        return WorkflowAIError(response=response, error=BaseError(message=""))

    async def test_should_retry_count(self, request_error: WorkflowAIError):
        should_retry, wait_for_exception = build_retryable_wait(60, 1)
        assert should_retry()
        await wait_for_exception(request_error)
        assert not should_retry()


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [("p", "production"), ("production", "production"), ("dev", "dev"), ("staging", "staging"), ("1", 1)],
)
def test_global_default_version_reference(env_var: str, expected: Any):
    with patch.dict("os.environ", {"WORKFLOWAI_DEFAULT_VERSION": env_var}):
        assert global_default_version_reference() == expected


# Create a nested object with only required properties
class Recipe(BaseModel):
    class Ingredient(BaseModel):
        name: str
        quantity: int

    ingredients: list[Ingredient]


class TestTolerantValidator:
    def test_tolerant_validator_nested_object(self):
        validated = tolerant_validator(Recipe)(
            {
                "ingredients": [{"name": "salt"}],
            },
            has_tool_call_requests=False,
        )
        for ingredient in validated.ingredients:
            assert isinstance(ingredient, Recipe.Ingredient)
