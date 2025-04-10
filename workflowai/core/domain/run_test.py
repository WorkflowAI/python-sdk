from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from workflowai.core.domain.completion import Completion, CompletionUsage, Message
from workflowai.core.domain.run import (
    Run,
    _AgentBase,  # pyright: ignore [reportPrivateUsage]
)
from workflowai.core.domain.tool_call import ToolCallRequest
from workflowai.core.domain.version import Version
from workflowai.core.domain.version_properties import VersionProperties


class _TestOutput(BaseModel):
    message: str


class _TestOutputWithDatetime(BaseModel):
    timestamp: datetime
    message: str


@pytest.fixture
def mock_agent() -> Mock:
    mock = Mock(spec=_AgentBase)
    return mock


@pytest.fixture
def run1(mock_agent: Mock) -> Run[_TestOutput]:
    run = Run[_TestOutput](
        id="run-id",
        agent_id="agent-id",
        schema_id=1,
        output=_TestOutput(message="test output"),
        duration_seconds=1.0,
        cost_usd=0.1,
        version=Version(properties=VersionProperties()),
        metadata={"test": "data"},
        tool_calls=[],
        tool_call_requests=[],
    )
    run._agent = mock_agent  # pyright: ignore [reportPrivateUsage]
    return run


@pytest.fixture
def run2(run1: Run[_TestOutput]) -> Run[_TestOutput]:
    return run1.model_copy(deep=True)


class TestRunEquality:
    def test_identical(self, run1: Run[_TestOutput], run2: Run[_TestOutput]):
        assert run1 == run2

    def test_different_output(self, run1: Run[_TestOutput], run2: Run[_TestOutput]):
        run2.output.message = "different output"
        assert run1 != run2

    def test_different_agents(self, run1: Run[_TestOutput], run2: Run[_TestOutput]):
        run2._agent = Mock()  # pyright: ignore [reportPrivateUsage]
        assert run1._agent != run2._agent, "sanity check"  # pyright: ignore [reportPrivateUsage]
        assert run1 == run2


class TestRunFormatOutput:
    @pytest.fixture(autouse=True)
    def mock_app_url(self):
        with patch("workflowai.env.WORKFLOWAI_APP_URL", "https://workflowai.hello") as mock:
            yield mock

    # Test that format_output correctly formats:
    # 1. The output as a JSON object
    # 2. The cost with $ prefix and correct precision
    # 3. The latency with 2 decimal places and 's' suffix
    # 4. The run URL

    def test_format_output_full(self):
        run = Run[_TestOutput](
            id="run-id",
            agent_id="agent-id",
            schema_id=1,
            output=_TestOutput(message="hello"),
            duration_seconds=1.23,
            cost_usd=0.001,
        )

        expected = """\nOutput:
==================================================
{
  "message": "hello"
}
==================================================
Cost: $ 0.00100
Latency: 1.23s
URL: https://workflowai.hello/_/agents/agent-id/runs/run-id"""

        assert run.format_output() == expected

    def test_format_output_very_low_cost(self):
        run = Run[_TestOutput](
            id="run-id",
            agent_id="agent-id",
            schema_id=1,
            output=_TestOutput(message="hello"),
            duration_seconds=1.23,
            cost_usd=4.97625e-05,
        )

        expected = """\nOutput:
==================================================
{
  "message": "hello"
}
==================================================
Cost: $ 0.00005
Latency: 1.23s
URL: https://workflowai.hello/_/agents/agent-id/runs/run-id"""

        assert run.format_output() == expected

    # Test that format_output works correctly when cost and latency are not provided:
    # 1. The output is still formatted as a JSON object
    # 2. No cost or latency lines are included in the output
    # 3. The run URL is still included
    def test_format_output_no_cost_latency(self):
        run = Run[_TestOutput](
            id="run-id",
            agent_id="agent-id",
            schema_id=1,
            output=_TestOutput(message="hello"),
        )

        expected = """\nOutput:
==================================================
{
  "message": "hello"
}
==================================================
URL: https://workflowai.hello/_/agents/agent-id/runs/run-id"""

        assert run.format_output() == expected

    def test_format_output_tool_call_requests(self):
        run = Run[_TestOutput](
            id="run-id",
            agent_id="agent-id",
            schema_id=1,
            output=_TestOutput.model_construct(),
            tool_call_requests=[
                ToolCallRequest(
                    id="tool-call-id",
                    name="tool-call-name",
                    input={"key": "value"},
                ),
            ],
        )
        assert (
            run.format_output()
            == """\nTool Call Requests:
==================================================
[
  {
    "id": "tool-call-id",
    "name": "tool-call-name",
    "input": {
      "key": "value"
    }
  }
]
==================================================
URL: https://workflowai.hello/_/agents/agent-id/runs/run-id"""
        )

    def test_format_output_with_datetime(self):
        """Test that datetimes in the output model are correctly serialized to ISO strings."""
        test_dt = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        run = Run[_TestOutputWithDatetime](
            id="run-dt-id",
            agent_id="agent-dt-id",
            schema_id=2,
            output=_TestOutputWithDatetime(timestamp=test_dt, message="datetime test"),
            duration_seconds=0.5,
            cost_usd=0.0001,
        )

        expected_json_part = '{\n  "timestamp": "2024-01-01T12:30:00Z",\n  "message": "datetime test"\n}'
        expected = f"""\nOutput:
==================================================
{expected_json_part}
==================================================
Cost: $ 0.00010
Latency: 0.50s
URL: https://workflowai.hello/_/agents/agent-dt-id/runs/run-dt-id"""

        assert run.format_output() == expected


class TestRunURL:
    # The @patch decorator from unittest.mock temporarily replaces the value of an attribute
    # during the execution of the decorated test function. The original value is restored
    # after the test completes.

    # To check what happens in different environemnt configurations, see env_test.py

    # Here we patch WORKFLOWAI_APP_URL to test the direct app URL case
    @patch("workflowai.env.WORKFLOWAI_APP_URL", "https://workflowai.hello")
    def test_run_url(self, run1: Run[_TestOutput]):
        # The patched value is only active during this test method
        assert run1.run_url == "https://workflowai.hello/_/agents/agent-id/runs/run-id"


class TestFetchCompletions:
    """Tests for the fetch_completions method of the Run class."""

    # Test that the underlying agent is called with the proper run id
    async def test_fetch_completions_success(self, run1: Run[_TestOutput], mock_agent: Mock):
        mock_agent.fetch_completions.return_value = [
            Completion(
                messages=[
                    Message(role="system", content="You are a helpful assistant"),
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi there!"),
                ],
                response="Hi there!",
                usage=CompletionUsage(
                    completion_token_count=3,
                    completion_cost_usd=0.001,
                    reasoning_token_count=10,
                    prompt_token_count=20,
                    prompt_token_count_cached=0,
                    prompt_cost_usd=0.002,
                    prompt_audio_token_count=0,
                    prompt_audio_duration_seconds=0,
                    prompt_image_count=0,
                    model_context_window_size=32000,
                ),
            ),
        ]

        # Call fetch_completions
        completions = await run1.fetch_completions()

        # Verify the API was called correctly
        mock_agent.fetch_completions.assert_called_once_with("run-id")

        # Verify the response
        assert len(completions) == 1
        completion = completions[0]
        assert len(completion.messages) == 3
        assert completion.messages[0].role == "system"
        assert completion.messages[0].content == "You are a helpful assistant"
        assert completion.response == "Hi there!"
        assert completion.usage.completion_token_count == 3
        assert completion.usage.completion_cost_usd == 0.001

    # Test that fetch_completions fails appropriately when the agent is not set:
    # 1. This is a common error case that occurs when a Run object is created without an agent
    # 2. The method should fail fast with a clear error message before attempting any API calls
    # 3. This protects users from confusing errors that would occur if we tried to use the API client
    async def test_fetch_completions_no_agent(self, run1: Run[_TestOutput]):
        run1._agent = None  # pyright: ignore [reportPrivateUsage]
        with pytest.raises(ValueError, match="Agent is not set"):
            await run1.fetch_completions()

    # Test that fetch_completions fails appropriately when the run ID is not set:
    # 1. The run ID is required to construct the API endpoint URL
    # 2. Without it, we can't make a valid API request
    # 3. This validates that we fail fast with a clear error message
    # 4. This should never happen in practice (as Run objects always have an ID),
    #    but we test it for completeness and to ensure robust error handling
    async def test_fetch_completions_no_id(self, run1: Run[_TestOutput]):
        mock_agent = Mock()
        run1._agent = mock_agent  # pyright: ignore [reportPrivateUsage]
        run1.id = ""  # Empty ID
        with pytest.raises(ValueError, match="Run id is not set"):
            await run1.fetch_completions()
