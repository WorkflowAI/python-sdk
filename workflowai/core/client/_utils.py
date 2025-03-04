# Sometimes, 2 payloads are sent in a single message.
# By adding the " at the end we more or less guarantee that
# the delimiter is not withing a quoted string
import asyncio
import os
import re
from json import JSONDecodeError
from time import time
from typing import Any

from workflowai.core._common_types import OutputValidator
from workflowai.core._logger import logger
from workflowai.core.domain.errors import BaseError, WorkflowAIError
from workflowai.core.domain.task import AgentOutput
from workflowai.core.domain.version_reference import VersionReference
from workflowai.core.utils._pydantic import construct_model_recursive

delimiter = re.compile(r'\}\n\ndata: \{"')


def split_chunks(chunk: bytes):
    start = 0
    chunk_str = chunk.removeprefix(b"data: ").removesuffix(b"\n\n").decode()
    for match in delimiter.finditer(chunk_str):
        yield chunk_str[start : match.start() + 1]
        start = match.end() - 2
    yield chunk_str[start:]


# Returns two functions:
# - _should_retry: returns True if we should retry
# - _wait_for_exception: waits after an exception only if we should retry, otherwise raises
# This is a bit convoluted and would be better in a function wrapper, but since we are dealing
# with both Awaitable and AsyncGenerator, a wrapper would just be too complex
def build_retryable_wait(
    max_retry_delay: float = 60,
    max_retry_count: float = 1,
):
    now = time()
    retry_count = 0

    def _leftover_delay():
        # Time remaining before we hit the max retry delay
        return max_retry_delay - (time() - now)

    def _should_retry():
        return retry_count < max_retry_count and _leftover_delay() >= 0

    async def _wait_for_exception(e: WorkflowAIError):
        retry_after = e.retry_after_delay_seconds
        if retry_after is None:
            raise e

        nonlocal retry_count
        leftover_delay = _leftover_delay()
        if not retry_after or leftover_delay < 0 or retry_count >= max_retry_count:
            if not e.response:
                raise e

            # Convert error to WorkflowAIError
            try:
                response_json = e.response.json()
                r_err = response_json.get("error", {})
                error_message = response_json.get("detail", {}) or r_err.get("message", "Unknown Error")
                details = r_err.get("details", {})
                error_code = r_err.get("code", "unknown_error")
                status_code = r_err.get("status_code", e.response.status_code)
            except JSONDecodeError:
                error_message = "Unknown error"
                details = {"raw": e.response.content.decode()}
                error_code = "unknown_error"
                status_code = e.response.status_code

            raise WorkflowAIError(
                error=BaseError(
                    message=error_message,
                    details=details,
                    status_code=status_code,
                    code=error_code,
                ),
                response=e.response,
            ) from None

        await asyncio.sleep(retry_after)
        retry_count += 1

    return _should_retry, _wait_for_exception


def tolerant_validator(m: type[AgentOutput]) -> OutputValidator[AgentOutput]:
    def _validator(data: dict[str, Any], has_tool_call_requests: bool) -> AgentOutput:  # noqa: ARG001
        return construct_model_recursive(m, data)

    return _validator


def intolerant_validator(m: type[AgentOutput]) -> OutputValidator[AgentOutput]:
    def _validator(data: dict[str, Any], has_tool_call_requests: bool) -> AgentOutput:
        # When we have tool call requests, the output can be empty
        if has_tool_call_requests:
            return tolerant_validator(m)(data, has_tool_call_requests)

        return m.model_validate(data)

    return _validator


def global_default_version_reference() -> VersionReference:
    version = os.getenv("WORKFLOWAI_DEFAULT_VERSION")
    if not version:
        return "production"

    if version in {"dev", "staging", "production"}:
        return version  # pyright: ignore [reportReturnType]

    try:
        return int(version)
    except ValueError:
        pass

    logger.warning("Invalid default version: %s", version)

    return "production"
