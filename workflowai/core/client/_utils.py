# Sometimes, 2 payloads are sent in a single message.
# By adding the " at the end we more or less guarantee that
# the delimiter is not withing a quoted string
import asyncio
import os
import re
from collections.abc import Mapping
from json import JSONDecodeError
from time import time
from typing import Any, NamedTuple, Optional, Union

from typing_extensions import Self

from workflowai.core._common_types import OutputValidator
from workflowai.core._logger import logger
from workflowai.core.domain.errors import BaseError, WorkflowAIError
from workflowai.core.domain.task import AgentOutput
from workflowai.core.domain.version_properties import VersionProperties
from workflowai.core.domain.version_reference import VersionReference
from workflowai.core.utils._pydantic import partial_model

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


def default_validator(m: type[AgentOutput]) -> OutputValidator[AgentOutput]:
    partial_cls = partial_model(m)

    def _validator(data: dict[str, Any], partial: bool) -> AgentOutput:
        model_cls = partial_cls if partial else m
        return model_cls.model_validate(data)

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


class ModelInstructionTemperature(NamedTuple):
    """A combination of run properties, with useful method
    for combination"""

    model: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        return cls(
            model=d.get("model"),
            instructions=d.get("instructions"),
            temperature=d.get("temperature"),
        )

    @classmethod
    def from_version(cls, version: Union[int, str, VersionProperties, None]):
        if isinstance(version, VersionProperties):
            return cls(
                model=version.model,
                instructions=version.instructions,
                temperature=version.temperature,
            )
        return cls()

    @classmethod
    def combine(cls, *args: Self):
        return cls(
            model=next((a.model for a in args if a.model is not None), None),
            instructions=next((a.instructions for a in args if a.instructions is not None), None),
            temperature=next((a.temperature for a in args if a.temperature is not None), None),
        )
