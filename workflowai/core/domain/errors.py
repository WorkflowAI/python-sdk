from email.utils import parsedate_to_datetime
from json import JSONDecodeError
from time import time
from typing import Any, Literal, Optional, Union

from httpx import Response
from pydantic import BaseModel

from workflowai.core.domain import tool_call

ProviderErrorCode = Literal[
    # Max number of tokens were exceeded in the prompt
    "max_tokens_exceeded",
    # The model failed to generate a response
    "failed_generation",
    # The model generated a response but it was not valid
    "invalid_generation",
    # The model returned an error that we currently do not handle
    # The returned status code will match the provider status code and the entire
    # provider response will be provided the error details.
    #
    # This error is intended as a fallback since we do not control what the providers
    # return. We track this error on our end and the error should eventually
    # be assigned a different status code
    "unknown_provider_error",
    # The provider returned a rate limit error
    "rate_limit",
    # The provider returned a server overloaded error
    "server_overloaded",
    # The requested provider does not support the model
    "invalid_provider_config",
    # The provider returned a 500
    "provider_internal_error",
    # The provider returned a 502 or 503
    "provider_unavailable",
    # The request timed out
    "read_timeout",
    # The requested model does not support the requested generation mode
    # (e-g a model that does not support images generation was sent an image)
    "model_does_not_support_mode",
    # Invalid file provided
    "invalid_file",
    # The maximum number of tool call iterations was reached
    "max_tool_call_iteration",
    # The current configuration does not support structured generation
    "structured_generation_error",
    # The content was moderated
    "content_moderation",
    # Task banned
    "task_banned",
    # The request timed out
    "timeout",
    # Agent run failed
    "agent_run_failed",
]

ErrorCode = Union[
    ProviderErrorCode,
    Literal[
        # The object was not found
        "object_not_found",
        # There are no configured providers supporting the requested model
        # This error will never happen when using WorkflowAI keys
        "no_provider_supporting_model",
        # The requested provider does not support the model
        "provider_does_not_support_model",
        # Run properties are invalid, for example the model does not exist
        "invalid_run_properties",
        # An internal error occurred
        "internal_error",
        # The request was invalid
        "bad_request",
        "invalid_file",
        "connection_error",
    ],
    str,  # Using as a fallback to avoid validation error if an error code is added to the API
]


class BaseError(BaseModel):
    details: Optional[dict[str, Any]] = None
    message: str
    status_code: Optional[int] = None
    code: Optional[ErrorCode] = None


class ErrorResponse(BaseModel):
    error: BaseError
    id: Optional[str] = None
    task_output: Optional[dict[str, Any]] = None


def _retry_after_to_delay_seconds(retry_after: Any) -> Optional[float]:
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        pass
    try:
        retry_after_date = parsedate_to_datetime(retry_after)
        current_time = time()
        return retry_after_date.timestamp() - current_time
    except (TypeError, ValueError, OverflowError):
        return None


class WorkflowAIError(Exception):
    def __init__(
        self,
        response: Optional[Response],
        error: BaseError,
        run_id: Optional[str] = None,
        retry_after_delay_seconds: Optional[float] = None,
        partial_output: Optional[dict[str, Any]] = None,
        tool_call_requests: Optional[list["tool_call.ToolCallRequest"]] = None,
    ):
        self.error = error
        self.run_id = run_id
        self.response = response
        self._retry_after_delay_seconds = retry_after_delay_seconds
        self.partial_output = partial_output
        self.tool_call_requests = tool_call_requests

    def __str__(self):
        return f"WorkflowAIError : [{self.error.code}] ({self.error.status_code}): [{self.error.message}]"

    @classmethod
    def error_cls(cls, code: str):
        if code == "invalid_generation" or code == "failed_generation" or code == "agent_run_failed":
            return InvalidGenerationError
        return cls

    @classmethod
    def from_response(cls, response: Response):
        try:
            response_json = response.json()
            r_error = response_json.get("error", {})
            error_message = response_json.get("detail", {}) or r_error.get("message", "Unknown Error")
            details = r_error.get("details", {})
            error_code = r_error.get("code", "unknown_error")
            status_code = response.status_code
            run_id = response_json.get("id", None)
            partial_output = response_json.get("task_output", None)
        except JSONDecodeError:
            error_message = "Unknown error"
            details = {"raw": response.content.decode()}
            error_code = "unknown_error"
            status_code = response.status_code
            run_id = None
            partial_output = None

        return cls.error_cls(error_code)(
            response=response,
            error=BaseError(
                message=error_message,
                details=details,
                status_code=status_code,
                code=error_code,
            ),
            run_id=run_id,
            partial_output=partial_output,
        )

    @property
    def retry_after_delay_seconds(self) -> Optional[float]:
        if self._retry_after_delay_seconds:
            return self._retry_after_delay_seconds

        if self.response:
            return _retry_after_to_delay_seconds(self.response.headers.get("Retry-After"))

        return None

    @property
    def code(self) -> Optional[ErrorCode]:
        return self.error.code

    @property
    def status_code(self) -> Optional[int]:
        return self.error.status_code

    @property
    def message(self) -> str:
        return self.error.message

    @property
    def details(self) -> Optional[dict[str, Any]]:
        return self.error.details


class InvalidGenerationError(WorkflowAIError): ...


class MaxTurnsReachedError(WorkflowAIError): ...
