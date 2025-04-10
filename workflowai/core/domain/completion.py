from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class CompletionUsage(BaseModel):
    """Usage information for a completion."""

    completion_token_count: Optional[int] = None
    completion_cost_usd: Optional[float] = None
    reasoning_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    prompt_token_count_cached: Optional[int] = None
    prompt_cost_usd: Optional[float] = None
    prompt_audio_token_count: Optional[int] = None
    prompt_audio_duration_seconds: Optional[float] = None
    prompt_image_count: Optional[int] = None
    model_context_window_size: Optional[int] = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class DocumentURL(BaseModel):
    url: str


class DocumentContent(BaseModel):
    type: Literal["document_url"] = "document_url"
    source: DocumentURL


class ImageURL(BaseModel):
    url: str


class AudioURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class AudioContent(BaseModel):
    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioURL


class ToolCallRequest(BaseModel):
    type: Literal["tool_call_request"] = "tool_call_request"
    id: Union[str, None] = None
    tool_name: str
    tool_input_dict: Union[dict[str, Any], None] = None


class ToolCallResult(BaseModel):
    type: Literal["tool_call_result"] = "tool_call_result"
    id: Union[str, None] = None
    tool_name: Union[str, None] = None
    tool_input_dict: Union[dict[str, Any], None] = None
    result: Union[Any, None] = None
    error: Union[str, None] = None


MessageContent = Annotated[Union[TextContent, DocumentContent, ImageContent, AudioContent], Field(discriminator="type")]


class Message(BaseModel):
    """A message in a completion."""

    role: str = ""
    content: Union[str, MessageContent] = Field(default="")


class Completion(BaseModel):
    """A completion from the model."""

    messages: list[Message] = Field(default_factory=list)
    response: Optional[str] = Field(default=None)
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
