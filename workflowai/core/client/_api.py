from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, TypeVar, Union, overload

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from workflowai.core._logger import logger
from workflowai.core.domain.errors import BaseError, InvalidAPIKeyError, WorkflowAIError

# A type for return values
_R = TypeVar("_R")
_M = TypeVar("_M", bound=BaseModel)


class APIClient:
    def __init__(self, url: str, api_key: str, source_headers: Optional[dict[str, str]] = None):
        self.url = url
        self.api_key = api_key
        self.source_headers = source_headers or {}

    def _get_url(self, run: bool = False):
        if run:
            return self.url
        return self.url.replace("https://run.", "https://api.")

    @asynccontextmanager
    async def _client(self, run: bool = False):
        if not self.api_key:
            raise InvalidAPIKeyError(
                response=None,
                error=BaseError(message="No API key provided", code="invalid_api_key"),
            )
        source_headers = self.source_headers or {}
        async with httpx.AsyncClient(
            base_url=self._get_url(run),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                **source_headers,
            },
            timeout=180.0,
        ) as client:
            try:
                yield client
            except (httpx.ReadError, httpx.ConnectError) as e:
                raise WorkflowAIError(
                    response=None,
                    error=BaseError(message="Could not read response", code="connection_error"),
                    # We can retry after 10ms
                    retry_after_delay_seconds=0.010,
                ) from e

    async def get(self, path: str, returns: type[_R], query: Union[dict[str, Any], None] = None) -> _R:
        async with self._client() as client:
            response = await client.get(path, params=query)
            await self.raise_for_status(response)
            return TypeAdapter(returns).validate_python(response.json())

    @overload
    async def post(self, path: str, data: BaseModel, returns: type[_R], run: bool = False) -> _R: ...

    @overload
    async def post(self, path: str, data: BaseModel, returns: None = None, run: bool = False) -> None: ...

    async def post(
        self,
        path: str,
        data: BaseModel,
        returns: Optional[type[_R]] = None,
        run: bool = False,
    ) -> Optional[_R]:
        async with self._client(run) as client:
            response = await client.post(
                path,
                content=data.model_dump_json(exclude_none=True),
                headers={"Content-Type": "application/json"},
            )
            await self.raise_for_status(response)
            if not returns:
                return None
            return TypeAdapter(returns).validate_python(response.json())

    @overload
    async def patch(self, path: str, data: BaseModel, returns: type[_R]) -> _R: ...

    @overload
    async def patch(self, path: str, data: BaseModel) -> None: ...

    async def patch(
        self,
        path: str,
        data: BaseModel,
        returns: Optional[type[_R]] = None,
    ) -> Optional[_R]:
        async with self._client() as client:
            response = await client.patch(
                path,
                content=data.model_dump_json(exclude_none=True),
                headers={"Content-Type": "application/json"},
            )
            await self.raise_for_status(response)
            if not returns:
                return None
            return TypeAdapter(returns).validate_python(response.json())

    async def delete(self, path: str) -> None:
        async with self._client() as client:
            response = await client.delete(path)
            await self.raise_for_status(response)

    async def _wrap_sse(self, raw: AsyncIterator[bytes], termination_chars: bytes = b"\n\n"):
        data = b""
        in_data = False
        async for chunk in raw:
            data += chunk
            if not in_data:
                if data.startswith(b"data: "):
                    data = data[6:]
                    in_data = True
                else:
                    # We will wait for the next chunk, we might be in the middle
                    # of 'data: '
                    continue

            # Splitting the chunk by separator
            splits = data.split(b"\n\ndata: ")
            if len(splits) > 1:
                # Yielding the rest of the splits except the last one
                for data in splits[0:-1]:
                    yield data
                # The last split could be incomplete
                data = splits[-1]

            if data.endswith(termination_chars):
                yield data[: -len(termination_chars)]
                data = b""
                in_data = False

        if data:
            logger.warning("Data left after processing", extra={"data": data})

    async def stream(
        self,
        method: Literal["GET", "POST"],
        path: str,
        data: BaseModel,
        returns: type[_M],
        run: bool = False,
    ) -> AsyncIterator[_M]:
        async with (
            self._client(run=run) as client,
            client.stream(
                method,
                path,
                content=data.model_dump_json(exclude_none=True),
                headers={"Content-Type": "application/json"},
            ) as response,
        ):
            if not response.is_success:
                # We need to read the response to get the error message
                await response.aread()
                await self.raise_for_status(response)
                return

            async for chunk in self._wrap_sse(response.aiter_bytes()):
                try:
                    yield returns.model_validate_json(chunk)
                except ValidationError as e:
                    raise WorkflowAIError.from_response(response, chunk) from e

    async def raise_for_status(self, response: httpx.Response):
        if response.status_code < 200 or response.status_code >= 300:
            raise WorkflowAIError.from_response(response) from None
