from collections.abc import Callable, Iterable
from typing import Any, Optional

from typing_extensions import deprecated

from workflowai import env
from workflowai.core.client._types import AgentDecorator
from workflowai.core.client.client import WorkflowAI as WorkflowAI
from workflowai.core.domain import model
from workflowai.core.domain.cache_usage import CacheUsage as CacheUsage
from workflowai.core.domain.errors import WorkflowAIError as WorkflowAIError
from workflowai.core.domain.model import Model as Model
from workflowai.core.domain.run import Run as Run
from workflowai.core.domain.version import Version as Version
from workflowai.core.domain.version_properties import VersionProperties as VersionProperties
from workflowai.core.domain.version_reference import (
    VersionReference as VersionReference,
)


def _build_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    default_version: Optional[VersionReference] = None,
):
    return WorkflowAI(
        url=url or env.WORKFLOWAI_API_URL,
        api_key=api_key or env.WORKFLOWAI_API_KEY,
        default_version=default_version,
    )


# By default the shared client is created using the default environment variables
shared_client: WorkflowAI = _build_client()

# The default model to use when running agents without a deployment
DEFAULT_MODEL: "model.ModelOrStr" = env.WORKFLOWAI_DEFAULT_MODEL


def init(api_key: Optional[str] = None, url: Optional[str] = None, default_version: Optional[VersionReference] = None):
    """Create a new workflowai client

    Args:
        url (Optional[str], optional): The API url to use.
            If not provided, the env variable WORKFLOWAI_API_URL is used. Otherwise defaults to https://api.workflowai.com
        api_key (Optional[str], optional): _description_. If not provided, the env variable WORKFLOWAI_API_KEY is used.
        default_version (Optional[VersionReference], optional): The default version to use for tasks. If not provided,
            the env var 'WORKFLOWAI_DEFAULT_VERSION' is used. Defaults to production.
    Returns:
        client.Client: a client instance
    """

    global shared_client  # noqa: PLW0603
    shared_client = _build_client(url, api_key, default_version)


@deprecated("Use `workflowai.agent` instead")
def task(
    schema_id: Optional[int] = None,
    task_id: Optional[str] = None,
    version: Optional[VersionReference] = None,
):
    from workflowai.core.client._fn_utils import agent_wrapper

    return agent_wrapper(lambda: shared_client.api, schema_id, task_id, version)


def agent(
    id: Optional[str] = None,  # noqa: A002
    schema_id: Optional[int] = None,
    version: Optional[VersionReference] = None,
    model: Optional["model.ModelOrStr"] = None,
    tools: Optional[Iterable[Callable[..., Any]]] = None,
) -> AgentDecorator:
    from workflowai.core.client._fn_utils import agent_wrapper

    return agent_wrapper(
        lambda: shared_client.api,
        schema_id=schema_id,
        agent_id=id,
        version=version,
        model=model,
        tools=tools,
    )
