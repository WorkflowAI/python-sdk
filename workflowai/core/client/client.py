import importlib.metadata
from typing import (
    Literal,
    Optional,
)

from workflowai.core.client._api import APIClient
from workflowai.core.client._fn_utils import agent_wrapper
from workflowai.core.client._models import CreateFeedbackRequest
from workflowai.core.client._utils import global_default_version_reference
from workflowai.core.domain.version_reference import VersionReference


class WorkflowAI:
    def __init__(
        self,
        api_key: str,
        url: Optional[str] = None,
        default_version: Optional[VersionReference] = None,
    ):
        self.additional_headers = {
            "x-workflowai-source": "sdk",
            "x-workflowai-language": "python",
            "x-workflowai-version": importlib.metadata.version("workflowai"),
        }
        self.api = APIClient(url or "https://run.workflowai.com", api_key, self.additional_headers)
        self.default_version: VersionReference = default_version or global_default_version_reference()

    def task(
        self,
        schema_id: int,
        task_id: Optional[str] = None,
        version: Optional[VersionReference] = None,
    ):
        return agent_wrapper(lambda: self.api, schema_id, agent_id=task_id, version=version)

    def agent(
        self,
        id: Optional[str] = None,  # noqa: A002
        schema_id: Optional[int] = None,
        version: Optional[VersionReference] = None,
    ):
        return agent_wrapper(lambda: self.api, schema_id=schema_id, agent_id=id, version=version)

    async def send_feedback(
        self,
        feedback_token: str,
        outcome: Literal["positive", "negative"],
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        await self.api.post(
            "/v1/feedback",
            CreateFeedbackRequest(feedback_token=feedback_token, outcome=outcome, comment=comment, user_id=user_id),
        )
