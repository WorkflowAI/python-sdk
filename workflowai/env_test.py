import importlib
import os
from typing import Optional
from unittest.mock import patch

import pytest

from workflowai import env


# Check what happens when the environment is fully empty
@patch.dict(os.environ, clear=True)
def test_default_app_url_clear_env():
    # We need to reload the env module so that the environment variables are read again
    importlib.reload(env)
    assert env.WORKFLOWAI_API_URL == "https://run.workflowai.com"
    assert env.WORKFLOWAI_APP_URL == "https://workflowai.com"


# Check with the default app url when an api url is provided
@patch.dict(os.environ, {"WORKFLOWAI_API_URL": "https://run.workflowai.dev"}, clear=True)
def test_default_app_url_dev_url():
    importlib.reload(env)
    assert env.WORKFLOWAI_API_URL == "https://run.workflowai.dev"
    assert env.WORKFLOWAI_APP_URL == "https://workflowai.dev"


# Check the app url when both api url and app url are provided
@patch.dict(
    os.environ,
    {"WORKFLOWAI_API_URL": "https://run.workflowai.dev", "WORKFLOWAI_APP_URL": "https://workflowai.app"},
    clear=True,
)
def test_with_app_url():
    importlib.reload(env)
    assert env.WORKFLOWAI_API_URL == "https://run.workflowai.dev"
    assert env.WORKFLOWAI_APP_URL == "https://workflowai.app"


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("https://run.workflowai.com", "https://workflowai.com"),
        ("https://api.workflowai.com", "https://workflowai.com"),
        ("https://workflowai.com", "https://workflowai.com"),
        (None, "https://workflowai.com"),
    ],
)
def test_default_app_url(api_url: Optional[str], expected: str):
    # Importing here to avoid setting the environment variables before the test
    with patch("workflowai.env.WORKFLOWAI_API_URL", api_url):
        assert env._default_app_url() == expected  # pyright: ignore[reportPrivateUsage]
