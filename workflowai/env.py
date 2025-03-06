"""A file to describe all environment variables"""

import os

WORKFLOWAI_DEFAULT_MODEL = os.getenv("WORKFLOWAI_DEFAULT_MODEL", "gemini-1.5-pro-latest")

WORKFLOWAI_API_URL = os.getenv("WORKFLOWAI_API_URL", "https://run.workflowai.com")


def _default_app_url():
    if not WORKFLOWAI_API_URL:
        return "https://workflowai.com"
    if WORKFLOWAI_API_URL.startswith("https://run."):
        return "https://" + WORKFLOWAI_API_URL.removeprefix("https://run.")
    if WORKFLOWAI_API_URL.startswith("https://api."):
        return "https://" + WORKFLOWAI_API_URL.removeprefix("https://api.")
    return "https://workflowai.com"


WORKFLOWAI_APP_URL = os.getenv("WORKFLOWAI_APP_URL", _default_app_url())

WORKFLOWAI_API_KEY = os.getenv("WORKFLOWAI_API_KEY", "")
