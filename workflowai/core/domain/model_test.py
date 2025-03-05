import httpx

from workflowai.core.domain.model import Model


async def test_model_exhaustive():
    """Make sure the list of models is synchronized with the prod API"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://run.workflowai.com/v1/models")
        response.raise_for_status()
        models: list[str] = response.json()

    # Converting to a set of strings should not be needed
    # but it makes pytest errors prettier
    assert set(models) == {m.value for m in Model}
