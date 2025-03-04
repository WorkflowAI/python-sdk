from typing import Optional

from pydantic import BaseModel, Field

import workflowai
from workflowai.core.client.agent import Agent


class SummarizeTaskInput(BaseModel):
    text: Optional[str] = None


class SummarizeTaskOutput(BaseModel):
    summary_points: Optional[list[str]] = None


@workflowai.agent(id="summarize", model="gemini-1.5-flash-latest")
async def summarize(_: SummarizeTaskInput) -> SummarizeTaskOutput: ...


async def test_summarize():
    summarized = await summarize(
        SummarizeTaskInput(
            text="""The first computer programmer was Ada Lovelace. She wrote the first algorithm
intended to be processed by a machine in the 1840s. Her work was on Charles Babbage's
proposed mechanical computer, the Analytical Engine. She is celebrated annually on Ada
Lovelace Day, which promotes women in science and technology.""",
        ),
        use_cache="never",
    )
    assert summarized.summary_points


async def test_same_schema():
    class InputWithNullableList(BaseModel):
        opt_list: Optional[list[str]] = None

    class InputWithNonNullableList(BaseModel):
        opt_list: list[str] = Field(default_factory=list)

    agent1 = Agent(
        agent_id="summarize",
        input_cls=InputWithNullableList,
        output_cls=SummarizeTaskOutput,
        api=lambda: workflowai.shared_client.api,
    )

    schema_id1 = await agent1.register()

    agent2 = Agent(
        agent_id="summarize",
        input_cls=InputWithNonNullableList,
        output_cls=SummarizeTaskOutput,
        api=lambda: workflowai.shared_client.api,
    )

    schema_id2 = await agent2.register()

    assert schema_id1 == schema_id2
