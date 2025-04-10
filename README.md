![header](/examples/assets/readme-header.png)

# Python SDK for WorkflowAI

[![PyPI version](https://img.shields.io/pypi/v/workflowai.svg)](https://pypi.org/project/workflowai/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python versions](https://img.shields.io/pypi/pyversions/workflowai.svg)](https://pypi.org/project/workflowai/)

Official SDK from [WorkflowAI](https://workflowai.com) for Python.

This SDK is designed for Python teams who prefer code-first development. It provides greater control through direct code integration while still leveraging the full power of the WorkflowAI platform, complementing the web-app experience.

#### Try in CursorAI:
```
install `pip workflowai` and from https://docs.workflowai.com/python-sdk/agent build an agent that...
```

https://github.com/user-attachments/assets/634c1100-f354-46bc-9aee-92c3f2044cd6

## Key Features

- **Model-agnostic**: Works with all major AI models including OpenAI, Anthropic, Claude, Google/Gemini, Mistral, Deepseek, with a unified interface that makes switching between providers seamless. [View all supported models](https://github.com/WorkflowAI/python-sdk/blob/main/workflowai/core/domain/model.py).

https://github.com/user-attachments/assets/7259adee-1152-44a4-9a15-78fc0a5935e1

- **Open-source and flexible deployment**: WorkflowAI is fully open-source with flexible deployment options. Run it self-hosted on your own infrastructure for maximum data control, or use the managed [WorkflowAI Cloud](https://docs.workflowai.com/workflowai-cloud/introduction) service for hassle-free updates and automatic scaling.

- **Structured output**: Uses Pydantic models to validate and structure AI responses. WorkflowAI ensures your AI responses always match your defined structure, simplifying integrations, reducing parsing errors, and making your data reliable and ready for use. Learn more about [structured input and output](https://docs.workflowai.com/python-sdk/agent#schema-input-output).

https://github.com/user-attachments/assets/0d05bf43-abdb-48fa-b96f-a6c8917c5479

- **Observability integrated**: Built-in monitoring and logging capabilities that provide insights into your AI workflows, making debugging and optimization straightforward. Learn more about [observability features](https://docs.workflowai.com/concepts/runs).

https://github.com/user-attachments/assets/7bc99d61-5c49-4c65-9cf2-36c1c9415559

- **Streaming supported**: Enables real-time streaming of AI responses for low latency applications, with immediate validation of partial outputs. Learn more about [streaming capabilities](https://docs.workflowai.com/python-sdk/agent#streaming).

https://github.com/user-attachments/assets/bcb52412-4dcb-45f8-b812-4275824ed543

- **Provider fallback**: Automatically switches to alternative AI providers when the primary provider fails, ensuring high availability and reliability for your AI applications. This feature allows you to define fallback strategies that maintain service continuity even during provider outages or rate limiting.

![provider-fallback](https://github.com/user-attachments/assets/cc493e94-1249-4516-b8d7-b78de7d24eb3)

- **Built-in tools**: Comes with powerful built-in tools like web search and web browsing capabilities, allowing your agents to access real-time information from the internet. These tools enable your AI applications to retrieve up-to-date data, research topics, and interact with web content without requiring complex integrations. Learn more about [built-in tools](https://docs.workflowai.com/python-sdk/tools).

https://github.com/user-attachments/assets/9e1cabd1-8d1f-4cec-bad5-64871d7f033f

- **Custom tools support**: Easily extend your agents' capabilities by creating custom tools tailored to your specific needs. Whether you need to query internal databases, call external APIs, or perform specialized calculations, WorkflowAI's tool framework makes it simple to augment your AI with domain-specific functionality. Learn more about [custom tools](https://docs.workflowai.com/python-sdk/tools#defining-custom-tools).

```python
# Sync tool
def get_current_time(timezone: Annotated[str, "The timezone to get the current time in. e-g Europe/Paris"]) -> str:
    """Return the current time in the given timezone in iso format"""
    return datetime.now(ZoneInfo(timezone)).isoformat()

# Tools can also be async
async def get_latest_pip_version(package_name: Annotated[str, "The name of the pip package to check"]) -> str:
    """Fetch the latest version of a pip package from PyPI"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return data['info']['version']

@workflowai.agent(
    id="research-helper",
    tools=[get_current_time, get_latest_pip_version],
    model=Model.GPT_4O_LATEST,
)
async def answer_question(_: AnswerQuestionInput) -> AnswerQuestionOutput:
    ...
```

- **Integrated with WorkflowAI**: The SDK seamlessly syncs with the WorkflowAI web application, giving you access to a powerful playground where you can edit prompts and compare models side-by-side. This hybrid approach combines the flexibility of code-first development with the visual tools needed for effective prompt engineering and model evaluation.

- **Multimodality support**: Build agents that can handle multiple modalities, such as images, PDFs, documents, and audio. Learn more about [multimodal capabilities](https://docs.workflowai.com/python-sdk/multimodality).

https://github.com/user-attachments/assets/65d0f34e-2bb7-42bf-ab5c-be1cca96a2c6

- **Caching support**: To save money and improve latency, WorkflowAI supports caching. When enabled, identical requests return cached results instead of making new API calls to AI providers. Learn more about [caching capabilities](https://docs.workflowai.com/python-sdk/agent#cache).

- **Cost tracking**: Automatically calculates and tracks the cost of each AI model run, providing transparency and helping you manage your AI budget effectively. Learn more about [cost tracking](https://docs.workflowai.com/python-sdk/agent#cost-latency).

## Get Started

`workflowai` requires Python 3.9 or higher.

```sh
pip install workflowai
```

### API Key

To get started quickly, get an API key from [WorkflowAI Cloud](https://workflowai.com/organization/settings/api-keys). For maximum control over your data, you can also use your [self-hosted instance](https://github.com/WorkflowAI/workflowai), though this requires additional setup time.

Then, set the `WORKFLOWAI_API_KEY` environment variable:

```sh
export WORKFLOWAI_API_KEY="your-api-key"
```

### First Agent

Here's a simple example of a WorkflowAI agent that extracts structured flight information from email content:


```python
import asyncio
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model

# Input class
class EmailInput(BaseModel):
    email_content: str

# Output class
class FlightInfo(BaseModel):
    # Enum for standardizing flight status values
    class Status(str, Enum):
        """Possible statuses for a flight booking."""
        CONFIRMED = "Confirmed"
        PENDING = "Pending"
        CANCELLED = "Cancelled"
        DELAYED = "Delayed"
        COMPLETED = "Completed"

    passenger: str
    airline: str
    flight_number: str
    from_airport: str = Field(description="Three-letter IATA airport code for departure")
    to_airport: str = Field(description="Three-letter IATA airport code for arrival")
    departure: datetime
    arrival: datetime
    status: Status

# Agent definition
@workflowai.agent(
    id="flight-info-extractor",
    model=Model.GEMINI_2_0_FLASH_LATEST,
)
async def extract_flight_info(email_input: EmailInput) -> FlightInfo:
    # Agent prompt
    """
    Extract flight information from an email containing booking details.
    """
    ...


async def main():
    email = """
    Dear Jane Smith,

    Your flight booking has been confirmed. Here are your flight details:

    Flight: UA789
    From: SFO
    To: JFK
    Departure: 2024-03-25 9:00 AM
    Arrival: 2024-03-25 5:15 PM
    Booking Reference: XYZ789

    Total Journey Time: 8 hours 15 minutes
    Status: Confirmed

    Thank you for choosing United Airlines!
    """
    run = await extract_flight_info.run(EmailInput(email_content=email))
    print(run)


if __name__ == "__main__":
    asyncio.run(main())


# Output:
# ==================================================
# {
#   "passenger": "Jane Smith",
#   "airline": "United Airlines",
#   "flight_number": "UA789",
#   "from_airport": "SFO",
#   "to_airport": "JFK",
#   "departure": "2024-03-25T09:00:00",
#   "arrival": "2024-03-25T17:15:00",
#   "status": "Confirmed"
# }
# ==================================================
# Cost: $ 0.00009
# Latency: 1.18s
# URL: https://workflowai.com/_/agents/flight-info-extractor/runs/0195ee02-bdc3-72b6-0e0b-671f0b22b3dc
```
> **Ready to run!** This example works straight out of the box - no tweaking needed.

Agents built with `workflowai` SDK can be run in the [WorkflowAI web application](https://workflowai.com/docs/agents/flight-info-extractor/1?showDiffMode=false&show2ColumnLayout=false&taskRunId1=0195ee21-988e-7309-eb32-cd49a9b90f46&taskRunId2=0195ee21-9898-723a-0469-1458a180d3b0&taskRunId3=0195ee21-9892-72f1-ca2d-c29e18285073&versionId=fb7b29cd00031675d0c19e3d09852b27) too.

[![WorkflowAI Playground](/examples/assets/web/playground-flight-info-extractor.png)](https://workflowai.com/docs/agents/flight-info-extractor/1?showDiffMode=false&show2ColumnLayout=false&taskRunId1=0195ee21-988e-7309-eb32-cd49a9b90f46&taskRunId2=0195ee21-9898-723a-0469-1458a180d3b0&taskRunId3=0195ee21-9892-72f1-ca2d-c29e18285073&versionId=fb7b29cd00031675d0c19e3d09852b27)

And the runs executed via the SDK are synced with the web application.

[![WorkflowAI Runs](/examples/assets/web/runs-flight-info-extractor.png)](https://workflowai.com/docs/agents/flight-info-extractor/1/runs?page=0)

## Documentation

Complete documentation is available at [docs.workflowai.com/python-sdk](https://docs.workflowai.com/python-sdk).

## Examples

- [01_basic_agent.py](./examples/01_basic_agent.py): Demonstrates basic agent creation, input/output models, and cost/latency tracking.
- [02_agent_with_tools.py](./examples/02_agent_with_tools.py): Shows how to use hosted tools (like `@browser-text`) and custom tools with an agent.
- [03_caching.py](./examples/03_caching.py): Illustrates different caching strategies (`auto`, `always`, `never`) for agent runs.
- [04_audio_classifier_agent.py](./examples/04_audio_classifier_agent.py): An agent that analyzes audio files for spam/robocall detection using audio input.
- [05_browser_text_uptime_agent.py](./examples/05_browser_text_uptime_agent.py): Uses the `@browser-text` tool to fetch and extract information from web pages.
- [06_streaming_summary.py](./examples/06_streaming_summary.py): Demonstrates how to stream agent responses in real-time.
- [07_image_agent.py](./examples/07_image_agent.py): An agent that analyzes images to identify cities and landmarks.
- [08_pdf_agent.py](./examples/08_pdf_agent.py): An agent that answers questions based on the content of a PDF document.
- [09_reply.py](./examples/09_reply.py): Shows how to use the `run.reply()` method to have a conversation with an agent, maintaining context.
- [10_calendar_event_extraction.py](./examples/10_calendar_event_extraction.py): Extracts structured calendar event details from text or images.
- [11_ecommerce_chatbot.py](./examples/11_ecommerce_chatbot.py): A chatbot that provides product recommendations based on user queries.
- [12_contextual_retrieval.py](./examples/12_contextual_retrieval.py): Generates concise contextual descriptions for document chunks to improve search retrieval.
- [13_rag.py](./examples/13_rag.py): Demonstrates a RAG (Retrieval-Augmented Generation) pattern using a search tool to answer questions based on a knowledge base.
- [14_templated_instructions.py](./examples/14_templated_instructions.py): Uses Jinja2 templating in agent instructions to adapt behavior based on input variables.
- [15_pii_extraction.py](./examples/15_pii_extraction.py): Extracts and redacts Personal Identifiable Information (PII) from text.
- [15_text_to_sql.py](./examples/15_text_to_sql.py): Converts natural language questions into safe and efficient SQL queries based on a provided database schema.
- [16_multi_model_consensus.py](./examples/16_multi_model_consensus.py): Queries multiple LLMs with the same question and uses another LLM to synthesize a combined answer.
- [17_multi_model_consensus_with_tools.py](./examples/17_multi_model_consensus_with_tools.py): An advanced multi-model consensus agent that uses tools to dynamically decide which models to query.
- [18_flight_info_extraction.py](./examples/18_flight_info_extraction.py): Extracts structured flight information (number, dates, times, airports) from emails.
- [workflows/](./examples/workflows): Contains examples of different workflow patterns (chaining, routing, parallel, orchestrator-worker). See [workflows/README.md](./examples/workflows/README.md) for details.

## Workflows

For advanced workflow patterns and examples, please refer to the [Workflows README](examples/workflows/README.md) for more details.

## Contributing

See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more details. Thank you!

## Acknowledgments

Thanks to [ell](https://github.com/MadcowD/ell) for the inspiration! ✨