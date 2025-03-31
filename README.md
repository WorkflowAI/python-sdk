# Python SDK for WorkflowAI

[![PyPI version](https://img.shields.io/pypi/v/workflowai.svg)](https://pypi.org/project/workflowai/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official SDK from [WorkflowAI](https://workflowai.com) for Python.

## WorkflowAI

[WorkflowAI](https://workflowai.com) is an open-source platform where product and engineering teams collaborate to build and iterate on AI features.

This SDK is designed for Python teams who prefer code-first development. It provides greater control through direct code integration while still leveraging the full power of the WorkflowAI platform, complementing the web-app experience.

## Key Features

- **Model-agnostic**: Works with all major AI models including OpenAI, Anthropic, Claude, Google/Gemini, Mistral, Deepseek, with a unified interface that makes switching between providers seamless.

- **Open-source and flexible deployment**: WorkflowAI is fully open-source with flexible deployment options. Run it self-hosted on your own infrastructure for maximum data control, or use the managed WorkflowAI Cloud service for hassle-free updates and automatic scaling.

- **Observability integrated**: Built-in monitoring and logging capabilities that provide insights into your AI workflows, making debugging and optimization straightforward. Learn more about [observability features](https://docs.workflowai.com/concepts/runs).

- **Cost tracking**: Automatically calculates and tracks the cost of each AI model run, providing transparency and helping you manage your AI budget effectively.

- **Type-safe**: Leverages Python's type system to catch errors at development time rather than runtime, ensuring more reliable AI applications.

- **Structured output**: Uses Pydantic models to validate and structure AI responses. WorkflowAI ensures your AI responses always match your defined structure, simplifying integrations, reducing parsing errors, and making your data reliable and ready for use.

- **Streaming supported**: Enables real-time streaming of AI responses for low latency applications, with immediate validation of partial outputs. Learn more about [streaming capabilities](https://docs.workflowai.com/python-sdk/agent#streaming).

- **Provider fallback**: Automatically switches to alternative AI providers when the primary provider fails, ensuring high availability and reliability for your AI applications. This feature allows you to define fallback strategies that maintain service continuity even during provider outages or rate limiting.

- **Built-in tools**: Comes with powerful built-in tools like web search and web browsing capabilities, allowing your agents to access real-time information from the internet. These tools enable your AI applications to retrieve up-to-date data, research topics, and interact with web content without requiring complex integrations.

- **Custom tools support**: Easily extend your agents' capabilities by creating custom tools tailored to your specific needs. Whether you need to query internal databases, call external APIs, or perform specialized calculations, WorkflowAI's tool framework makes it simple to augment your AI with domain-specific functionality.

- **Integrated with WorkflowAI**: The SDK seamlessly syncs with the WorkflowAI web application, giving you access to a powerful playground where you can edit prompts and compare models side-by-side. This hybrid approach combines the flexibility of code-first development with the visual tools needed for effective prompt engineering and model evaluation.

## Get Started

`workflowai` requires Python 3.9 or higher.

```sh
pip install workflowai
```

Here's a simple example of a WorkflowAI agent that extracts structured flight information from email content:


```python
import asyncio
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class EmailInput(BaseModel):
    email_content: str

class FlightInfo(BaseModel):
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

@workflowai.agent(
    id="flight-info-extractor",
    model=Model.GEMINI_2_0_FLASH_LATEST,
)
async def extract_flight_info(email_input: EmailInput) -> FlightInfo:
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

Agents built with `workflowai` SDK can be run in the [WorkflowAI web application](https://workflowai.com/_/agents/flight-info-extractor/runs/0195ee02-bdc3-72b6-0e0b-671f0b22b3dc) too.

![WorkflowAI Playground](/examples/assets/web/playground-flight-info-extractor.png)

## Documentation

Complete documentation is available at [docs.workflowai.com/python-sdk](https://docs.workflowai.com/python-sdk).

## Example

Examples are available in the [examples](./examples/) directory.

## Workflows

For advanced workflow patterns and examples, please refer to the [Workflows README](examples/workflows/README.md) for more details.

## Contributing

See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more details. Thank you!
