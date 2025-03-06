"""
This example demonstrates the different caching options in WorkflowAI:
1. 'auto' - Cache only when temperature is 0 (default)
2. 'always' - Always use cache if available
3. 'never' - Never use cache, always execute new runs

The example uses a medical SOAP notes extractor to show how caching affects:
- Response consistency (important for medical documentation)
- Cost efficiency
- Execution time
"""

import asyncio
import time
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model

# Import CacheUsage type
CacheUsage = Literal["auto", "always", "never"]


class SOAPInput(BaseModel):
    """Input containing a medical consultation transcript."""
    transcript: str = Field(
        description="The medical consultation transcript to analyze",
    )


class SOAPNote(BaseModel):
    """Structured SOAP note components."""
    subjective: list[str] = Field(
        description="Patient's symptoms, complaints, and history as reported",
        examples=["Patient reports severe headache for 3 days", "Denies fever or nausea"],
    )
    objective: list[str] = Field(
        description="Observable, measurable findings from examination",
        examples=["BP 120/80", "Temperature 98.6°F", "No visible inflammation"],
    )
    assessment: list[str] = Field(
        description="Diagnosis or clinical impressions",
        examples=["Tension headache", "Rule out migraine"],
    )
    plan: list[str] = Field(
        description="Treatment plan and next steps",
        examples=["Prescribed ibuprofen 400mg", "Follow up in 2 weeks"],
    )


@workflowai.agent(
    id="soap-extractor",
    model=Model.LLAMA_3_3_70B,
)
async def extract_soap_notes(soap_input: SOAPInput) -> SOAPNote:
    """
    Extract SOAP notes from a medical consultation transcript.

    Guidelines:
    1. Analyze the transcript to identify key medical information
    2. Organize information into SOAP format:
       - Subjective: Patient's symptoms, complaints, and history
       - Objective: Physical examination findings and test results
       - Assessment: Diagnosis or clinical impression
       - Plan: Treatment plan and next steps

    3. Be thorough but concise
    4. Use medical terminology appropriately
    5. Include all relevant information from the transcript
    """
    ...


class ResultMetrics(TypedDict):
    option: str
    duration: float
    cost: float


async def demonstrate_caching(transcript: str):
    """Run the same transcript with different caching options and compare results."""

    print("\nComparing caching options")
    print("-" * 50)

    cache_options: list[CacheUsage] = ["auto", "always", "never"]
    results: list[ResultMetrics] = []

    for cache_option in cache_options:
        start_time = time.time()

        run = await extract_soap_notes.run(
            SOAPInput(transcript=transcript),
            use_cache=cache_option,
        )

        duration = time.time() - start_time

        # Store metrics for comparison
        results.append({
            "option": cache_option,
            "duration": duration,
            "cost": float(run.cost_usd or 0.0),  # Convert None to 0.0
        })

    # Print comparison table
    print("\nResults Comparison:")
    print("-" * 50)
    print(f"{'Cache Option':<12} {'Duration':<10} {'Cost':<8}")
    print("-" * 50)

    for r in results:
        print(
            f"{r['option']:<12} "
            f"{r['duration']:.2f}s{'*' if r['duration'] < 0.1 else '':<8} "
            f"${r['cost']:<7}",
        )

    print("-" * 50)
    print("* Very fast response indicates cached result")


async def main():
    # Example medical consultation transcript
    transcript = """
    Patient is a 45-year-old female presenting with severe headache for the past 3 days.
    She describes the pain as throbbing, primarily on the right side of her head.
    Pain level reported as 7/10. Denies fever, nausea, or visual disturbances.
    Previous history of migraines, but states this feels different.

    Vital signs stable: BP 120/80, HR 72, Temp 98.6°F.
    Physical exam shows mild tenderness in right temporal area.
    No neurological deficits noted.
    Eye examination normal, no papilledema.

    Assessment suggests tension headache, but need to rule out migraine.
    No red flags for secondary causes identified.

    Plan: Prescribed ibuprofen 400mg q6h for pain.
    Recommended stress reduction techniques.
    Patient education provided regarding headache triggers.
    Follow up in 2 weeks, sooner if symptoms worsen.
    Return precautions discussed.
    """

    print("\nDemonstrating different caching options")
    print("=" * 50)
    print("This example shows how caching affects the agent's behavior:")
    print("- 'auto': Caches only when temperature is 0 (default)")
    print("- 'always': Reuses cached results when available")
    print("- 'never': Generates new results every time")

    await demonstrate_caching(transcript)


if __name__ == "__main__":
    asyncio.run(main())
