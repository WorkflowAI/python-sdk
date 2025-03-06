"""
This example demonstrates how to use streaming with WorkflowAI agents. It shows how to:
1. Stream outputs as they are generated
2. Get real-time updates during processing
3. Access run metadata like cost and duration
"""

import asyncio

from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]

import workflowai
from workflowai import Model


class TranslationInput(BaseModel):
    """Input for text translation."""

    text: str = Field(description="The French text to translate.")


class TranslationOutput(BaseModel):
    """Output containing the translated text."""

    translation: str = Field(
        default="",
        description="The text translated into English.",
    )


@workflowai.agent(id="french-translator", model=Model.CLAUDE_3_5_SONNET_LATEST)
async def translate_to_english(_: TranslationInput) -> TranslationOutput:
    """
    Translate French text into natural, fluent English.

    Guidelines:
    - Maintain the original tone and style
    - Ensure accurate translation of idioms and expressions
    - Preserve formatting and paragraph structure
    - Focus on clarity and readability in English
    """
    ...


async def main():
    # Example French text
    french_text = """
    Cher journal,

    Aujourd'hui, j'ai fait une magnifique randonnée dans les Alpes.
    Le temps était parfait, avec un ciel bleu éclatant et une légère brise.
    Les montagnes étaient encore couvertes de neige, créant un paysage à couper le souffle.
    Les sommets majestueux se dressaient devant moi comme des géants silencieux.

    En chemin, j'ai rencontré des randonneurs sympathiques qui m'ont partagé leur pique-nique.
    Nous avons échangé des histoires et des conseils sur les meilleurs sentiers de la région.

    Cette expérience restera gravée dans ma mémoire. La nature a vraiment le pouvoir de nous
    ressourcer et de nous faire oublier le stress du quotidien.

    À bientôt,
    Pierre
    """

    print("Starting translation...\n")

    # Keep track of the last chunk to get cost info
    last_chunk = None
    chunk_num = 1

    # Stream the translation with run information
    # We set use_cache='never' to prevent caching the response
    # This ensures we can see the streaming effect in the example
    # Otherwise, subsequent runs would return the cached result instantly,
    # making it hard to observe the incremental streaming behavior
    async for chunk in translate_to_english.stream(TranslationInput(text=french_text), use_cache="never"):
        print(f"--- Translation Progress (Chunk {chunk_num}) ---")
        print(chunk.output.translation)
        print("-" * 50)
        chunk_num += 1
        last_chunk = chunk

    if last_chunk:
        # Cost and duration are only available on the last streaming chunk
        # since they represent the final totals for the complete run.
        # We store the last chunk in last_chunk to access these values
        # after streaming completes.
        print(f"\nCost: ${last_chunk.cost_usd}")
        print(f"Latency: {last_chunk.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
