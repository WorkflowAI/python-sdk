import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field  # pyright: ignore [reportUnknownVariableType]

import workflowai
from workflowai import WorkflowAIError
from workflowai.core.domain.model import Model
from workflowai.fields import PDF


class PDFQuestionInput(BaseModel):
    pdf: PDF = Field(description="The PDF document to analyze")
    question: str = Field(description="The question to answer about the PDF content")


class PDFAnswerOutput(BaseModel):
    answer: str = Field(description="The answer to the question based on the PDF content")
    quotes: list[str] = Field(description="Relevant quotes from the PDF that support the answer")


@workflowai.agent(id="pdf-answer", model=Model.CLAUDE_3_5_SONNET_LATEST)
async def answer_pdf_question(_: PDFQuestionInput) -> PDFAnswerOutput:
    """
    Analyze the provided PDF document and answer the given question.
    Provide a clear and concise answer based on the content found in the PDF.

    Focus on:
    - Accurate information extraction from the PDF
    - Direct and relevant answers to the question
    - Context-aware responses that consider the full document
    - Citing specific sections or pages when relevant

    If the question cannot be answered based on the PDF content,
    provide a clear explanation of why the information is not available.
    """
    ...


async def run_pdf_answer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "assets", "sec-form-4.pdf")

    # With a properly async function you should use an async open
    # see https://github.com/Tinche/aiofiles for example
    with open(pdf_path, "rb") as pdf_file:  # noqa: ASYNC230
        import base64

        content = base64.b64encode(pdf_file.read()).decode("utf-8")

    pdf = PDF(content_type="application/pdf", data=content)
    # Could also pass the content via url
    # pdf = PDF(url="https://example.com/sample.pdf")
    question = "How many stocks were sold? What is the total amount in USD?"

    try:
        run = await answer_pdf_question.run(
            PDFQuestionInput(pdf=pdf, question=question),
            use_cache="auto",
        )
    except WorkflowAIError as e:
        print(f"Failed to run task. Code: {e.error.code}. Message: {e.error.message}")
        return

    print(run)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(override=True)
    asyncio.run(run_pdf_answer())
