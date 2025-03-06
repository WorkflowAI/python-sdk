"""
This example demonstrates how to use templated instructions that adapt based on input variables.
The template variables are automatically populated from the input model's fields.

The templating uses Jinja2 syntax (server-side rendering):
- {{ variable }} for variable substitution
- {% if condition %} ... {% endif %} for conditionals
- {% for item in items %} ... {% endfor %} for loops
- {{ loop.index }} for loop indices

For full Jinja2 template syntax documentation, see:
https://jinja.palletsprojects.com/en/stable/

It showcases:
1. Simple variable substitution
2. Conditional logic
3. Loops
4. Nested conditionals
"""

import asyncio

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class CodeReviewInput(BaseModel):
    """Input model for the code review agent."""

    language: str = Field(
        description="The programming language of the code to review",
        examples=["python", "javascript", "typescript"],
    )
    code: str = Field(
        description="The code to review",
    )
    style_guide: str = Field(
        description="The style guide to follow",
        examples=["PEP 8", "Google Style", "Airbnb"],
    )
    is_production: bool = Field(
        description="Whether this is a production code review",
        default=False,
    )
    required_checks: list[str] = Field(
        description="List of specific checks to perform",
        default=["code style", "performance", "maintainability"],
    )
    security_level: str = Field(
        description="Required security level",
        default="standard",
        examples=["standard", "high"],
    )


class CodeReviewOutput(BaseModel):
    """Output model containing the code review results."""

    overall_assessment: str = Field(
        description="Overall assessment of the code quality",
    )
    style_violations: list[str] = Field(
        description="List of style guide violations",
    )
    security_issues: list[str] = Field(
        description="List of security concerns",
        default_factory=list,
    )
    suggested_improvements: list[str] = Field(
        description="List of suggested improvements",
    )


@workflowai.agent(
    id="templated-code-reviewer",
    model=Model.CLAUDE_3_5_SONNET_LATEST,
)
async def review_code(review_input: CodeReviewInput) -> CodeReviewOutput:
    """
    Review code based on specified parameters and guidelines.

    You are a code reviewer for {{ language }} code.
    Please review the code according to the {{ style_guide }} style guide.

    {% if is_production %}
    This is a PRODUCTION code review - please be extra thorough and strict about best practices.
    {% else %}
    This is a development code review - focus on maintainability and clarity.
    {% endif %}

    Required checks to perform:
    {% for check in required_checks %}{{ loop.index }}. {{ check }}
    {% endfor %}

    {% if security_level == "high" %}
    Additional security requirements:
    - Must follow secure coding practices
    - Check for potential security vulnerabilities
    - Ensure all inputs are properly sanitized
    {% endif %}

    Guidelines:
    1. Check for adherence to {{ style_guide }} conventions
    2. Look for potential bugs and performance issues
    3. Suggest improvements while keeping the {{ language }} best practices in mind

    {% if language == "python" %}
    Python-specific checks:
    - Type hints usage
    - PEP 8 compliance
    - Docstring format
    {% elif language == "javascript" or language == "typescript" %}
    JavaScript/TypeScript-specific checks:
    - ESLint rules compliance
    - Modern ES6+ features usage
    - Browser compatibility
    {% endif %}

    Please analyze the following code and provide:
    1. An overall assessment
    2. Style guide violations
    3. Security issues (if any)
    4. Suggested improvements

    The code is:
    {{ code }}
    """
    ...


async def main():
    # Example 1: Python code review for development
    print("\nExample 1: Python Development Code Review")
    print("-" * 50)
    python_code = """
def calculate_sum(numbers):
    result = 0
    for n in numbers:
        result = result + n
    return result
    """

    run = await review_code.run(
        CodeReviewInput(
            language="python",
            code=python_code,
            style_guide="PEP 8",
            required_checks=["type hints", "docstring", "performance"],
        ),
    )
    print(run)

    # Example 2: TypeScript production code with high security
    print("\nExample 2: TypeScript Production Code Review (High Security)")
    print("-" * 50)
    typescript_code = """
function processUserData(data: any) {
    const userId = data.id;
    const query = `SELECT * FROM users WHERE id = ${userId}`;
    return executeQuery(query);
}
    """

    run = await review_code(
        CodeReviewInput(
            language="typescript",
            code=typescript_code,
            style_guide="Airbnb",
            is_production=True,
            security_level="high",
            required_checks=["security", "type safety", "SQL injection"],
        ),
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
