"""
This example demonstrates how to convert natural language questions to SQL queries.
It uses a sample e-commerce database schema and shows how to generate safe and efficient SQL queries.

Like example 14 (templated instructions), this example shows how to use variables in the agent's
instructions. The template variables ({{ db_schema }} and {{ question }}) are automatically populated
from the input model's fields, allowing the instructions to adapt based on the input.

The example includes:
1. Simple SELECT query with conditions
2. JOIN query with aggregation
3. Complex query with multiple JOINs, grouping, and ordering
"""

import asyncio

from pydantic import BaseModel, Field

import workflowai
from workflowai import Model


class SQLGenerationInput(BaseModel):
    """Input model for the SQL generation agent."""

    db_schema: str = Field(
        description="The complete SQL schema with CREATE TABLE statements",
    )
    question: str = Field(
        description="The natural language question to convert to SQL",
    )


class SQLGenerationOutput(BaseModel):
    """Output model containing the generated SQL query and explanation."""

    sql_query: str = Field(
        description="The generated SQL query",
    )
    explanation: str = Field(
        description="Explanation of what the query does and why certain choices were made",
    )
    tables_used: list[str] = Field(
        description="List of tables referenced in the query",
    )


@workflowai.agent(
    id="text-to-sql",
    model=Model.CLAUDE_3_5_SONNET_LATEST,
)
async def generate_sql(review_input: SQLGenerationInput) -> SQLGenerationOutput:
    """
    Convert natural language questions to SQL queries based on the provided schema.

    You are a SQL expert that converts natural language questions into safe and efficient SQL queries.
    The queries should be compatible with standard SQL databases.

    Important guidelines:
    1. NEVER trust user input directly in queries to prevent SQL injection
    2. Use proper quoting and escaping for string values
    3. Use meaningful table aliases for better readability
    4. Format queries with proper indentation and line breaks
    5. Use explicit JOIN conditions (no implicit joins)
    6. Include column names in GROUP BY rather than positions

    Schema:
    {{ db_schema }}

    Question to convert to SQL:
    {{ question }}

    Please provide:
    1. A safe and efficient SQL query
    2. An explanation of the query and any important considerations
    3. List of tables used in the query
    """
    ...


async def main():
    # Example schema for an e-commerce database
    schema = """
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        price DECIMAL(10,2) NOT NULL,
        category TEXT NOT NULL,
        stock_quantity INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL DEFAULT 'pending',
        total_amount DECIMAL(10,2) NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );

    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10,2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """

    # Example 1: Simple SELECT with conditions
    print("\nExample 1: Find expensive products")
    print("-" * 50)
    run = await generate_sql.run(
        SQLGenerationInput(
            db_schema=schema,
            question="Show me all products that cost more than $100, ordered by price descending",
        ),
    )
    print(run)

    # Example 2: JOIN with aggregation
    print("\nExample 2: Customer order summary")
    print("-" * 50)
    run = await generate_sql.run(
        SQLGenerationInput(
            db_schema=schema,
            question=(
                "List all customers with their total number of orders and total spend, "
                "only showing customers who have made at least 2 orders"
            ),
        ),
    )
    print(run)

    # Example 3: Complex query
    print("\nExample 3: Product category analysis")
    print("-" * 50)
    run = await generate_sql.run(
        SQLGenerationInput(
            db_schema=schema,
            question=(
                "What are the top 3 product categories by revenue in the last 30 days, "
                "including the number of unique customers who bought from each category?"
            ),
        ),
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
