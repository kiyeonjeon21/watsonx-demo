#!/usr/bin/env python3
"""
Simple MCP Server for watsonx Orchestrate using FastMCP
Provides basic mathematical operations as tools
"""

from mcp.server.fastmcp import FastMCP

# Create FastMCP instance
mcp = FastMCP("Math")

@mcp.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers"""
    result = a + b
    return f"The sum of {a} and {b} is {result}"

@mcp.tool()
async def subtract(a: float, b: float) -> str:
    """Subtract two numbers"""
    result = a - b
    return f"The difference of {a} and {b} is {result}"

@mcp.tool()
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers"""
    result = a * b
    return f"The product of {a} and {b} is {result}"

@mcp.tool()
async def divide(a: float, b: float) -> str:
    """Divide two numbers"""
    if b == 0:
        return "Error: Cannot divide by zero"
    result = a / b
    return f"The quotient of {a} divided by {b} is {result}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
