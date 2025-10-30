#!/usr/bin/env bash
set -x

# Remove Basic MCP Server toolkit from watsonx Orchestrate
echo "Removing Basic MCP Server toolkit..."

orchestrate toolkits remove -n basic-mcp-toolkit

echo "Removal completed successfully!"
echo ""
echo "The following toolkit and its tools have been removed:"
echo "- basic-mcp-toolkit"
echo "  - add: Add two numbers"
echo "  - subtract: Subtract two numbers"
echo "  - multiply: Multiply two numbers"
echo "  - divide: Divide two numbers"

