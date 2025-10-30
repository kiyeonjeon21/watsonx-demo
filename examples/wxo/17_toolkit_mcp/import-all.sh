#!/bin/bash

# Import Basic MCP Server toolkit into watsonx Orchestrate
echo "Importing Basic MCP Server toolkit..."

orchestrate toolkits import \
    --kind mcp \
    --name basic-mcp-toolkit \
    --description "Basic mathematical operations toolkit using MCP" \
    --package-root "$(pwd)" \
    --command "python3 mcp_server.py" \
    --tools "add,subtract,multiply,divide"

echo "Import completed successfully!"
echo ""
echo "Available tools:"
echo "- add: Add two numbers"
echo "- subtract: Subtract two numbers"
echo "- multiply: Multiply two numbers"
echo "- divide: Divide two numbers"
echo ""
echo "You can now use these tools in your watsonx Orchestrate agents."
