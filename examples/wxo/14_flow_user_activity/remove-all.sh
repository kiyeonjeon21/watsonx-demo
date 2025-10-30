#!/usr/bin/env bash
set -x

# Remove user flow agent and tool from watsonx Orchestrate
echo "Removing user flow agent and tools..."

# Remove agent first
orchestrate agents remove -n user_flow_agent -k native

# Remove flow tool
orchestrate tools remove -n user_flow_example

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: user_flow_agent"
echo "- Flow Tool: user_flow_example"

