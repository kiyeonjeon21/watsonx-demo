#!/usr/bin/env bash
set -x

# Remove user flow agent (no files) and tool from watsonx Orchestrate
echo "Removing user flow agent (no files) and tools..."

# Remove agent first
orchestrate agents remove -n user_flow_agent_no_files -k native

# Remove flow tool
orchestrate tools remove -n user_flow_example_no_files

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: user_flow_agent_no_files"
echo "- Flow Tool: user_flow_example_no_files"

