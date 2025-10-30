#!/usr/bin/env bash
set -x

# Remove text extraction agent and flow tool from watsonx Orchestrate
echo "Removing text extraction agent and tools..."

# Remove agent first
orchestrate agents remove -n text_extraction_agent -k native

# Remove flow tool
orchestrate tools remove -n text_extraction_flow_example

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: text_extraction_agent"
echo "- Flow Tool: text_extraction_flow_example"

