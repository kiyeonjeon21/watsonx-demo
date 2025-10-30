#!/usr/bin/env bash
set -x

# Remove document extractor agent and flow tool from watsonx Orchestrate
echo "Removing document extractor agent and tools..."

# Remove agent first
orchestrate agents remove -n document_extractor_agent -k native

# Remove flow tool
orchestrate tools remove -n custom_flow_docext_example

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: document_extractor_agent"
echo "- Flow Tool: custom_flow_docext_example"

