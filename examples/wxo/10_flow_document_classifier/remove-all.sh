#!/usr/bin/env bash
set -x

# Remove document classifier agent and flow tool from watsonx Orchestrate
echo "Removing document classifier agent and tools..."

# Remove agent first
orchestrate agents remove -n document_classifier_agent -k native

# Remove flow tool
orchestrate tools remove -n custom_flow_docclassifier_example

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: document_classifier_agent"
echo "- Flow Tool: custom_flow_docclassifier_example"

