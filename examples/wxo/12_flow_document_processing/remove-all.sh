#!/usr/bin/env bash
set -x

# Remove document processing agent and tools from watsonx Orchestrate
echo "Removing document processing agent and tools..."

# Remove agent first
orchestrate agents remove -n document_processing_agent -k native

# Remove flow tool
orchestrate tools remove -n document_processing_flow

# Remove python tools
orchestrate tools remove -n get_kvp_schemas_for_invoice
orchestrate tools remove -n get_kvp_schemas_for_utility_bill

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: document_processing_agent"
echo "- Flow Tool: document_processing_flow"
echo "- Python Tool: get_kvp_schemas_for_invoice"
echo "- Python Tool: get_kvp_schemas_for_utility_bill"

