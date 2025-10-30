#!/usr/bin/env bash
set -x

# Remove ticket processing agent and tools from watsonx Orchestrate
echo "Removing ticket processing agent and tools..."

# Remove agent first
orchestrate agents remove -n ticket_processing_agent -k native

# Remove flow tool
orchestrate tools remove -n extract_support_info

# Remove python tool
orchestrate tools remove -n email_helpdesk

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: ticket_processing_agent"
echo "- Flow Tool: extract_support_info"
echo "- Python Tool: email_helpdesk"

