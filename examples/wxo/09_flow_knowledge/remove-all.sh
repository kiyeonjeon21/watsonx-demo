#!/usr/bin/env bash
set -x

# Remove email agents, flow tool, and python tool from watsonx Orchestrate
echo "Removing email agents and tools..."

# Remove agents first
orchestrate agents remove -n email_agent -k native
orchestrate agents remove -n ibm_email_agent -k native

# Remove flow tool
orchestrate tools remove -n ibm_knowledge_to_emails

# Remove python tool
orchestrate tools remove -n send_emails

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: email_agent"
echo "- Agent: ibm_email_agent"
echo "- Flow Tool: ibm_knowledge_to_emails"
echo "- Python Tool: send_emails"
echo ""
echo "Note: IBM knowledge base and ibm_agent from 02_agent_knowledge were imported separately."
echo "Run './remove-all.sh' in 02_agent_knowledge to remove them if needed."

