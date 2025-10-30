#!/usr/bin/env bash
set -x

# Remove healthcare agent and OpenAPI toolkit
echo "Removing healthcare agent and tools..."

# Remove agent first
orchestrate agents remove -n healthcare_agent -k native

# Remove tools
orchestrate tools remove -n getHealthCareProviders

echo ""
echo "Removal completed successfully!"
echo "The following have been removed:"
echo "- Agent: healthcare_agent"
echo "- Tool: getHealthCareProviders"

