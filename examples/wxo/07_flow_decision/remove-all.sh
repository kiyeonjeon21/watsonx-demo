#!/usr/bin/env bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Remove agent first
orchestrate agents remove -n insurance_assessment_agent -k native

# Remove tools
orchestrate tools remove -n get_insurance_rate

echo "Flow decision tools and agent have been removed successfully!"
