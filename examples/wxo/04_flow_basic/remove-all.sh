#!/usr/bin/env bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Remove agent first
orchestrate agents remove -n hello_message_agent -k native

# Remove tools
orchestrate tools remove -n get_hello_message
orchestrate tools remove -n combine_names
orchestrate tools remove -n hello_message_flow

echo "Simple flow tools and agent have been removed successfully!"
