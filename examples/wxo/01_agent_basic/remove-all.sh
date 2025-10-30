#!/usr/bin/env bash
set -x

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Remove tools
orchestrate tools remove -n greeting

# Remove agent first
orchestrate agents remove -n greeter -k native

echo "Hello World Agent and tools have been removed successfully!"