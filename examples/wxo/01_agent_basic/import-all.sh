#!/usr/bin/env bash
set -x

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Import tools first
for python_tool in greetings.py; do
  orchestrate tools import -k python -f ${SCRIPT_DIR}/tools/${python_tool} -r ${SCRIPT_DIR}/tools/requirements.txt
done

# Import agent
for agent in greeter.yaml; do
  orchestrate agents import -f ${SCRIPT_DIR}/agents/${agent}
done

echo "Hello World Agent and tools have been imported successfully!"