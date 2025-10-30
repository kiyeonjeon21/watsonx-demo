#!/usr/bin/env bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Remove agent first
orchestrate agents remove -n pet_agent -k native

# Remove tools
orchestrate tools remove -n getCatFact
orchestrate tools remove -n getDogFact
orchestrate tools remove -n get_pet_facts

echo "Flow branching tools and agent have been removed successfully!"
