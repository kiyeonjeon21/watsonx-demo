#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 도구 제거
tools=(
    # WXO 도구
    "get_wxo_info"
    "get_wxo_features"
    "get_wxo_pricing"
    "get_wxo_resources"
    "get_wxo_integration"
    # Watson Assistant 도구
    "get_wx_assistant_info"
    "get_wx_assistant_features"
    "get_wx_assistant_pricing"
    "get_wx_assistant_resources"
    # Cognos Analytics 도구
    "get_cognos_info"
    "get_cognos_features"
    "get_cognos_pricing"
    "get_cognos_resources"
)

for tool in "${tools[@]}"; do
    echo "Removing tool: $tool"
    orchestrate tools remove -n "$tool"
done

# 에이전트 제거
agents=(
    "IBM_Product_Specialist"
    "Wx_Orchestrate"
    "Wx_Assistant"
    "Cognos_Analytics"
)

for agent in "${agents[@]}"; do
    echo "Removing agent: $agent"
    orchestrate agents remove -n "$agent" -k native
done

echo "All tools and agents have been removed"