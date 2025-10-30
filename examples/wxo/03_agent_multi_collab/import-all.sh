#!/usr/bin/env bash
set -e
set -x

# orchestrate env activate local
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 에러 처리 함수 추가
handle_error() {
    echo "Error: $1"
    exit 1
}

# WXO 도구 임포트
for wxo_tool in get_wxo_info.py get_wxo_features.py get_wxo_pricing.py get_wxo_resources.py get_wxo_integration.py; do
    orchestrate tools import -k python -f ${SCRIPT_DIR}/tools/wxo/${wxo_tool} -r ${SCRIPT_DIR}/tools/requirements.txt || handle_error "Failed to import WXO tool: ${wxo_tool}"
done

# Watson Assistant 도구 임포트
for wx_assistant_tool in get_wx_assistant_info.py get_wx_assistant_features.py get_wx_assistant_pricing.py get_wx_assistant_resources.py; do
    orchestrate tools import -k python -f ${SCRIPT_DIR}/tools/wx-assistant/${wx_assistant_tool} -r ${SCRIPT_DIR}/tools/requirements.txt || handle_error "Failed to import Assistant tool: ${wx_assistant_tool}"
done

# Cognos Analytics 도구 임포트
for cognos_tool in get_cognos_info.py get_cognos_features.py get_cognos_pricing.py get_cognos_resources.py; do
    orchestrate tools import -k python -f ${SCRIPT_DIR}/tools/cognos-analytics/${cognos_tool} -r ${SCRIPT_DIR}/tools/requirements.txt || handle_error "Failed to import Cognos tool: ${cognos_tool}"
done

# 에이전트 임포트 (의존성 순서에 따라)
for agent in wxo-agent.yaml cognos-analytics-agent.yaml wx-assistant-agent.yaml ibm-product-specialist.yaml; do
    orchestrate agents import -f ${SCRIPT_DIR}/agents/${agent} || handle_error "Failed to import agent: ${agent}"
done

echo "Successfully imported all tools and agents"