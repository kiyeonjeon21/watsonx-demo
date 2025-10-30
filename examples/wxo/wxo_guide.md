# Watsonx Orchestrate CLI Guide

## Environment Management

Manage orchestrate environments (list, add, activate).

### List environments
```bash
orchestrate env list
```

### Add new environment
```bash
orchestrate env add -n <env-name> -u <wxo-url>
```

### Activate environment
```bash
orchestrate env activate <env-name>
```

## Import Tools

### Import Python tool
```bash
orchestrate tools import -k python -f <python-file-tool-absolute-path> -r <requirements-text-absolute-path>
```

### Import flow
```bash
orchestrate tools import -k flow -f <path-to-python-file>
```

## Import Agents

### Import agent from YAML
```bash
orchestrate agents import -f <agent-yaml-absolute-path>
```

## Delete Resources

### Delete agents (batch)
```bash
for agent in [agent-name-list]; do
    echo "Removing: $agent"
    orchestrate agents remove -n "$agent" -k native
done
```

### Delete tools (batch)
```bash
for tool in [tool-name-list]; do
    echo "Removing tool: $tool"
    orchestrate tools remove -n "$tool"
done
```

### Delete MCP toolkit
```bash
orchestrate toolkits remove -n <my-toolkit-name>
```

### Delete external agent
```bash
orchestrate agents remove -n <external-agent> -k external
```

## Connection Management

### Add connection
```bash
orchestrate connections add --app-id <app-id-name>
```

### Configure connection
```bash
orchestrate connections configure --app-id <app-id-name> --env draft --kind api_key --type team
```

### Set connection credentials
```bash
orchestrate connections set-credentials --app-id <app-id-name> --env draft --api-key <api-key>
```