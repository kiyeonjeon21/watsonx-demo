"""
Example 13: AI Service Deployment
Demonstrates deploying an agent as an AI service using IBM watsonx SDK.
Based on: samples/ai service deploy sample.ipynb

Self-Contained Design:
- AI service function definition
- Local testing
- Deployment to watsonx
- Testing deployed service
- Use # %% cell markers for easy conversion to notebook
"""

import os
import tempfile
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.deployments import RuntimeContext
from dotenv import load_dotenv

# %% [markdown]
# # Example 13: AI Service Deployment
# This demonstrates how to deploy an agent as an AI service.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
WATSONX_SPACE_ID = os.getenv('WATSONX_SPACE_ID_DEV')

print("="*60)
print("EXAMPLE 13: AI Service Deployment")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")

credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

client = APIClient(credentials)
print(f"✓ Credentials initialized")
print(f"  Project ID: {WATSONX_PROJECT_ID}")

# %% Cell 3: Connect to Space
print("\n2. Connecting to space...")

if WATSONX_SPACE_ID:
    client.set.default_space(WATSONX_SPACE_ID)
    print(f"✓ Connected to space: {WATSONX_SPACE_ID}")
else:
    print("❌ ERROR: WATSONX_SPACE_ID not set")
    raise ValueError("WATSONX_SPACE_ID is required to deploy the AI service. Please set it in your .env file.")

# %% Cell 4: Define AI Service Function
print("\n3. Defining AI service function...")

# Define service parameters
params = {
    "space_id": WATSONX_SPACE_ID
}

def gen_ai_service(context, params=params, **custom):
    """AI Service function with agent and tool integration."""

    # Merge custom kwargs into params
    if custom:
        params = {**params, **custom}
    
    # Import dependencies
    from langchain_ibm import ChatWatsonx
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.tools import StructuredTool
    import json
    import requests
    
    service_url = "https://us-south.ml.cloud.ibm.com" # need hardcoding because of deployed service
    model_id = "meta-llama/llama-3-2-90b-vision-instruct"
    # Get credentials token
    inner_credentials = {
        "url": service_url,
        "token": context.generate_token()
    }
    
    # Setup client
    inner_client = APIClient(inner_credentials)
    space_id = params.get("space_id")
    inner_client.set.default_space(space_id)
    
    def create_chat_model(watsonx_client):
        """Create ChatWatsonx model instance."""
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }
        
        chat_model = ChatWatsonx(
            model_id=model_id,
            url=service_url,
            space_id=space_id,
            params=parameters,
            watsonx_client=watsonx_client,
        )
        return chat_model
    
    def create_utility_agent_tool(tool_name, params, api_client, **kwargs):
        """Create a utility agent tool."""
        utility_agent_tool = Toolkit(api_client=api_client).get_tool(tool_name)
        
        tool_description = utility_agent_tool.get("description")
        
        if kwargs.get("tool_description"):
            tool_description = kwargs.get("tool_description")
        elif utility_agent_tool.get("agent_description"):
            tool_description = utility_agent_tool.get("agent_description")
        
        tool_schema = utility_agent_tool.get("input_schema")
        if tool_schema is None:
            tool_schema = {
                "type": "object",
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {
                    "input": {
                        "description": "input for the tool",
                        "type": "string"
                    }
                }
            }
        
        def run_tool(**tool_input):
            # tool_input is already unpacked kwargs (e.g., query='IBM')
            # Convert to dict if needed
            tool_input_dict = dict(tool_input) if tool_input else {}
            
            # Check if we have input_schema to determine the expected format
            input_schema = utility_agent_tool.get("input_schema")
            
            # According to documentation:
            # - If input_schema is None: use string input directly  
            # - If input_schema exists: use dict matching the schema (keep keys as-is)
            if input_schema is None:
                # No schema - use string input directly
                if len(tool_input_dict) == 1:
                    query = list(tool_input_dict.values())[0]
                elif "input" in tool_input_dict:
                    query = tool_input_dict["input"]
                else:
                    query = str(tool_input_dict) if tool_input_dict else ""
            else:
                # Schema exists - use dict format matching the schema
                # IMPORTANT: Keep the original keys (don't convert "query" to "input")
                # tool_input_dict already has the correct keys from the tool call
                query = tool_input_dict
            
            try:
                results = utility_agent_tool.run(
                    input=query,
                    config=params
                )
                output = results.get("output") if results else str(results)
                if not output:
                    output = str(results)
                return output
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                print(f"  Tool error: {error_msg}", flush=True)
                print(f"  Input schema: {input_schema}", flush=True)
                print(f"  Tool input dict: {tool_input_dict}", flush=True)
                print(f"  Attempted query: {query}", flush=True)
                return error_msg
        
        return StructuredTool(
            name=tool_name,
            description=tool_description,
            func=run_tool,
            args_schema=tool_schema
        )
    
    def create_tools(inner_client, context):
        """Create tools for the agent."""
        tools = []
        
        # Add Wikipedia tool
        try:
            config = {"maxResults": 5}
            wikipedia_tool = create_utility_agent_tool("Wikipedia", config, inner_client)
            tools.append(wikipedia_tool)
        except Exception as e:
            print(f"  ⚠ Could not add Wikipedia tool: {e}", flush=True)
        
        return tools
    
    def create_react_agent_graph(model, tools, system_prompt=None):
        """Create a ReAct agent using LangGraph StateGraph."""
        
        def should_continue(state):
            """Determine if agent should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END
        
        # Bind tools only if tools are provided
        if tools:
            # Debug logging (may be verbose in deployed service)
            # print(f"  Binding {len(tools)} tools to model", flush=True)
            # for tool in tools:
            #     print(f"    - {tool.name}: {tool.description[:100] if tool.description else 'No description'}...", flush=True)
            model_with_tools = model.bind_tools(tools)
        else:
            model_with_tools = model
        
        def call_model(state):
            """Call the model with tools."""
            messages = state["messages"]
            if system_prompt:
                # Add system prompt as first message if not already present
                if not any(isinstance(msg, HumanMessage) and "system" in str(msg.content).lower()[:20] for msg in messages):
                    system_msg = HumanMessage(content=system_prompt)
                    messages = [system_msg] + messages
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def call_tools(state):
            """Execute tool calls."""
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_messages = []
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Debug logging (may be verbose in deployed service)
                # print(f"  Tool calls detected: {len(last_message.tool_calls)}", flush=True)
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                    tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                    
                    # print(f"  Executing tool: {tool_name} with args: {tool_args}", flush=True)
                    
                    tool_func = None
                    for t in tools:
                        if t.name == tool_name:
                            tool_func = t
                            break
                    
                    if tool_func:
                        try:
                            result = tool_func.invoke(tool_args)
                            # Limit tool response length
                            result_str = str(result)
                            if len(result_str) > 2000:
                                result_str = result_str[:2000] + "... [truncated]"
                            
                            tool_call_id = tool_call.get("id", "") if isinstance(tool_call, dict) else getattr(tool_call, "id", "")
                            # Debug logging (may be verbose in deployed service)
                            # print(f"  Tool result length: {len(result_str)}", flush=True)
                            
                            tool_messages.append(
                                ToolMessage(
                                    content=result_str,
                                    tool_call_id=tool_call_id
                                )
                            )
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            # Error logging is important even in deployed service
                            print(f"  {error_msg}", flush=True)
                            tool_call_id = tool_call.get("id", "") if isinstance(tool_call, dict) else getattr(tool_call, "id", "")
                            tool_messages.append(
                                ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call_id
                                )
                            )
                    else:
                        error_msg = f"Tool {tool_name} not found"
                        # Error logging is important even in deployed service
                        print(f"  {error_msg}", flush=True)
                        tool_call_id = tool_call.get("id", "") if isinstance(tool_call, dict) else getattr(tool_call, "id", "")
                        tool_messages.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id
                            )
                        )
            
            return {"messages": tool_messages}
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def create_agent(model, tools, messages):
        """Create the agent with instructions using LangGraph."""
        instructions = """# Notes
- Use markdown syntax for formatting code snippets, links, JSON, tables, images, files.
- Any HTML tags must be wrapped in block quotes, for example ```<html>```.
- When returning code blocks, specify language.
- Sometimes, things don't go as planned. Tools may not provide useful information on the first few tries. You should always try a few different approaches before declaring the problem unsolvable.
- When the tool doesn't give you what you were asking for, you must either use another tool or a different tool input.
- When using search engines, you try different formulations of the query, possibly even in a different language.
- You cannot do complex calculations, computations, or data manipulations without using tools.
- If you need to call a tool to compute something, always call it instead of saying you will call it.

If a tool returns an IMAGE in the result, you must include it in your answer as Markdown.

Example:

Tool result: IMAGE({commonApiUrl}/wx/v1-beta/utility_agent_tools/cache/images/plt-04e3c91ae04b47f8934a4e6b7d1fdc2c.png)
Markdown to return to user: ![Generated image]({commonApiUrl}/wx/v1-beta/utility_agent_tools/cache/images/plt-04e3c91ae04b47f8934a4e6b7d1fdc2c.png)

You are a helpful assistant that uses tools to answer questions in detail.
When greeted, say "Hi, I am watsonx.ai agent. How can I help you?"
"""
        
        # Add system message if present
        for message in messages:
            if message.get("role") == "system":
                instructions += "\n\n" + message.get("content", "")
        
        graph = create_react_agent_graph(model, tools, system_prompt=instructions)
        return graph
    
    def convert_messages(messages):
        """Convert messages to LangGraph format."""
        converted_messages = []
        for message in messages:
            if message["role"] == "user":
                converted_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                converted_messages.append(AIMessage(content=message["content"]))
        return converted_messages
    
    def generate(context):
        """Generate response using the agent."""
        try:
            payload = context.get_json()
            messages = payload.get("messages")
            
            if not messages:
                raise ValueError("Messages are required in payload")
            
            inner_credentials_local = {
                "url": service_url,
                "token": context.get_token()
            }
            
            inner_client_local = APIClient(inner_credentials_local)
            model = create_chat_model(inner_client_local)
            tools = create_tools(inner_client_local, context)
            agent = create_agent(model, tools, messages)
            
            # Use same pattern as 09_simple_agents.py and 10_complex_agents.py
            config = {"recursion_limit": 50}
            inputs = {"messages": convert_messages(messages)}
            
            generated_response = agent.invoke(inputs, config)
            
            # Get last message content
            if not generated_response.get("messages"):
                raise ValueError("No messages in agent response")
            
            last_message = generated_response["messages"][-1]
            
            # Extract content - handle both string and object content
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
            
            if not content:
                # If empty content, get previous messages for context
                for msg in reversed(generated_response["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content
                        break
            
            execute_response = {
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content if content else "No response generated",
                        }
                    }],
                    "model_id": model_id
                }
            }
            
            return execute_response
            
        except Exception as e:
            # Return error response instead of empty response
            error_msg = f"Error generating response: {str(e)}"
            print(f"Error in generate: {error_msg}", flush=True)
            return {
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": error_msg,
                        }
                    }],
                    "model_id": model_id,
                    "error": str(e)
                }
            }
    
    def generate_stream(context):
        """Generate streaming response using the agent."""
        try:
            print("Generate stream", flush=True)
            payload = context.get_json()
            headers = context.get_headers()
            is_assistant = headers.get("X-Ai-Interface") == "assistant"
            messages = payload.get("messages")
            
            if not messages:
                raise ValueError("Messages are required in payload")
            
            inner_credentials_stream = {
                "url": service_url,
                "token": context.get_token()
            }
            
            inner_client_stream = APIClient(inner_credentials_stream)
            model = create_chat_model(inner_client_stream)
            tools = create_tools(inner_client_stream, context)
            agent = create_agent(model, tools, messages)
            
            # Use same pattern as 09_simple_agents.py - config as dict
            config = {"recursion_limit": 50}
            inputs = {"messages": convert_messages(messages)}
            
            # Stream with explicit stream_mode for langgraph 1.0.1
            # stream_mode=["updates", "messages"] allows tracking both node updates and message chunks
            response_stream = agent.stream(
                inputs, 
                config,
                stream_mode=["updates", "messages"]
            )
            
            for chunk in response_stream:
                chunk_type = chunk[0]
                finish_reason = ""
                usage = None
                message = None
                
                if chunk_type == "messages":
                    # Handle message chunks (streaming tokens)
                    message_list = chunk[1]
                    if message_list and len(message_list) > 0:
                        message_object = message_list[0]
                        # Check if it's an AIMessageChunk with content
                        if hasattr(message_object, 'type') and message_object.type == "ai" and hasattr(message_object, 'content'):
                            content = message_object.content
                            if content and content != "":
                                message = {
                                    "role": "assistant",
                                    "content": content
                                }
                            else:
                                continue
                        # Also handle AIMessageChunk directly
                        elif hasattr(message_object, 'type') and "AIMessageChunk" in str(type(message_object)):
                            content = getattr(message_object, 'content', '')
                            if content and content != "":
                                message = {
                                    "role": "assistant",
                                    "content": str(content)
                                }
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                elif chunk_type == "updates":
                    update = chunk[1]
                    if "agent" in update:
                        agent_update = update["agent"]
                        agent_result = agent_update["messages"][0]
                        
                        if agent_result.additional_kwargs:
                            kwargs = agent_update["messages"][0].additional_kwargs
                            tool_call = kwargs.get("tool_calls", [None])[0]
                            if tool_call:
                                if is_assistant:
                                    message = {
                                        "role": "assistant",
                                        "step_details": {
                                            "type": "tool_calls",
                                            "tool_calls": [{
                                                "id": tool_call["id"],
                                                "name": tool_call["function"]["name"],
                                                "args": tool_call["function"]["arguments"]
                                            }]
                                        }
                                    }
                                else:
                                    message = {
                                        "role": "assistant",
                                        "tool_calls": [{
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": tool_call["function"]["arguments"]
                                            }
                                        }]
                                    }
                        elif hasattr(agent_result, 'content') or (hasattr(agent_result, 'response_metadata') and agent_result.response_metadata):
                            # Final message from agent
                            content = getattr(agent_result, 'content', '')
                            
                            # Check if this is a final response (has response_metadata)
                            response_metadata = getattr(agent_result, 'response_metadata', None)
                            usage_metadata = getattr(agent_result, 'usage_metadata', None)
                            
                            if response_metadata or usage_metadata:
                                # This is the final response
                                message = {
                                    "role": "assistant",
                                    "content": str(content) if content else ""
                                }
                                
                                if response_metadata:
                                    finish_reason = response_metadata.get("finish_reason")
                                    if finish_reason:
                                        message["finish_reason"] = finish_reason
                                
                                if usage_metadata:
                                    usage = {
                                        "completion_tokens": usage_metadata.get("output_tokens", 0),
                                        "prompt_tokens": usage_metadata.get("input_tokens", 0),
                                        "total_tokens": usage_metadata.get("total_tokens", 0)
                                    }
                            elif content:
                                # Intermediate content chunk
                                message = {
                                    "role": "assistant",
                                    "content": str(content)
                                }
                            else:
                                continue
                    elif "tools" in update:
                        tools_update = update["tools"]
                        tool_result = tools_update["messages"][0]
                        if is_assistant:
                            message = {
                                "role": "assistant",
                                "step_details": {
                                    "type": "tool_response",
                                    "id": tool_result.id,
                                    "tool_call_id": tool_result.tool_call_id,
                                    "name": tool_result.name,
                                    "content": tool_result.content
                                }
                            }
                        else:
                            message = {
                                "role": "tool",
                                "id": tool_result.id,
                                "tool_call_id": tool_result.tool_call_id,
                                "name": tool_result.name,
                                "content": tool_result.content
                            }
                    else:
                        continue
                
                # Make sure message is defined before creating chunk_response
                if not message:
                    continue
                
                chunk_response = {
                    "choices": [{
                        "index": 0,
                        "delta": message
                    }],
                    "model_id": model_id
                }
                
                # Add finish_reason if present (from message or explicit)
                if "finish_reason" in message:
                    chunk_response["choices"][0]["finish_reason"] = message["finish_reason"]
                elif finish_reason:
                    chunk_response["choices"][0]["finish_reason"] = finish_reason
                
                # Add usage information if available
                if usage:
                    chunk_response["usage"] = usage
                
                yield chunk_response
                
        except Exception as e:
            # Yield error response
            error_msg = f"Error in stream: {str(e)}"
            print(f"Error in generate_stream: {error_msg}", flush=True)
            yield {
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": error_msg
                    }
                }],
                "model_id": model_id,
                "error": str(e)
            }
    
    return generate, generate_stream

print("✓ AI service function defined")

# %% Cell 5: Test Locally (Optional)
print("\n4. Testing AI service locally (optional)...")

if WATSONX_SPACE_ID:
    try:
        context = RuntimeContext(api_client=client)
        
        # Get the non-streaming function (index 0)
        streaming = False
        findex = 1 if streaming else 0
        local_function = gen_ai_service(context, space_id=WATSONX_SPACE_ID)[findex]
        
        # Test with a query that should use Wikipedia tool
        messages = [{"role": "user", "content": "Search Wikipedia for information about IBM and explain what IBM is"}]
        
        test_context = RuntimeContext(
            api_client=client, 
            request_payload_json={"messages": messages}
        )
        
        response = local_function(test_context)
        
        if streaming:
            print("\n  Streaming response:")
            for chunk in response:
                print(chunk, end="\n\n", flush=True)
        else:
            print("\n  Response:")
            content = response.get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            if content:
                print(f"  {content[:500]}...")
                if len(content) > 500:
                    print(f"  (Full response length: {len(content)} characters)")
            else:
                print(f"  Empty response! Full response: {response}")
        
        print("✓ Local test completed")
        
    except Exception as e:
        print(f"⚠ Local test failed: {e}")
        print("  Continuing to deployment steps...")
else:
    print("  Skipping local test (no space ID)")

# %% Cell 6: Store AI Service
print("\n5. Using custom software specification with langgraph==1.0.1...")

try:
    software_spec_id = None
    custom_spec_name = "runtime-24.1-py3.11-with-langgraph-1.0.1"
    
    # Check if custom software spec already exists
    try:
        software_spec_id = client.software_specifications.get_id_by_name(custom_spec_name)
        print(f"  ✓ Found existing custom software spec: {custom_spec_name}")
        print(f"  Using software spec: {custom_spec_name} ({software_spec_id})")
    except:
        # Create new custom software spec
        print(f"  Creating new custom software spec: {custom_spec_name}...")
        
        # 1. Create conda YAML file for package extension
        yaml_content = """name: langgraph-1.0.1
dependencies:
    - pip:
        - langgraph==1.0.1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_file_path = f.name
            f.write(yaml_content)
        
        try:
            # 2. Check if package extension already exists
            pe_name = "langgraph-1.0.1"
            try:
                pe_asset_id = client.package_extensions.get_id_by_name(pe_name)
                print(f"  ✓ Reusing existing package extension: {pe_name}")
            except:
                # Create new package extension
                pe_metadata = {
                    client.package_extensions.ConfigurationMetaNames.NAME: pe_name,
                    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Package extension with langgraph==1.0.1",
                    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
                }
                
                pe_asset_details = client.package_extensions.store(
                    meta_props=pe_metadata,
                    file_path=yaml_file_path
                )
                pe_asset_id = client.package_extensions.get_id(pe_asset_details)
                print(f"  ✓ Created package extension: {pe_name} ({pe_asset_id})")
            
            # 3. Get base software specification ID
            base_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")
            print(f"  Using base software spec: runtime-24.1-py3.11 ({base_id})")
            
            # 4. Create custom software specification
            ss_metadata = {
                client.software_specifications.ConfigurationMetaNames.NAME: custom_spec_name,
                client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Python 3.11 runtime with langgraph==1.0.1",
                client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_id},
                client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS: [{"guid": pe_asset_id}]
            }
            
            ss_asset_details = client.software_specifications.store(meta_props=ss_metadata)
            software_spec_id = client.software_specifications.get_id(ss_asset_details)
            print(f"  ✓ Created custom software spec: {custom_spec_name} ({software_spec_id})")
            print(f"  Using software spec: {custom_spec_name}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(yaml_file_path):
                os.unlink(yaml_file_path)
    
    # Define request and response schemas
    request_schema = {
        "application/json": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "messages": {
                    "title": "The messages for this chat session.",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "title": "The role of the message author.",
                                "type": "string",
                                "enum": ["user", "assistant"]
                            },
                            "content": {
                                "title": "The contents of the message.",
                                "type": "string"
                            }
                        },
                        "required": ["role", "content"]
                    }
                }
            },
            "required": ["messages"]
        }
    }
    
    response_schema = {
        "application/json": {
            "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
        }
    }
    
    if software_spec_id:
        # Store the AI service
        ai_service_metadata = {
            client.repository.AIServiceMetaNames.NAME: "Watsonx AI Service Example",
            client.repository.AIServiceMetaNames.DESCRIPTION: "AI service with agent and tool integration",
            client.repository.AIServiceMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
            client.repository.AIServiceMetaNames.CUSTOM: {},
            client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema,
            client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema,
            client.repository.AIServiceMetaNames.TAGS: ["wx-agent"]
        }
        
        ai_service_details = client.repository.store_ai_service(
            meta_props=ai_service_metadata,
            ai_service=gen_ai_service
        )
        
        ai_service_id = client.repository.get_ai_service_id(ai_service_details)
        print(f"✓ AI service stored")
        print(f"  Service ID: {ai_service_id}")
        
    else:
        print("⚠ Skipping storage (no software spec)")
        ai_service_id = None
        
except Exception as e:
    print(f"⚠ Failed to store AI service: {e}")
    ai_service_id = None

# %% Cell 7: Deploy AI Service
print("\n6. Deploying AI service...")

if ai_service_id:
    try:
        deployment_custom = {
            "avatar_icon": "Bot",
            "avatar_color": "background",
            # "placeholder_image": "placeholder2.png"
        }
        
        deployment_metadata = {
            client.deployments.ConfigurationMetaNames.NAME: "Watsonx AI Service Example",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
            client.deployments.ConfigurationMetaNames.CUSTOM: deployment_custom,
            client.deployments.ConfigurationMetaNames.DESCRIPTION: "AI service with agent and tool integration",
            client.repository.AIServiceMetaNames.TAGS: ["wx-agent"]
        }
        
        function_deployment_details = client.deployments.create(
            ai_service_id,
            meta_props=deployment_metadata,
            space_id=WATSONX_SPACE_ID
        )
        
        deployment_id = client.deployments.get_id(function_deployment_details)
        print(f"✓ AI service deployed")
        print(f"  Deployment ID: {deployment_id}")
        
    except Exception as e:
        print(f"⚠ Failed to deploy AI service: {e}")
        deployment_id = None
else:
    print("  Skipping deployment (no service ID)")
    deployment_id = None

# %% Cell 8: Test Deployed Service
print("\n7. Testing deployed service...")

if deployment_id:
    try:
        # Test with a query that should use Wikipedia tool
        messages = [{"role": "user", "content": "Search Wikipedia for information about IBM and explain what IBM is"}]
        payload = {"messages": messages}
        
        print(f"  Query: Search Wikipedia for information about IBM and explain what IBM is")
        print("  (Processing...)")
        
        result = client.deployments.run_ai_service(deployment_id, payload)
        
        if "error" in result:
            print(f"  ⚠ Error: {result['error']}")
        else:
            response = result.get("body", {})
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    print(f"  ✓ Response received:")
                    print(f"  {content[:500]}...")
                    if len(content) > 500:
                        print(f"  (Full response length: {len(content)} characters)")
                else:
                    print(f"  ⚠ Empty content in response")
                    print(f"  Full response: {response}")
            else:
                print(f"  ⚠ No choices in response")
                print(f"  Response: {response}")
        
    except Exception as e:
        print(f"  ⚠ Test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  Skipping test (no deployment)")

# %% Cell 9: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 AI Service capabilities:")
print("  ✓ Agent with tool integration")
print("  ✓ Wikipedia search tool")
print("  ✓ Conversational memory")
print("  ✓ Streaming support")
print("\n💡 To use the deployed service:")
print("  result = client.deployments.run_ai_service(deployment_id, payload)")
print("\n📋 Deployment Information:")
if deployment_id:
    print(f"  Deployment ID: {deployment_id}")
    print(f"  Service ID: {ai_service_id}")
    print(f"  Space ID: {WATSONX_SPACE_ID}")

