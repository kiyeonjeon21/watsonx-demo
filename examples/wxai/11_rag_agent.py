"""
Example 11: RAG Agent
Demonstrates a RAG-enabled agent using IBM watsonx SDK.
Based on: samples/sample rag agent.ipynb

Self-Contained Design:
- RAG tool with Vector Index
- Wikipedia tool integration
- Use # %% cell markers for easy conversion to notebook
"""

import os
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.utils import Toolkit
from langchain_ibm import ChatWatsonx
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv

# %% [markdown]
# # Example 11: RAG Agent
# This demonstrates a RAG-enabled agent with document search and web search capabilities.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
VECTORIZED_DOCUMENT_ASSET_ID = os.getenv('VECTORIZED_DOCUMENT_ASSET_ID')

MODEL_NAME = os.getenv('MODEL_NAME')

print("="*60)
print("EXAMPLE 11: RAG Agent")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")
credentials = {
    "url": WATSONX_URL,
    "apikey": WATSONX_API_KEY
}

try:
    client = APIClient(credentials=credentials, project_id=WATSONX_PROJECT_ID)
    print(f"✓ SDK client initialized")
    print(f"  Project ID: {WATSONX_PROJECT_ID}")
except Exception as e:
    print(f"⚠ Client initialization failed: {e}")
    exit(1)

# %% Cell 3: Initialize Chat Model
print("\n2. Creating chat model...")

def create_chat_model():
    """Create ChatWatsonx model instance."""
    chat_model = ChatWatsonx(
        model_id=MODEL_NAME,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=WATSONX_PROJECT_ID
    )
    return chat_model

chat_model = create_chat_model()
print(f"✓ Chat model created: {MODEL_NAME}")

# %% Cell 4: Define RAG Tool
print("\n3. Creating RAG tool...")

def create_rag_tool(vector_index_id, api_client):
    """Create RAG tool for document search."""
    config = {
        "vectorIndexId": vector_index_id,
        "projectId": WATSONX_PROJECT_ID
    }

    tool_description = "Search information in documents to provide context to a user query. Useful when asked to ground the answer in specific knowledge from documents."
    
    return create_utility_agent_tool("RAGQuery", config, api_client, tool_description=tool_description)

def create_utility_agent_tool(tool_name, params, api_client, **kwargs):
    """Create a utility agent tool."""
    utility_agent_tool = Toolkit(api_client=api_client).get_tool(tool_name)
    
    tool_description = utility_agent_tool.get("description")
    
    if kwargs.get("tool_description"):
        tool_description = kwargs.get("tool_description")
    elif utility_agent_tool.get("agent_description"):
        tool_description = utility_agent_tool.get("agent_description")
    
    tool_schema = utility_agent_tool.get("input_schema")
    if tool_schema == None:
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
        """Execute the utility tool with proper error handling."""
        query = tool_input
        if utility_agent_tool.get("input_schema") == None:
            query = tool_input.get("input")
        
        # input이 dict인 경우 문자열로 변환
        if isinstance(query, dict):
            query = query.get("input", str(query))
        if not isinstance(query, str):
            query = str(query)

        try:
            results = utility_agent_tool.run(
                input=query,
                config=params
            )
            output = results.get("output") if isinstance(results, dict) else str(results)
            # 결과가 너무 길면 자르기
            if isinstance(output, str) and len(output) > 2000:
                output = output[:2000] + "... [truncated]"
            return output
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
    
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        func=run_tool,
        args_schema=tool_schema
    )

tools = []

# Add RAG tool if available
if VECTORIZED_DOCUMENT_ASSET_ID:
    try:
        rag_tool = create_rag_tool(VECTORIZED_DOCUMENT_ASSET_ID, client)
        tools.append(rag_tool)
        print("  ✓ RAG tool added")
    except Exception as e:
        print(f"  ⚠ RAG tool failed: {e}")
else:
    print("  ⚠ VECTORIZED_DOCUMENT_ASSET_ID not set, skipping RAG tool")

# Add Wikipedia tool
try:
    config = {"maxResults": 5}
    wiki_tool = create_utility_agent_tool("Wikipedia", config, client)
    tools.append(wiki_tool)
    print("  ✓ Wikipedia tool added")
except Exception as e:
    print(f"  ⚠ Wikipedia tool failed: {e}")

# %% Cell 5: Create Agent with LangGraph
print("\n4. Creating agent with LangGraph...")

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
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model
    
    def call_model(state):
        """Call the model with tools."""
        messages = state["messages"]
        if system_prompt:
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
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                tool_func = None
                for t in tools:
                    if t.name == tool_name:
                        tool_func = t
                        break
                
                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(result)[:2000] if len(str(result)) > 2000 else str(result),
                                tool_call_id=tool_call.get("id", "")
                            )
                        )
                    except Exception as e:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call.get("id", "")
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

try:
    system_prompt = """You are a helpful assistant that uses tools to answer questions.

IMPORTANT:
1. Call tools with actual function calls, not just describe them
2. After getting tool results, provide a clear, concise answer and STOP
3. Do NOT repeat calling the same tool if you already have the answer

When asked about information:
- First search documents using RAGQuery tool if available
- If needed, use Wikipedia tool for general knowledge
- After getting results, provide a direct answer and STOP."""
    
    agent = create_react_agent_graph(chat_model, tools, system_prompt)
    
    print(f"✓ Agent created successfully with LangGraph")
    print(f"  Tools: {len(tools)}")
    
except Exception as e:
    print(f"⚠ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# %% Cell 6: Test Agent
print("\n5. Testing agent...")

config = {"recursion_limit": 20}  # 무한 루프 방지를 위해 제한

# Test 1
if VECTORIZED_DOCUMENT_ASSET_ID:
    print("\n  Test 1: RAG query")
    try:
        inputs = {"messages": [HumanMessage(content="What is Watsonx?")]}
        
        print("    (Processing...)")
        result = agent.invoke(inputs, config=config)
        last_message = result["messages"][-1]
        
        print(f"    Query: What is Watsonx?")
        print(f"    Response: {last_message.content[:200]}...")
    except Exception as e:
        print(f"    ⚠ Test failed: {e}")

# Test 2
print("\n  Test 2: Wikipedia query")
try:
    inputs = {"messages": [HumanMessage(content="Tell me about artificial intelligence")]}
    
    print("    (Processing...)")
    result = agent.invoke(inputs, config=config)
    last_message = result["messages"][-1]
    
    print(f"    Query: Tell me about artificial intelligence")
    print(f"    Response: {last_message.content[:200]}...")
except Exception as e:
    print(f"    ⚠ Test failed: {e}")

# %% Cell 7: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Agent capabilities:")
if VECTORIZED_DOCUMENT_ASSET_ID:
    print("  ✓ Document search (RAG)")
print("  ✓ Web search (Wikipedia)")
print("  ✓ Tool integration")
print("\n💡 To use the agent:")
print("  from langchain_core.messages import HumanMessage")
print("  config = {'recursion_limit': 20}")
print("  result = agent.invoke({'messages': [HumanMessage(content='query')]}, config)")

