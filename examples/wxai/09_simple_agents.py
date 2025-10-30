"""
Example 09: Simple Agent with Tools
Demonstrates a simple agent with tool integration using IBM watsonx SDK.
Based on: samples/Use watsonx, and mistral-large with support for tools to perform simple calculations.ipynb

Self-Contained Design:
- Simple tool integration
- Use # %% cell markers for easy conversion to notebook
"""

import os
from ibm_watsonx_ai import Credentials
from langchain_ibm import ChatWatsonx
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv

# %% [markdown]
# # Example 09: Simple Agent with Tools
# This demonstrates a simple agent with tool capabilities.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
MODEL_NAME = os.getenv('MODEL_NAME')

print("="*60)
print("EXAMPLE 09: Simple Agent with Tools")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

print(f"✓ Credentials initialized")
print(f"  Project ID: {WATSONX_PROJECT_ID}")

# %% Cell 3: Initialize Chat Model
print("\n2. Creating chat model...")

chat = ChatWatsonx(
    url=credentials["url"],
    apikey=credentials["apikey"],
    model_id=MODEL_NAME,
    project_id=WATSONX_PROJECT_ID,
    temperature=0.0,
    max_tokens=500,  # 토큰 수를 줄여서 더 빠르게
    request_timeout=30  # timeout 추가
)

print(f"✓ Chat model created: {MODEL_NAME}")

# %% Cell 4: Define Tools
print("\n3. Defining tools...")

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition operations."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number. Use this for subtraction operations."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for multiplication operations."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number. Use this for division operations."""
    return a / b

tools = [add, subtract, multiply, divide]
print(f"✓ Tools defined: {len(tools)}")

# %% Cell 5: Create Agent with LangGraph
print("\n4. Creating agent with LangGraph...")

def create_react_agent_graph(model, tools):
    """Create a ReAct agent using LangGraph StateGraph."""
    
    def should_continue(state):
        """Determine if agent should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If last message is AIMessage and has tool calls, go to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end (final answer or error)
        return END
    
    # Bind tools to model
    if tools:
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model
    
    def call_model(state):
        """Call the model with tools."""
        messages = state["messages"]
        
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def call_tools(state):
        """Execute tool calls."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_messages = []
        
        # Check if last message has tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            # Find the tool
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
                            content=str(result),
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
    
    # Create graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile and return
    return workflow.compile()

try:
    graph = create_react_agent_graph(chat, tools)
    print(f"✓ Agent created successfully with LangGraph")
except Exception as e:
    print(f"⚠ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# %% Cell 6: Test Agent
print("\n5. Testing agent...")

def print_stream(stream, verbose=False):
    """Print messages from the agent."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            if verbose:
                print(message)
        else:
            if verbose:
                message.pretty_print()
            else:
                # 마지막 답변만 출력
                if hasattr(message, 'content') and message.content:
                    print(f"  Answer: {message.content}...")

# 설정 - recursion limit 적절히 설정
config = {"recursion_limit": 50}  # 복잡한 계산을 위해 증가

try:
    # Test query 1 - 더 명확한 쿼리로 시도
    query = "Calculate 11 + 13 + 20"
    inputs = {"messages": [HumanMessage(content=query)]}
    
    print(f"  Query: {query}")
    
    # 간단한 출력
    print("  (Processing...)")
    result = graph.invoke(inputs, config=config)
    
    # 마지막 메시지 확인
    last_message = result["messages"][-1]
    
    # Tool 호출 통계 확인
    tool_calls = [msg for msg in result["messages"] if hasattr(msg, 'tool_calls') and msg.tool_calls]
    tool_results = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    
    if tool_calls:
        print(f"  ✓ Tool call attempts: {len(tool_calls)}")
        print(f"  ✓ Tool executions: {len(tool_results)}")
    
    print(f"\n✓ Agent response:")
    print(f"  {last_message.content[:150]}...")
    
except Exception as e:
    print(f"⚠ Agent test failed: {e}")
    import traceback
    traceback.print_exc()
    print(f"  Continuing with other tests...")

# %% Cell 7: Additional Tests
print("\n6. Additional tests...")

test_queries = [
    "What is the result when 81 is subtracted from 100?",
    "Calculate the result of multiplying 10 by 12.",
]

for i, query in enumerate(test_queries, 1):
    try:
        print(f"  Test {i}: Processing...")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = graph.invoke(inputs, config=config)
        last_message = result["messages"][-1]
        print(f"  ✓ Query: {query}...")
        print(f"    Response: {last_message.content}...")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

# %% Cell 8: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Agent capabilities:")
print("  ✓ Tool integration (add, subtract, multiply, divide)")
print("  ✓ Automatic tool selection")
print("  ✓ Multi-step calculations")
print("\n💡 To use the agent:")
print("  from langchain_core.messages import HumanMessage")
print("  result = graph.invoke({'messages': [HumanMessage(content='your question')]})")
