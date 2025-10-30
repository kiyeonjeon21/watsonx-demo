"""
Example 10: Complex Multi-Agent System
Demonstrates a complex multi-agent orchestration with specialized agents.
Based on: samples/Use watsonx, and mistral-large with support for tools to perform simple calculations.ipynb

Self-Contained Design:
- Multiple specialized agents
- Agent orchestration pattern
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
# # Example 06: Complex Multi-Agent System
# This demonstrates a complex 3-agent orchestration pattern.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

MODEL_NAME = os.getenv('MODEL_NAME')

print("="*60)
print("EXAMPLE 06: Complex Multi-Agent System")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

print(f"✓ Credentials initialized")
print(f"  Project ID: {WATSONX_PROJECT_ID}")

# %% Cell 3: Helper Functions
print("\n2. Setting up helper functions...")

def create_chat_model():
    """Create ChatWatsonx model instance."""
    return ChatWatsonx(
        url=credentials["url"],
        apikey=credentials["apikey"],
        model_id=MODEL_NAME,
        project_id=WATSONX_PROJECT_ID,
        temperature=0.0,
        max_tokens=500,  # 토큰 수를 줄여서 더 빠르게
        request_timeout=30  # timeout 추가
    )

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

def create_enhanced_agent(chat_model, tools, agent_type="general"):
    """Create agent using LangGraph."""
    system_prompts = {
        "calculator": "You are a specialized calculation assistant. ACTUALLY CALL the calculation tools (add, subtract, multiply, divide) to get real results. Break down complex calculations into clear steps. Provide the final answer and STOP.",
        
        "planner": "You are a task planning assistant. Analyze complex tasks and break them down into smaller steps. Focus on planning and organization, not execution.",
        
        "verifier": "You are a verification assistant. Use calculation tools to ACTUALLY VERIFY results. Call the tools and compare the results. Provide verification confirmation and STOP.",
    }
    system_prompt = system_prompts.get(agent_type, "You are a helpful assistant.")
    return create_react_agent_graph(chat_model, tools, system_prompt)

# Define calculation tools
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

calculation_tools = [add, subtract, multiply, divide]

print("✓ Helper functions ready")

# %% Cell 4: Agent 1 - Calculator
print("\n3. Creating specialized agents...")
print("\n  --- Agent 1: Calculator ---")

try:
    chat_model = create_chat_model()
    calculator = create_enhanced_agent(chat_model, tools=calculation_tools, agent_type="calculator")
    print("  ✓ Calculator agent created")
except Exception as e:
    print(f"  ⚠ Calculator agent creation failed: {e}")
    calculator = None

# %% Cell 5: Agent 2 - Planner
print("\n  --- Agent 2: Planner ---")

try:
    chat_model = create_chat_model()
    # Planner doesn't need tools
    planner = create_enhanced_agent(chat_model, tools=[], agent_type="planner")
    print("  ✓ Planner agent created")
except Exception as e:
    print(f"  ⚠ Planner agent creation failed: {e}")
    planner = None

# %% Cell 6: Agent 3 - Verifier
print("\n  --- Agent 3: Verifier ---")

try:
    chat_model = create_chat_model()
    verifier = create_enhanced_agent(chat_model, tools=calculation_tools, agent_type="verifier")
    print("  ✓ Verifier agent created")
except Exception as e:
    print(f"  ⚠ Verifier agent creation failed: {e}")
    verifier = None

# %% Cell 7: Test Agents
print("\n4. Testing agents...")

def safe_test_agent(agent, query, agent_name, config=None):
    """Test an agent with a query and proper error handling."""
    if config is None:
        config = {"recursion_limit": 30}
    
    try:
        inputs = {"messages": [HumanMessage(content=query)]}
        result = agent.invoke(inputs, config)
        last_message = result["messages"][-1]
        return last_message.content
    except Exception as e:
        print(f"  ⚠ {agent_name} test failed: {e}")
        return None

# 설정 - 09_simple_agents.py와 동일하게
config = {
    "recursion_limit": 30,  # 적절한 limit
}

# Test Calculator
if calculator:
    test_query = "Add the numbers 2, 3, multiply by 6, then divide by 10"
    print(f"\n  Calculator test:")
    print(f"    Query: {test_query}")
    print("    (Processing...)")
    response = safe_test_agent(calculator, test_query, "Calculator", config)
    if response:
        print(f"    Response: {response}")

# %% Cell 8: Orchestration Demo
print("\n5. Multi-agent orchestration demo...")

if calculator:
    print("\n  Task: Complex calculation workflow")
    
    # Phase 1: Calculate
    phase1_query = "Calculate 15 * 8"
    print(f"\n  Phase 1 - Calculate:")
    print(f"    {phase1_query}")
    print("    (Processing...)")
    result1 = safe_test_agent(calculator, phase1_query, "Calculator", config)
    if result1:
        print(f"    Result: {result1}")
    
    # Phase 2: Verify (if verifier is available)
    if verifier and result1:
        phase2_query = "Verify: 15 * 8 = 120"
        print(f"\n  Phase 2 - Verify:")
        print(f"    {phase2_query}")
        print("    (Processing...)")
        result2 = safe_test_agent(verifier, phase2_query, "Verifier", config)
        if result2:
            print(f"    Verification: {result2}")
    
    # Phase 3: Plan next steps (if planner is available)
    if planner:
        phase3_query = "Plan steps for solving: (a + b) * c / d"
        print(f"\n  Phase 3 - Plan:")
        print(f"    {phase3_query}")
        print("    (Processing...)")
        result3 = safe_test_agent(planner, phase3_query, "Planner", config)
        if result3:
            print(f"    Plan: {result3}")

# %% Cell 9: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Multi-agent capabilities:")
print("  ✓ Calculator agent with tool integration")
print("  ✓ Planner agent for task planning")
print("  ✓ Verifier agent for result verification")
print("  ✓ Sequential agent workflow")
print("\n💡 To use agents:")
print("  from langchain_core.messages import HumanMessage")
print("  result = agent.invoke({'messages': [HumanMessage(content='task')]})")
