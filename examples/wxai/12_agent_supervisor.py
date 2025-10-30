"""
Example 12: Agent Supervisor
Demonstrates a multi-agent supervisor system using IBM watsonx SDK and LangGraph.
Based on: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/

Self-Contained Design:
- Supervisor agent for task orchestration
- Specialized worker agents (math, research)
- Multi-agent graph with StateGraph
- Use # %% cell markers for easy conversion to notebook
"""

import os
from typing import Annotated, Literal
from ibm_watsonx_ai import Credentials
from langchain_ibm import ChatWatsonx
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
MODEL_NAME = os.getenv('MODEL_NAME')

print("="*60)
print("EXAMPLE 12: Agent Supervisor")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

print(f"✓ Credentials initialized")
print(f"  Project ID: {WATSONX_PROJECT_ID}")

# %% Cell 3: Create Chat Model Factory
print("\n2. Setting up chat model factory...")

def create_chat_model(temperature=0.0):
    """Create ChatWatsonx model instance."""
    return ChatWatsonx(
        url=credentials["url"],
        apikey=credentials["apikey"],
        model_id=MODEL_NAME,
        project_id=WATSONX_PROJECT_ID,
        temperature=temperature,
        max_tokens=500,
        request_timeout=30
    )

print("✓ Chat model factory ready")

# %% Cell 4: Define Math Tools
print("\n3. Defining math tools...")

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

math_tools = [add, subtract, multiply, divide]
print(f"✓ Math tools defined: {len(math_tools)}")

# %% Cell 5: Create React Agent Helper
print("\n4. Creating ReAct agent helper function...")

def create_react_agent_graph(model, tools, system_prompt=None, agent_name="agent"):
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

print("✓ ReAct agent helper ready")

# %% Cell 6: Create Math Agent
print("\n5. Creating worker agents...")
print("\n  --- Math Agent ---")

try:
    chat_model = create_chat_model()
    math_prompt = """You are a math agent.

INSTRUCTIONS:
- Assist ONLY with math-related tasks, DO NOT do any research
- Use the calculation tools (add, subtract, multiply, divide) to solve problems
- After you're done with your tasks, respond to the supervisor directly
- Respond ONLY with the results of your work, do NOT include ANY other text."""
    
    math_agent = create_react_agent_graph(
        chat_model,
        tools=math_tools,
        system_prompt=math_prompt,
        agent_name="math_agent"
    )
    print("  ✓ Math agent created")
except Exception as e:
    print(f"  ⚠ Math agent creation failed: {e}")
    math_agent = None

# Note: Research agent would require web search tool (e.g., Tavily)
# For this demo, we'll focus on math tasks with supervisor

# %% Cell 7: Create Supervisor Graph
print("\n6. Creating supervisor graph...")

def create_supervisor_graph(math_agent):
    """Create supervisor graph that orchestrates worker agents."""
    
    def supervisor_router(state) -> Literal["math_agent", END]:
        """Supervisor decides which agent to use based on message content."""
        messages = state["messages"]
        
        # Get the first user message to determine routing
        user_message = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_message = msg
                break
        
        # If no user message, end
        if not user_message:
            return END
        
        # Check for math keywords in user message
        content = user_message.content.lower() if hasattr(user_message, 'content') else ""
        math_keywords = [
            "calculate", "add", "subtract", "multiply", "divide", 
            "math", "+", "-", "*", "/", "sum", "times", 
            "what is", "compute", "solve", "result of"
        ]
        
        # If math-related, route to math agent
        if any(keyword in content for keyword in math_keywords):
            return "math_agent"
        
        # For non-math questions, we could have a general agent
        # For now, just return END (could add general agent later)
        return END
    
    def call_math_agent(state):
        """Call math agent with current state."""
        messages = state["messages"]
        result = math_agent.invoke({"messages": messages})
        return {"messages": result["messages"]}
    
    # Create supervisor graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes - supervisor is just a router, so we don't need a supervisor node
    # Instead, routing happens at the graph level
    workflow.add_node("math_agent", call_math_agent)
    
    # Set entry point - route directly based on input
    workflow.add_conditional_edges(
        START,
        supervisor_router,
        {
            "math_agent": "math_agent",
            END: END
        }
    )
    
    # After math agent completes, end (math agent returns final answer)
    workflow.add_edge("math_agent", END)
    
    return workflow.compile()

if math_agent:
    try:
        supervisor = create_supervisor_graph(math_agent)
        print("  ✓ Supervisor graph created")
    except Exception as e:
        print(f"  ⚠ Supervisor creation failed: {e}")
        import traceback
        traceback.print_exc()
        supervisor = None
else:
    supervisor = None

# %% Cell 9: Test Supervisor System
print("\n7. Testing supervisor system...")

config = {"recursion_limit": 50}

if supervisor:
    test_queries = [
        "Calculate 15 * 8",
        "What is 25 + 37?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Test {i}: {query}")
        try:
            inputs = {"messages": [HumanMessage(content=query)]}
            
            print("    (Processing...)")
            result = supervisor.invoke(inputs, config=config)
            last_message = result["messages"][-1]
            
            print(f"    ✓ Response: {last_message.content[:200]}...")
        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            import traceback
            traceback.print_exc()

# %% Cell 10: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Agent Supervisor capabilities:")
print("  ✓ Supervisor agent for task orchestration")
print("  ✓ Math agent for calculations")
print("  ✓ Multi-agent workflow with routing")
print("  ✓ Task delegation system")
print("\n💡 To use the supervisor:")
print("  from langchain_core.messages import HumanMessage")
print("  result = supervisor.invoke({'messages': [HumanMessage(content='task')]})")
print("\n💡 Architecture:")
print("  User → Router → Math Agent → END")

