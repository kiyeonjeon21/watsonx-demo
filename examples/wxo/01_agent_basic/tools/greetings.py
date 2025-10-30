# greetings.py
from ibm_watsonx_orchestrate.agent_builder.tools import tool


@tool
def greeting() -> str:
    """
    Greeting for everyone
    
    Returns:
        str: A greeting message
    """
    greeting = "Hello World"
    return greeting