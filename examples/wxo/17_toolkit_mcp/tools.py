"""
Basic math tools for watsonx Orchestrate
"""

def add(a: float, b: float) -> str:
    """
    Add two numbers
    
    Args:
        a (float): First number
        b (float): Second number
        
    Returns:
        str: Result of addition
    """
    result = a + b
    return f"The sum of {a} and {b} is {result}"


def subtract(a: float, b: float) -> str:
    """
    Subtract two numbers
    
    Args:
        a (float): First number  
        b (float): Second number
        
    Returns:
        str: Result of subtraction
    """
    result = a - b
    return f"The difference of {a} and {b} is {result}"


def multiply(a: float, b: float) -> str:
    """
    Multiply two numbers
    
    Args:
        a (float): First number
        b (float): Second number
        
    Returns:
        str: Result of multiplication
    """
    result = a * b
    return f"The product of {a} and {b} is {result}"


def divide(a: float, b: float) -> str:
    """
    Divide two numbers
    
    Args:
        a (float): Dividend
        b (float): Divisor
        
    Returns:
        str: Result of division or error message
    """
    if b == 0:
        return "Error: Cannot divide by zero"
    
    result = a / b
    return f"The quotient of {a} divided by {b} is {result}"
