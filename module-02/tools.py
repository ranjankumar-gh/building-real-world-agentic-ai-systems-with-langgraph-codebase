"""
Tools module for LangChain agents.
Contains all tool definitions and implementations.
"""

from langchain.tools import tool


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., '15 * 20 / 100')

    Returns:
        Result of the calculation
    """
    try:
        # Security: Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/().  ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"

        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"

    except Exception as e:
        return f"Error: {str(e)}"


# Tool 2: Web Search (simulated)
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)

    Returns:
        Search results as formatted string
    """
    # In production, integrate with actual search API
    # This is a simulation

    # Simulate API call
    simulated_results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "snippet": f"This is a simulated search result about {query}...",
            "url": f"https://example.com/result{i+1}"
        }
        for i in range(min(max_results, 3))
    ]

    # Format results
    formatted = f"Search results for '{query}':\n\n"
    for i, result in enumerate(simulated_results, 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   {result['snippet']}\n"
        formatted += f"   {result['url']}\n\n"

    return formatted


# Tool 3: Weather API
@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.

    Args:
        city: City name

    Returns:
        Weather information
    """
    # In production, integrate with weather API
    # This is a simulation

    import random

    temps = list(range(5, 30))
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Overcast"]

    temp = random.choice(temps)
    condition = random.choice(conditions)

    return f"Current weather in {city}: {temp}°C, {condition}"


# Export tools
calculator_tool = calculator
web_search_tool = web_search
weather_tool = get_weather
