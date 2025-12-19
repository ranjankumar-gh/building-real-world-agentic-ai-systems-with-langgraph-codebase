# agent_v1.py
import ollama
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define available tools
def calculator(operation: str, x: float, y: float) -> float:
    """Perform basic math operations."""
    ops = {
        "add": x + y,
        "subtract": x - y,
        "multiply": x * y,
        "divide": x / y if y != 0 else "Error: Division by zero"
    }
    return ops.get(operation, "Unknown operation")

def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    # Get API key and URL from environment variables
    api_key = os.getenv("WEATHER_API_KEY")
    weather_api_url = os.getenv("WEATHER_API_URL", "http://localhost:8000/data/2.5/weather")

    if not api_key:
        return {"error": "Weather API key not found. Please set WEATHER_API_KEY in .env file"}

    url = f"{weather_api_url}?q={city}&appid={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": round(data["main"]["temp"] - 273.15, 1),  # Convert to Celsius
                "condition": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"]
            }
        elif response.status_code == 401:
            return {"error": "Invalid API key"}
        else:
            return {"error": f"Could not fetch weather (status code: {response.status_code})"}
    except Exception as e:
        return {"error": f"Error fetching weather: {str(e)}"}

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (simulated)."""
    # In production, integrate with email service
    print(f"[SIMULATED] Sending email to {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return f"Email sent to {to}"

# Tool definitions for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic math operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                },
                "required": ["operation", "x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "calculator": calculator,
    "get_weather": get_weather,
    "send_email": send_email
}

def agent_with_tools(user_message: str) -> str:
    """Agent that can use tools to accomplish tasks."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user", "content": user_message}
    ]

    # Initial LLM call with tools
    response = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        tools=tools
    )

    # Add assistant's response to messages
    messages.append(response['message'])

    # Check if the model wants to use a tool
    if response['message'].get('tool_calls'):
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']

            # Execute the function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            # Add function response to messages
            messages.append({
                "role": "tool",
                "content": json.dumps(function_response) if not isinstance(function_response, str) else function_response
            })

        # Get final response from the model
        final_response = ollama.chat(
            model="qwen3:8b",
            messages=messages
        )
        return final_response['message']['content']

    return response['message']['content']

# Test the agent
print("=== Test 1: Calculation ===")
print(agent_with_tools("What's 25 * 17?"))

print("\n=== Test 2: Weather ===")
print(agent_with_tools("What's the weather in Tokyo?"))

print("\n=== Test 3: Email ===")
print(agent_with_tools("Send an email to john@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"))