# agent_v2.py
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

def intelligent_agent(user_message: str, conversation_history: list = None) -> dict:
    """
    Agent with decision logic:
    - Maintains conversation history (memory)
    - Makes decisions about which tools to use
    - Provides reasoning for actions
    """
    if conversation_history is None:
        conversation_history = []
    
    # Add system prompt with decision-making instructions
    messages = [
        {
            "role": "system", 
            "content": """You are an intelligent assistant that can:
1. Use tools when needed
2. Explain your reasoning
3. Ask for clarification if needed
4. Remember previous context

Before using a tool, briefly explain why you're using it.
If you cannot complete a task, explain what's missing."""
        }
    ]
    
    # Add conversation history
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    
    # Agent loop with reasoning
    response = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        tools=tools
    )

    response_message = response['message']
    reasoning = response_message.get('content', "Using tools to help...")

    messages.append(response_message)
    
    # Execute tools if needed
    tool_results = []
    if response_message.get('tool_calls'):
        for tool_call in response_message['tool_calls']:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']

            print(f"[AGENT REASONING] {reasoning}")
            print(f"[AGENT ACTION] Calling {function_name} with {function_args}")

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            tool_results.append({
                "tool": function_name,
                "input": function_args,
                "output": function_response
            })

            messages.append({
                "role": "tool",
                "content": json.dumps(function_response) if not isinstance(function_response, str) else function_response
            })

        # Get final response
        final_response = ollama.chat(
            model="qwen3:8b",
            messages=messages
        )

        return {
            "response": final_response['message']['content'],
            "reasoning": reasoning,
            "actions": tool_results,
            "conversation_history": messages
        }

    return {
        "response": response_message.get('content', ''),
        "reasoning": "No tools needed",
        "actions": [],
        "conversation_history": messages
    }

# Test with multi-turn conversation
print("=== Conversation Test ===")
history = []

# Turn 1
result1 = intelligent_agent("My name is Alice and I'm planning a trip to Tokyo", history)
print(f"Assistant: {result1['response']}\n")
history = result1['conversation_history']

# Turn 2
result2 = intelligent_agent("What's the weather there?", history)
print(f"Reasoning: {result2['reasoning']}")
print(f"Actions: {result2['actions']}")
print(f"Assistant: {result2['response']}\n")
history = result2['conversation_history']

# Turn 3
result3 = intelligent_agent("What was my name again?", history)
print(f"Assistant: {result3['response']}")