# Module 02: Production-Ready Agent with LangChain and Ollama

A production-ready AI agent implementation using LangChain with Ollama (Qwen3:8b), featuring tool calling, memory management, error handling, and execution tracking.

## Overview

This module demonstrates how to build a robust single-loop agent that can:
- Use multiple tools (calculator, web search, weather)
- Maintain conversation history (memory)
- Handle errors gracefully
- Track execution metrics
- Provide detailed intermediate steps

## Project Structure

```
module-02/
├── production_agent.py    # Main agent implementation
├── tools.py              # Tool definitions
├── requirements.txt      # Project dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Features

### 1. **Tool Calling**
The agent has access to three tools:
- **Calculator**: Evaluates mathematical expressions (e.g., "15 * 20 / 100")
- **Web Search**: Simulated web search for current information
- **Weather API**: Simulated weather data for any city

### 2. **Memory Management**
- Maintains conversation history across multiple turns
- Remembers context from previous messages
- Enables natural multi-turn conversations

### 3. **Error Handling**
- Try-catch blocks for robust execution
- Graceful error messages
- Error logging in execution metrics

### 4. **Execution Tracking**
- Tracks all agent executions
- Records execution time, steps taken, and success rate
- Provides detailed statistics

### 5. **Intermediate Steps**
- Returns all intermediate tool calls and observations
- Useful for debugging and understanding agent behavior

## Installation

### Quick Start

```bash
# 1. Navigate to module-02
cd module-02

# 2. Create virtual environment
python -m venv env

# 3. Activate virtual environment
# Windows: env\Scripts\activate
# macOS/Linux: source env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install and start Ollama (if not already done)
ollama pull qwen3:8b
ollama serve

# 6. Run the agent
python production_agent.py
```

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running locally
- Qwen3:8b model pulled in Ollama (`ollama pull qwen3:8b`)

### Step 1: Create a Virtual Environment

It's recommended to use a virtual environment to isolate project dependencies.

Navigate to the module-02 directory:

```bash
cd module-02
```

**Create a virtual environment:**

On Windows:
```bash
python -m venv env
```

On macOS/Linux:
```bash
python3 -m venv env
```

**Activate the virtual environment:**

On Windows (Command Prompt):
```bash
env\Scripts\activate
```

On Windows (PowerShell):
```bash
env\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
source env/bin/activate
```

You should see `(env)` prefix in your terminal, indicating the virtual environment is active.

**Deactivate the virtual environment** (when you're done working):

On all platforms:
```bash
deactivate
```

**Important Note:** The `env/` directory should be added to `.gitignore` to avoid committing virtual environment files to version control. This is already configured in most projects.

### Step 2: Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `langchain` - Core LangChain library
- `langchain-core` - Core LangChain components
- `langchain-ollama` - Ollama integration for LangChain
- `python-dotenv` - Environment variable management
- `requests` - HTTP library
- `pydantic` - Data validation

### Step 3: Install and Set Up Ollama

1. **Install Ollama**: Download and install from [https://ollama.ai](https://ollama.ai)

2. **Pull the Qwen3:8b model**:
```bash
ollama pull qwen3:8b
```

3. **Start Ollama** (if not already running):
```bash
ollama serve
```

Ollama will run on `http://localhost:11434` by default.

### Step 4: Optional Environment Configuration

You can optionally create a `.env` file for custom configuration:

```bash
cp .env.example .env
```

The default configuration works with Ollama running locally, but you can customize:

```
# Optional: Custom Ollama URL
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Different model
MODEL_NAME=qwen3:8b
```

## Usage

### Running the Agent

To run the production agent with all test cases:

```bash
python production_agent.py
```

This will execute four test scenarios:
1. **Calculator Test**: "What is 15% of 2500?"
2. **Weather Test**: "What's the weather like in Tokyo?"
3. **Web Search Test**: "Who won the 2024 Nobel Prize in Physics?"
4. **Memory Test**: Multi-turn conversation demonstrating context retention

### Using the Agent in Your Code

```python
from production_agent import ProductionAgent

# Initialize the agent
agent = ProductionAgent(
    model_name="qwen3:8b",
    temperature=0,
    max_iterations=10,
    verbose=True
)

# Run a query
result = agent.run("What is 25 * 47?")

# Access the results
print(f"Answer: {result['answer']}")
print(f"Execution Time: {result['execution_time']:.2f}s")
print(f"Steps: {result['intermediate_steps']}")

# Get execution statistics
stats = agent.get_execution_stats()
print(stats)

# Reset conversation history
agent.reset_conversation()
```

### Customizing the Agent

You can customize the agent behavior by modifying initialization parameters:

```python
agent = ProductionAgent(
    model_name="qwen3:8b",         # Choose your Ollama model
    temperature=0.7,               # Adjust creativity (0-1)
    max_iterations=15,             # Maximum tool calls per query
    verbose=False                  # Disable verbose logging
)
```

## Adding Custom Tools

To add your own tools, edit [tools.py](tools.py):

1. **Define and Implement the Tool Function with @tool decorator**:
```python
from langchain.tools import tool

@tool
def your_custom_tool(param: str) -> str:
    """
    Your tool description - this will be used by the LLM to understand when to use it.

    Args:
        param: Parameter description

    Returns:
        Result string
    """
    # Your implementation
    return "Result"
```

2. **Export the Tool**:
```python
# At the bottom of tools.py
your_custom_tool_export = your_custom_tool
```

3. **Add to Agent Tools List** in [production_agent.py](production_agent.py):
```python
from tools import calculator_tool, web_search_tool, weather_tool, your_custom_tool

# In ProductionAgent.__init__:
self.tools = [
    calculator_tool,
    web_search_tool,
    weather_tool,
    your_custom_tool  # Add your tool here
]
```

## Architecture

### Agent Flow

1. **User Input** → Agent receives query
2. **Reasoning** → LLM decides which tool(s) to use
3. **Tool Execution** → Selected tools are called with appropriate inputs
4. **Observation** → Tool results are collected
5. **Final Response** → LLM synthesizes final answer
6. **Memory Update** → Conversation history is updated

### Key Components

- **ProductionAgent**: Main agent class that orchestrates execution
- **ChatOllama**: LLM interface for Ollama models (Qwen3:8b)
- **create_agent**: Creates an agent graph with tool calling loop
- **@tool decorator**: Wraps Python functions as LangChain tools
- **System Prompt**: Defines agent behavior and tool usage guidelines
- **Message History**: Stores conversation context

## Output Format

The `run()` method returns a dictionary with:

```python
{
    "answer": str,              # Final answer from the agent
    "intermediate_steps": [     # List of tool calls made
        {
            "tool": str,        # Tool name
            "input": dict,      # Tool input parameters
            "output": str       # Tool output
        }
    ],
    "execution_time": float,    # Time in seconds
    "success": bool             # Whether execution succeeded
}
```

## Execution Statistics

The agent tracks detailed metrics:

```python
{
    "total_executions": int,           # Total number of runs
    "successful": int,                 # Number of successful runs
    "success_rate": float,             # Percentage of successful runs
    "avg_execution_time": float,       # Average time per execution
    "avg_steps_per_execution": float   # Average tool calls per execution
}
```

## Example Output

```
============================================================
PRODUCTION AGENT - SINGLE-LOOP WITH TOOLS
============================================================

============================================================
TEST 1: Calculator Tool
============================================================

> Entering new AgentExecutor chain...

Invoking: `calculator` with `{'expression': '15 * 2500 / 100'}`

Result: 375.0

15% of 2500 is 375.

> Finished chain.

Final Answer: 15% of 2500 is 375.
Execution Time: 1.23s
Steps Taken: 1
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check that Ollama is accessible at `http://localhost:11434`
   - Verify the model is pulled: `ollama pull qwen3:8b`

2. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're running Python 3.8 or higher
   - Install langchain-ollama: `pip install langchain-ollama`

3. **Model Not Found**
   - Pull the model: `ollama pull qwen3:8b`
   - List available models: `ollama list`

4. **Tool Not Found**
   - Verify tool imports in [production_agent.py](production_agent.py)
   - Check that tools are properly defined in [tools.py](tools.py)

## Next Steps

- Integrate real APIs for web search and weather
- Add more sophisticated tools (database queries, file operations, etc.)
- Implement persistent memory with vector databases
- Add streaming responses for better UX
- Deploy as a web service with FastAPI

## License

This project is part of the "Building Real-World Agentic AI Systems with LangGraph" course.

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)
- [Pydantic Documentation](https://docs.pydantic.dev/)
