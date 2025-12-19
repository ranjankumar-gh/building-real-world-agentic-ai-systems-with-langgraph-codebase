# Module 01 - Chatbot and Agent Implementations

This module contains implementations using Qwen3:8b hosted locally on Ollama:
1. **Baseline Chatbot** - Simple question-answering chatbot
2. **Agent with Tools** - Advanced agent that can use tools to accomplish tasks
3. **Local Weather API Service** - FastAPI service that mimics OpenWeatherMap API for local development

## Prerequisites

### Install Ollama

1. Download and install Ollama from [https://ollama.com/](https://ollama.com/)
2. Pull the Qwen3:8b model:

```bash
ollama pull qwen3:8b
```

3. Verify Ollama is running:

```bash
ollama list
```

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Implementations

**Baseline Chatbot:**
```bash
python baseline_chatbot.py
```

**Agent with Tools:**
```bash
python agent_v1.py
```

**Local Weather API Service (run in separate terminal):**
```bash
python weather_api.py
```

## Implementations

### baseline_chatbot.py

A simple chatbot that responds to user messages without any tools or capabilities beyond text generation.

**Features:**
- System prompt: "You are a helpful assistant."
- Direct message handling without conversation history
- Three test queries demonstrating basic capabilities

**How it works:**
- Uses Ollama's chat API with Qwen3:8b
- Each query is independent with no memory of previous interactions

### agent_v1.py

An advanced implementation that extends the baseline chatbot with tool-calling capabilities. The agent can use external tools to accomplish tasks.

**Available Tools:**

1. **calculator** - Perform basic math operations (add, subtract, multiply, divide)
2. **get_weather** - Get current weather for a city (uses local weather service)
3. **send_email** - Send an email (simulated for demo purposes)

**How it works:**
- The agent receives a user request
- If needed, it decides which tool(s) to use
- Executes the selected tools with appropriate parameters
- Uses tool results to formulate the final response
- Supports multi-step reasoning with tool chaining

**Example interactions:**
- "What's 25 * 17?" → Uses calculator tool
- "What's the weather in Tokyo?" → Uses get_weather tool (works with local weather service)
- "Send an email to mail@ranjankumar.in" → Uses send_email tool (simulated)

### weather_api.py

A local FastAPI service that provides mock weather data.

**Features:**
- RESTful API format for weather data
- Pre-configured mock data for major cities (Tokyo, New York, London, Paris, Sydney)
- Random weather generation for unknown cities
- API key authentication
- Interactive API documentation at `/docs`

**How it works:**
- Runs as a standalone FastAPI server on port 8000
- Accepts requests at `/data/2.5/weather?q=CITY&appid=API_KEY`
- Returns weather data in JSON format
- No external API calls - all data is mocked locally

## Files

- `baseline_chatbot.py` - Simple chatbot implementation
- `agent_v1.py` - Agent with tool-calling capabilities
- `weather_api.py` - Local weather API service (FastAPI)
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys, URLs)
- `.env.example` - Template for environment variables
- `README.md` - This file

## Configuration

### Weather API Service

The project uses a local weather service for development:

1. The `.env` file is already configured:
   ```
   WEATHER_API_KEY=test_api_key_12345
   WEATHER_API_URL=http://localhost:8000/data/2.5/weather
   ```

2. Start the local weather service in a separate terminal:
   ```bash
   python weather_api.py
   ```

3. Run the agent (in another terminal):
   ```bash
   python agent_v1.py
   ```

### Additional Notes

- The calculator and email tools work without any API configuration
- The email tool is simulated and doesn't require any configuration
- You can access the weather API docs at http://localhost:8000/docs when running locally
- All weather data is mocked locally - no external API calls are made

## Notes

- All implementations run completely locally using Ollama
- No external API keys or services required
- Ollama must be running in the background
- The chatbot does not maintain conversation history between calls
- Tool calling with Ollama requires models that support function calling
- Weather data is mocked locally for testing and development
