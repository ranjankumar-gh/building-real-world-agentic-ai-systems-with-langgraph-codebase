# Module 01 - Baseline Chatbot

A simple chatbot implementation using Qwen3:8b hosted locally on Ollama.

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

### 2. Run the Chatbot

```bash
python baseline_chatbot.py
```

## How It Works

The chatbot uses the Ollama Python client to send messages to the locally-hosted Qwen3:8b model. The implementation includes:

- System prompt: "You are a helpful assistant."
- Direct message handling without conversation history
- Three test queries demonstrating basic capabilities

## Files

- `baseline_chatbot.py` - Main chatbot implementation
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Notes

- The chatbot does not maintain conversation history between calls
- Each query is independent with no memory of previous interactions
- Runs completely locally - no API keys or internet connection required
- Ollama must be running in the background for the chatbot to work
