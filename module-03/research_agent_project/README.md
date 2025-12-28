# Research Agent with LangGraph

A production-ready, deterministic research agent built with LangGraph that demonstrates explicit state management, conditional routing, checkpointing, and retry logic.

## Features

✅ **Explicit State Management** - All agent state is observable and inspectable  
✅ **Deterministic Flow** - Predictable execution paths with conditional routing  
✅ **Checkpointing** - Resume from failures without starting over  
✅ **Retry Logic** - Intelligent error handling with configurable retries  
✅ **Multi-Stage Pipeline** - Planning → Searching → Validation → Processing → Generation  
✅ **Full Observability** - Stream execution and monitor progress in real-time  

## What This Agent Does

1. **Planning**: Breaks down your research query into specific search queries
2. **Searching**: Executes web searches using DuckDuckGo
3. **Validation**: Checks if results are sufficient (with retry on failure)
4. **Processing**: Extracts key findings from search results
5. **Generation**: Creates a structured research report
6. **Error Handling**: Retries failed searches or fails gracefully

## Project Structure

```
research_agent_project/
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── state.py           # State definition
│       ├── tools.py           # Search tools
│       ├── nodes.py           # Agent nodes (planning, search, etc.)
│       ├── graph.py           # Graph construction and routing
│       └── config.py          # Configuration
├── examples/
│   ├── basic_usage.py         # Simple example
│   ├── streaming_example.py   # Real-time streaming
│   └── checkpoint_example.py  # Checkpoint and resume
├── tests/
│   └── test_agent.py          # Unit tests
├── requirements.txt           # Dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
└── setup.py                   # Package installation
```

## Installation

### 1. Clone or Download

```bash
# If you have the project files
cd research_agent_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Quick Start

### Basic Usage

```python
from research_agent import create_research_agent, ResearchAgentState

# Create agent
agent = create_research_agent()

# Run research
initial_state = {
    "research_query": "Latest developments in quantum computing",
    "messages": [],
    "search_queries": [],
    "search_results": [],
    "key_findings": [],
    "report": "",
    "current_stage": "planning",
    "retry_count": 0,
    "max_retries": 2,
    "error_message": None,
    "research_plan": ""
}

config = {"configurable": {"thread_id": "research-001"}}
result = agent.invoke(initial_state, config=config)

print(result["report"])
```

### Streaming Execution

```python
from research_agent import create_research_agent

agent = create_research_agent()
config = {"configurable": {"thread_id": "research-002"}}

for step in agent.stream(initial_state, config=config):
    node_name = list(step.keys())[0]
    print(f"Executing: {node_name}")
```

### With Checkpointing

```python
from research_agent import create_research_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# Use persistent checkpointing
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
agent = create_research_agent(checkpointer=checkpointer)

# Execute
config = {"configurable": {"thread_id": "research-003"}}
result = agent.invoke(initial_state, config=config)

# Resume from checkpoint if needed
resumed = agent.invoke(None, config=config)
```

## Configuration

Edit `src/research_agent/config.py` to customize:

- **LLM Model**: Change between GPT-4, Claude, etc.
- **Temperature**: Adjust for deterministic/creative output
- **Max Retries**: Set retry limit
- **Search Limit**: Number of searches to execute

## Examples

### 1. Basic Research

```bash
python examples/basic_usage.py
```

### 2. Streaming with Progress

```bash
python examples/streaming_example.py
```

### 3. Checkpoint and Resume

```bash
python examples/checkpoint_example.py
```

## How It Works

### State Flow

```
START → PLANNING → SEARCHING → VALIDATING → PROCESSING → GENERATING → END
          ↓            ↓           ↓
        ERROR ← → RETRY_SEARCH
```

### State Management

All agent state is explicit and typed:

- **Conversation**: Messages between LLM and agent
- **Task**: Research query and plan
- **Search**: Queries and results
- **Processing**: Key findings
- **Control Flow**: Current stage, retry count, errors

### Conditional Routing

The agent makes decisions at validation:
- **Valid results** → Process findings
- **Insufficient results** → Retry search (up to max_retries)
- **Max retries exceeded** → Fail gracefully

### Checkpointing

State is saved after every node execution:
- Resume from failures automatically
- No duplicate work
- Full execution history

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/research_agent
```

## Production Deployment

### Use Persistent Checkpointing

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)
agent = create_research_agent(checkpointer=checkpointer)
```

### Add Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_agent")

# Agent will log all state transitions
```

### Scale with Thread Pools

```python
from concurrent.futures import ThreadPoolExecutor

def research_task(query):
    state = create_initial_state(query)
    config = {"configurable": {"thread_id": f"research-{query}"}}
    return agent.invoke(state, config=config)

with ThreadPoolExecutor(max_workers=10) as executor:
    queries = ["AI trends", "Climate tech", "Biotech advances"]
    results = executor.map(research_task, queries)
```

## Troubleshooting

### API Key Issues

```
Error: OpenAI API key not found
```

**Solution**: Set `OPENAI_API_KEY` in `.env` file

### Search Failures

```
Error: Insufficient search results
```

**Solution**: 
- Check internet connection
- Increase `max_retries` in config
- Use alternative search tool (SerpAPI, Bing)

### Rate Limits

```
Error: Rate limit exceeded
```

**Solution**:
- Add exponential backoff
- Reduce concurrent requests
- Use slower tier API key

## Extensions

### 1. Add Human-in-the-Loop

See `examples/human_approval.py` for approval workflow

### 2. Parallel Searches

See `examples/parallel_search.py` for concurrent execution

### 3. Cost Tracking

See `examples/cost_tracking.py` for token usage monitoring

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Learn More

This project demonstrates concepts from **Module 3: Deterministic Agent Flow with LangGraph**

Key concepts:
- State management
- Graph-based workflows
- Checkpointing and retries
- Conditional routing
- Production-ready patterns

For detailed explanations, see the module documentation.

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Start a discussion
- **Documentation**: See `/docs` folder

---

Built with ❤️ using [LangGraph](https://github.com/langchain-ai/langgraph)
