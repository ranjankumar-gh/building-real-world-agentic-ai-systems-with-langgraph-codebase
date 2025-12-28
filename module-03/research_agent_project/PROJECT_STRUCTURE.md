# Project Structure

Detailed overview of the Research Agent project organization.

## Directory Tree

```
research_agent_project/
├── src/
│   └── research_agent/          # Main package
│       ├── __init__.py          # Package exports
│       ├── state.py             # State definition
│       ├── config.py            # Configuration
│       ├── tools.py             # Search tools
│       ├── nodes.py             # Agent nodes
│       └── graph.py             # Graph construction
├── examples/
│   ├── basic_usage.py           # Simple example
│   ├── streaming_example.py     # Streaming execution
│   └── checkpoint_example.py    # Checkpointing demo
├── tests/
│   └── test_agent.py            # Unit tests
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── README.md                    # Main documentation
├── QUICKSTART.md                # Quick start guide
├── requirements.txt             # Dependencies
└── setup.py                     # Package setup
```

## Module Descriptions

### `src/research_agent/`

The main package containing all agent logic.

#### `__init__.py`
- Package initialization
- Public API exports
- Version information

**Exports:**
- `create_research_agent()` - Main factory function
- `create_initial_state()` - State creation helper
- `ResearchAgentState` - State type
- `AgentConfig` - Configuration class
- Individual node functions (for advanced use)

#### `state.py`
- `ResearchAgentState` TypedDict definition
- `create_initial_state()` helper function
- All state fields documented

**Key Fields:**
- `messages` - Conversation history
- `research_query` - Original query
- `search_queries` - Generated queries
- `search_results` - Search output
- `key_findings` - Extracted insights
- `report` - Final output
- `current_stage` - Execution stage
- `retry_count` - Retry tracking

#### `config.py`
- `AgentConfig` dataclass
- Configuration from environment
- Default settings

**Configurable:**
- LLM model and temperature
- Retry limits
- Search parameters
- API keys

#### `tools.py`
- Search tool creation
- DuckDuckGo integration
- Extensible for other search providers

**Functions:**
- `create_search_tool()` - Create DuckDuckGo tool
- `create_search_tools()` - List of all tools

#### `nodes.py`
- All 6 agent nodes
- LLM creation
- Node logic

**Nodes:**
1. `plan_research()` - Planning
2. `execute_search()` - Search execution
3. `validate_results()` - Validation
4. `process_results()` - Processing
5. `generate_report()` - Report generation
6. `handle_error()` - Error handling

#### `graph.py`
- Graph construction
- Routing functions
- Agent compilation

**Functions:**
- `create_workflow()` - Build graph
- `create_research_agent()` - Main entry point
- `route_after_validation()` - Validation router
- `route_after_error()` - Error router

### `examples/`

Executable examples demonstrating different features.

#### `basic_usage.py`
- Simplest usage
- Single research query
- Print final report
- Good starting point

#### `streaming_example.py`
- Real-time streaming
- Progress monitoring
- Step-by-step output
- Shows all node executions

#### `checkpoint_example.py`
- Persistent checkpointing
- Resume from failures
- Checkpoint inspection
- History viewing

### `tests/`

Unit tests for the agent.

#### `test_agent.py`
- State creation tests
- Configuration tests
- Agent creation tests
- Routing logic tests

**Test Classes:**
- `TestStateCreation` - State initialization
- `TestConfiguration` - Config validation
- `TestAgentCreation` - Agent setup
- `TestRouting` - Routing logic

## File Purposes

### Configuration Files

- `.env.example` - Environment template (copy to `.env`)
- `.gitignore` - Files to exclude from git
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation config

### Documentation Files

- `README.md` - Main documentation
- `QUICKSTART.md` - Quick start guide
- `LICENSE` - MIT license text
- `PROJECT_STRUCTURE.md` - This file

## Data Flow

```
User Query
    ↓
create_initial_state()
    ↓
create_research_agent()
    ↓
agent.invoke(state, config)
    ↓
[Graph Execution]
    ├── plan_research
    ├── execute_search
    ├── validate_results
    ├── process_results
    ├── generate_report
    └── handle_error (if needed)
    ↓
Final State (with report)
```

## Extension Points

### Add New Nodes

```python
# In nodes.py
def my_custom_node(state: ResearchAgentState, config: AgentConfig) -> dict:
    # Your logic here
    return {
        "current_stage": "next_stage",
        # other updates
    }

# In graph.py
workflow.add_node("my_node", my_custom_node)
workflow.add_edge("previous_node", "my_node")
```

### Add New Search Tools

```python
# In tools.py
from langchain_community.tools import SerpAPIWrapper

def create_serpapi_tool(api_key: str):
    return SerpAPIWrapper(serpapi_api_key=api_key)
```

### Customize Configuration

```python
# In your code
config = AgentConfig(
    llm_model="gpt-3.5-turbo",
    llm_temperature=0.7,
    max_retries=5,
    search_limit=5
)

agent = create_research_agent()
```

## Dependencies

### Core
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools
- `langgraph` - Graph-based workflows

### Tools
- `duckduckgo-search` - Web search

### Utilities
- `python-dotenv` - Environment variables

### Development (optional)
- `pytest` - Testing
- `black` - Code formatting
- `mypy` - Type checking

## Best Practices

1. **Always use virtual environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Set environment variables via .env**
   ```bash
   cp .env.example .env
   # Edit .env
   ```

3. **Import from package root**
   ```python
   from research_agent import create_research_agent
   ```

4. **Use checkpointing in production**
   ```python
   from langgraph.checkpoint.sqlite import SqliteSaver
   checkpointer = SqliteSaver.from_conn_string("db.sqlite")
   ```

5. **Monitor with streaming**
   ```python
   for step in agent.stream(state, config):
       log_step(step)
   ```

## Development Workflow

1. Make changes to source code
2. Run tests: `pytest tests/`
3. Try examples: `python examples/basic_usage.py`
4. Format code: `black src/`
5. Type check: `mypy src/`

## Deployment Checklist

- [ ] Set production API keys
- [ ] Use persistent checkpointing (PostgreSQL/SQLite)
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Add rate limiting
- [ ] Implement cost tracking
- [ ] Set up error alerting
- [ ] Document custom configurations
- [ ] Create deployment scripts
- [ ] Set up CI/CD

---

For questions or suggestions, open an issue on GitHub.
