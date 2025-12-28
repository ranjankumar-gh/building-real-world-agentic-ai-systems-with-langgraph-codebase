# Quick Start Guide

Get the Research Agent running in 5 minutes.

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Step 2: Set API Key

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

Or export directly:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

## Step 3: Run Your First Research

```bash
python examples/basic_usage.py
```

That's it! The agent will:
1. Plan search queries
2. Execute web searches
3. Extract key findings
4. Generate a research report

## What Happens?

```
Creating research agent...

Researching: Latest developments in quantum computing

============================================================

=== PLAN ===
Stage: searching
Retry count: 0

=== SEARCH ===
Stage: validating
Retry count: 0

=== VALIDATE ===
Stage: processing
Retry count: 0

=== PROCESS ===
Stage: generating
Retry count: 0

=== GENERATE ===
Stage: complete
Retry count: 0

============================================================
FINAL REPORT
============================================================

Executive Summary:
Quantum computing has seen significant progress...

[Full report appears here]

============================================================

Statistics:
  - Search queries: 5
  - Valid results: 3/3
  - Key findings: 5
  - Retry count: 0
  - Final stage: complete
```

## Try Other Examples

**Streaming execution** (see progress in real-time):
```bash
python examples/streaming_example.py
```

**Checkpointing** (resume from failures):
```bash
python examples/checkpoint_example.py
```

## Customize the Query

Edit `examples/basic_usage.py`:

```python
initial_state = create_initial_state(
    research_query="YOUR QUESTION HERE",
    max_retries=2
)
```

## Next Steps

- Read the full [README.md](README.md)
- Explore the code in `src/research_agent/`
- Run tests: `pytest tests/`
- Check configuration options in `src/research_agent/config.py`

## Troubleshooting

**API Key Error?**
```
Error: OpenAI API key not found
```
Solution: Set `OPENAI_API_KEY` in `.env` file

**Import Errors?**
```
ModuleNotFoundError: No module named 'research_agent'
```
Solution: Make sure you're running from the project root and virtual environment is activated

**Search Failures?**
```
Error: Insufficient search results
```
Solution: Check internet connection or increase `max_retries`

## Support

- Issues: Open an issue on GitHub
- Documentation: See `/docs` folder (coming soon)
- Examples: Check `/examples` directory

Happy researching! 🔬
