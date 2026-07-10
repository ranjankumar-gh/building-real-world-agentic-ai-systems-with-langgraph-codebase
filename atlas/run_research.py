"""Chapter 18, "Deep Agents: The Production Harness" - "Running it".

A distinct module from `atlas/run.py`'s `run_research` (Chapter 17's
`Send`-based map-reduce subgraph) - same idea, different arm: this one
drives the Deep Agent built in `atlas/deep_research.py`. `config` carries
both `thread_id` (the checkpointer's conversation identity, Chapter 9) and
`customer_id` (read back out by `research_namespace` via `get_config()` -
see that module's docstring for why it is not passed as a plain argument).

Wrapped in a small function, parameterized the same way every other
`atlas/run.py` helper is, rather than left as the chapter's bare top-level
`invoke(...)` illustration - so it is something a test (or a caller) can
actually call."""

from atlas.deep_research import deep_research_agent


def run_deep_research(thread_id: str, customer_id: str, request: str) -> dict:
    return deep_research_agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"configurable": {"thread_id": thread_id, "customer_id": customer_id}},
    )
