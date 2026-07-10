"""Chapter 18, "Deep Agents: The Production Harness" - atlas/run_research.py.

See "Running it". `run_deep_research` drives the compiled
`deep_research_agent` end to end - the main agent planning with
`write_todos`, delegating to `source_researcher` via `task`, and composing a
report from `findings/` - which needs a live Anthropic model call the same
way `atlas/run.py`'s live-agent paths do. Skip-guarded behind
`ANTHROPIC_API_KEY`, matching the `requires_postgres`/`requires_openai`
pattern already used in `tests/test_memory.py` and `tests/test_run.py` for
the external-service exception - not mocked, since deepagents' own agent
loop is not this repo's to fake."""

import os

import pytest

from atlas.run_research import run_deep_research

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a live Anthropic model call through create_deep_agent",
)


def test_run_deep_research_is_importable_and_callable_without_a_live_key():
    """Importing/wiring the module (Chapter 7's "building does not require a
    live key" convention) never touches the network - only calling
    `run_deep_research` does."""
    assert callable(run_deep_research)


@requires_anthropic
def test_run_deep_research_returns_a_final_message():
    """Skipped by default - see `requires_anthropic` above."""
    result = run_deep_research(
        thread_id="research-9001",
        customer_id="cust-42",
        request="Report on Q3 churn drivers using docs.internal/sla.",
    )

    assert result["messages"]
