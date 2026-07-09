"""Chapter 7, "Tools, Models, MCP, and create_agent" - structured triage.

See "Structured output: the contract pointed inward". This module replaces
the `classify` stub in `atlas.helpers` (a `Literal` string, parsed from
prose) with a validated `TriageResult` produced via `create_agent`'s
`response_format`. Triage acts on nothing - it only decides - so
`triage_agent` carries no tools; that keeps it clear of the Anthropic
`response_format`-plus-tools sharp edge described in "Which strategy, and
why it matters", and is why triage and the tool-using `resolve_agent`
(`atlas/agent.py`) are separate agents.

`atlas/graph.py`'s `triage` node imports `classify` from here now.
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    """The triage decision for an incoming support message."""

    route: Literal["answer", "retrieve", "escalate"] = Field(
        description="Where the conversation should go next."
    )
    reason: str = Field(description="One short sentence justifying the route.")


TRIAGE_PROMPT = (
    "You are the triage step for Atlas. Decide where the conversation goes "
    "next. Choose 'retrieve' for factual questions, 'answer' for simple "
    "replies, and 'escalate' when a human is needed."
)

triage_agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[],  # triage decides; it does not act
    response_format=ProviderStrategy(TriageResult),
    system_prompt=TRIAGE_PROMPT,
)


def classify(messages) -> TriageResult:
    result = triage_agent.invoke({"messages": messages})
    return result["structured_response"]  # a validated TriageResult
