"""Chapter 17: the resolve agent mounted into Atlas's topology."""

from langchain_core.messages import AIMessage

from atlas.resolve import make_resolve_node
from atlas.state import AtlasState


class FakeAgent:
    """Stands in for a compiled create_agent graph: takes a messages dict,
    returns one with the reply appended."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, payload: dict, config: dict | None = None) -> dict:
        self.calls.append(payload)
        return {"messages": [*payload["messages"], AIMessage("resolved")]}


def test_resolve_node_passes_conversation_and_returns_only_the_reply() -> None:
    agent = FakeAgent()
    node = make_resolve_node(agent)
    state: AtlasState = {
        "messages": [{"role": "user", "content": "where is my order"}],
        "retrieved": [{"id": "kb:1", "text": "Orders ship in 2 days.", "score": 1.0}],
        "ticket": None,
        "route": "answer",
        "retrieve_attempts": 1,
        "error": None,
        "refund_done": False,
    }

    delta = node(state)

    assert agent.calls[0]["messages"] == state["messages"]
    assert [m.content for m in delta["messages"]] == ["resolved"]
    assert "retrieved" not in delta
