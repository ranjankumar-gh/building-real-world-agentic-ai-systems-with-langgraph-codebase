"""Chapter 19, "Streaming" - atlas/stream.py.

See "Building the multiplexed stream". `stream_atlas` wraps a compiled graph
(by default `atlas/graph.py`'s `graph`) with `stream_mode=["updates",
"messages", "custom"]`, `subgraphs=True`, `version="v2"`, and normalizes every
chunk into `{"kind", "source", "data"}`. The first tests monkeypatch the same
seams `tests/test_graph.py` already does (`classify`/`search_kb`/
`compose_answer`) to avoid a live model call - the point under test is the
multiplexing and namespace-normalization this chapter adds.

The redaction tests drive a small graph whose mounted agent streams an email
address split across token deltas, the way a real tokenizer splits one, and
assert that no event on the stream carries the raw address. The disconnect
tests pin the two drivers: closing `stream_atlas` stops an in-process run;
closing `stream_detached` does not.
"""

import re
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Interrupt

from atlas import graph as graph_module
from atlas.stream import REDACTED, hold_back, redact, stream_atlas, stream_detached


def _decision(route: str):
    return SimpleNamespace(route=route)


def _stub_the_model_seams(monkeypatch, route: str = "answer") -> None:
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision(route))
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved, **_: "final answer"
    )


def test_stream_atlas_yields_events_shaped_as_kind_source_data(monkeypatch):
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-shape"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    assert events  # triage and answer both land as updates
    for event in events:
        assert set(event) == {"kind", "source", "data"}
        assert event["kind"] in {"updates", "messages", "custom"}


def test_stream_atlas_normalizes_the_root_namespace_to_main(monkeypatch):
    """chunk["ns"] is () for the root graph - stream_atlas maps that to
    ("main",) so a consumer never has to special-case the root graph."""
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-ns"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    assert all(event["source"] == ("main",) for event in events)


def test_stream_atlas_reports_an_updates_event_for_every_node_that_ran(monkeypatch):
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-updates"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    updated_nodes = {
        node
        for event in events
        if event["kind"] == "updates"
        for node in event["data"]
    }
    assert {"triage", "answer"} <= updated_nodes


def test_stream_atlas_carries_the_route_triage_decided_in_its_updates_payload(
    monkeypatch,
):
    _stub_the_model_seams(monkeypatch, route="escalate")
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-escalate"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    triage_updates = [
        event["data"]["triage"]
        for event in events
        if event["kind"] == "updates" and "triage" in event["data"]
    ]
    # triage's update also carries the Chapter 6 per-question guard reset
    assert [update["route"] for update in triage_updates] == ["escalate"]


# --- The resolved graph streams with its context ---------------------------


class TokenModel(BaseChatModel):
    """Streams a fixed answer one word at a time; calls no tools."""

    @property
    def _llm_type(self) -> str:
        return "token"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "TokenModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage("Your refund is on its way.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        for delta in ["Your ", "refund ", "is ", "on ", "its ", "way."]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


def _resolved_graph_with_a_token_model(monkeypatch):
    """Chapter 17's build_resolved_graph, whole middleware stack live, with
    triage forced to "answer" and the resolve agent's model scripted."""
    import atlas.agent as agent_module
    from atlas.resolve import build_resolved_graph

    _stub_the_model_seams(monkeypatch)
    monkeypatch.setattr(agent_module, "model", TokenModel())
    return build_resolved_graph()


def _model_tokens(events: list[dict]) -> tuple[str, set[tuple]]:
    tokens = [
        event
        for event in events
        if event["kind"] == "messages"
        and isinstance(event["data"][0], AIMessageChunk)
        and event["data"][1].get("langgraph_node") == "model"
    ]
    text = "".join(event["data"][0].text for event in tokens)
    return text, {event["source"] for event in tokens}


def test_stream_atlas_streams_the_resolved_graph_given_its_context(monkeypatch):
    """The chapter's production call: build_resolved_graph()'s mounted agent
    declares AtlasContext, so stream_atlas must pass context= through."""
    from atlas.security import AtlasContext

    graph = _resolved_graph_with_a_token_model(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-resolved"}}
    context = AtlasContext(role="customer", customer_id="c-1")
    inputs = {"messages": [{"role": "user", "content": "where is my refund?"}]}

    events = list(stream_atlas(inputs, config, graph=graph, context=context))

    text, sources = _model_tokens(events)
    assert text == "Your refund is on its way."
    assert sources and all(source[0].startswith("answer:") for source in sources)


def test_stream_detached_forwards_the_context_to_the_run(monkeypatch):
    from atlas.security import AtlasContext

    graph = _resolved_graph_with_a_token_model(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-detached-resolved"}}
    context = AtlasContext(role="customer", customer_id="c-1")
    inputs = {"messages": [{"role": "user", "content": "where is my refund?"}]}

    events = list(stream_detached(inputs, config, graph=graph, context=context))

    assert _model_tokens(events)[0] == "Your refund is on its way."


def test_the_resolved_graph_without_a_context_fails_at_the_first_gate(monkeypatch):
    """The control: with no context, the gates have no customer_id to read."""
    graph = _resolved_graph_with_a_token_model(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-no-context"}}
    inputs = {"messages": [{"role": "user", "content": "where is my refund?"}]}

    with pytest.raises(AttributeError, match="customer_id"):
        list(stream_atlas(inputs, config, graph=graph))


# --- Redaction on the stream ------------------------------------------------

ADDRESS = "jane.doe@example.com"
ANY_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# The address split across deltas, the way a tokenizer splits one.
DELTAS = ["Reach ", "me at jane", ".doe@exa", "mple", ".com today."]


class SplitDeltaModel(BaseChatModel):
    """Odd calls: a tool call whose args carry the address. Even calls: text
    whose address is split across token deltas."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "split-delta"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "SplitDeltaModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        chunks = [c.message for c in self._stream(messages, stop, run_manager)]
        merged = chunks[0]
        for chunk in chunks[1:]:
            merged = merged + chunk
        message = AIMessage(
            content=merged.content, tool_calls=merged.tool_calls, id=merged.id
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        self.calls += 1
        if self.calls % 2 == 1:
            call = {
                "name": "lookup",
                "args": f'{{"who": "{ADDRESS}"}}',
                "id": "call-1",
                "index": 0,
            }
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=[call])
            )
            return
        for delta in DELTAS:
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


@tool
def lookup(who: str) -> str:
    """Look a customer up."""
    get_stream_writer()({"progress": f"looking up {who}"})
    return f"found {who}"


def _graph_with_a_streaming_agent():
    agent = create_agent(SplitDeltaModel(), tools=[lookup], name="resolve-agent")

    def answer(state: MessagesState) -> dict:
        return agent.invoke(state)

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def _inputs() -> dict:
    return {"messages": [{"role": "user", "content": f"I am {ADDRESS}"}]}


def _answer_text(events: list[dict]) -> str:
    return "".join(
        event["data"][0].text
        for event in events
        if event["kind"] == "messages" and isinstance(event["data"][0], AIMessageChunk)
    )


def test_the_unredacted_stream_carries_the_address():
    """The control: graph.stream() itself redacts nothing."""
    graph = _graph_with_a_streaming_agent()
    chunks = list(
        graph.stream(
            _inputs(),
            stream_mode=["updates", "messages", "custom"],
            subgraphs=True,
            version="v2",
        )
    )
    assert any(ANY_ADDRESS.search(repr(chunk["data"])) for chunk in chunks)


def test_stream_atlas_sends_no_address_on_any_event():
    graph = _graph_with_a_streaming_agent()

    events = list(stream_atlas(_inputs(), {}, graph=graph))

    assert {"updates", "messages", "custom"} <= {event["kind"] for event in events}
    assert not [event for event in events if ANY_ADDRESS.search(repr(event))]


def test_stream_atlas_redacts_an_address_split_across_token_deltas():
    graph = _graph_with_a_streaming_agent()

    events = list(stream_atlas(_inputs(), {}, graph=graph))

    assert _answer_text(events) == f"Reach me at {REDACTED} today."


def test_stream_atlas_redacts_tool_reported_progress():
    graph = _graph_with_a_streaming_agent()

    events = list(stream_atlas(_inputs(), {}, graph=graph))

    progress = [event["data"] for event in events if event["kind"] == "custom"]
    assert progress == [{"progress": f"looking up {REDACTED}"}]


def test_stream_atlas_tags_a_mounted_agent_with_its_namespace():
    """ns names the node that started the subgraph: answer:<task_id>."""
    graph = _graph_with_a_streaming_agent()

    events = list(stream_atlas(_inputs(), {}, graph=graph))

    nested = {event["source"] for event in events if event["source"] != ("main",)}
    assert nested and all(source[0].startswith("answer:") for source in nested)


def test_hold_back_releases_text_only_up_to_the_last_whitespace():
    held: dict[str, str] = {}

    first = hold_back(held, AIMessageChunk(content="me at jane", id="m"))
    last = hold_back(
        held,
        AIMessageChunk(content=".doe@example.com", id="m", chunk_position="last"),
    )

    assert first.text == "me at "
    assert last.text == REDACTED
    assert held == {}


def test_hold_back_keeps_tool_call_chunks_and_redacts_them():
    fragment = {"name": "lookup", "args": f'{{"who": "{ADDRESS}"}}', "id": "c1",
                "index": 0}

    wire = hold_back({}, AIMessageChunk(content="", id="m", tool_call_chunks=[fragment]))

    assert [c["name"] for c in wire.tool_call_chunks] == ["lookup"]
    assert wire.tool_call_chunks[0]["args"] == f'{{"who": "{REDACTED}"}}'


def test_stream_atlas_streams_the_tool_call_without_the_address():
    graph = _graph_with_a_streaming_agent()

    events = list(stream_atlas(_inputs(), {}, graph=graph))

    streamed_calls = [
        chunk
        for event in events
        if event["kind"] == "messages" and isinstance(event["data"][0], AIMessageChunk)
        for chunk in event["data"][0].tool_call_chunks
    ]
    assert [c["name"] for c in streamed_calls] == ["lookup"]
    assert REDACTED in streamed_calls[0]["args"]


def test_redact_reaches_messages_tool_calls_and_interrupts():
    message = AIMessage(
        content=f"cc {ADDRESS}",
        tool_calls=[{"name": "lookup", "args": {"who": ADDRESS}, "id": "c1"}],
    )
    payload = {
        "answer": {"messages": [message, HumanMessage(f"I am {ADDRESS}")]},
        "__interrupt__": (Interrupt(value={"note": ADDRESS}, id="i1"),),
    }

    assert not ANY_ADDRESS.search(repr(redact(payload)))
    assert ADDRESS in repr(payload)  # the original is not mutated


# --- Disconnect: who drives the run -----------------------------------------


class Steps(TypedDict):
    ran: list[str]


def _three_steps():
    def step(name: str):
        def node(state: Steps) -> dict:
            time.sleep(0.05)
            return {"ran": state["ran"] + [name]}

        return node

    builder = StateGraph(Steps)
    for name in ("n1", "n2", "n3"):
        builder.add_node(name, step(name))
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", "n3")
    builder.add_edge("n3", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_closing_stream_atlas_stops_an_in_process_run_and_stream_none_resumes():
    graph = _three_steps()
    config = {"configurable": {"thread_id": "reader-drives"}}

    events = stream_atlas({"ran": []}, config, graph=graph)
    next(events)
    events.close()  # the client disconnects; nothing else is reading
    time.sleep(0.3)

    assert graph.get_state(config).values["ran"] == ["n1"]
    list(graph.stream(None, config))  # resume on the same thread (Chapter 9)
    assert graph.get_state(config).values["ran"] == ["n1", "n2", "n3"]


def test_closing_stream_detached_closes_the_live_channel_not_the_run():
    graph = _three_steps()
    config = {"configurable": {"thread_id": "worker-drives"}}

    events = stream_detached({"ran": []}, config, graph=graph, maxsize=1)
    next(events)
    events.close()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if graph.get_state(config).values.get("ran") == ["n1", "n2", "n3"]:
            break
        time.sleep(0.05)
    assert graph.get_state(config).values["ran"] == ["n1", "n2", "n3"]


class RunFailed(Exception):
    pass


def test_stream_detached_raises_the_runs_error_instead_of_ending_quietly():
    """A failed run must not look like a finished one to the reader - even
    with a one-slot buffer that drops the oldest event."""

    def boom(state: Steps) -> dict:
        raise RunFailed("backend down")

    builder = StateGraph(Steps)
    builder.add_node("n1", lambda state: {"ran": ["n1"]})
    builder.add_node("boom", boom)
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "boom")
    builder.add_edge("boom", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "worker-fails"}}

    events = stream_detached({"ran": []}, config, graph=graph, maxsize=1)

    with pytest.raises(RunFailed, match="backend down"):
        list(events)


def test_stream_detached_ends_cleanly_when_the_run_completes():
    graph = _three_steps()
    config = {"configurable": {"thread_id": "worker-completes"}}

    events = list(stream_detached({"ran": []}, config, graph=graph))

    assert [node for event in events for node in event["data"]] == ["n1", "n2", "n3"]
