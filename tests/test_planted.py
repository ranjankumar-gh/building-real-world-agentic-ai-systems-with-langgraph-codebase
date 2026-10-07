"""Chapter 20, "Observability and Debugging with LangSmith" -
atlas/planted.py.

See "Reproducing the regression from the trace alone". The planted
regression, run end to end and found in the trace: a scripted coordinator
built on `REGRESSED_PROMPT` delegates to `web_research`, which reports the
current policy document (DOC-207), then delegates to `doc_research` with the
stale ID (DOC-114) it copied from an earlier turn still in its history.

The run is traced through a LangSmith `Client` whose session captures each
request instead of sending it (the tests/test_tracing.py pattern), and the
captured multipart bodies are read back into spans. The assertions are the
chapter's three clicks: find the run by its tags and metadata, open the
`doc-research` span and read its input, and compare it with the `supervisor`
span (a sibling of `doc_research` under the same root) that holds both IDs.
No live model, no network: each specialist's scoped agent is a real
`create_agent` around a fake chat model, so its `doc-research` span exists.
"""

import inspect
import json
import re
from typing import Any

import langsmith as ls
import pytest
import requests
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import Client

from atlas import research as research_module
from atlas.planted import REGRESSED_PROMPT, regressed_supervisor_graph
from atlas.tracing import trace_config

REFUSED = "http://127.0.0.1:9"  # nothing listens here; a stray send fails locally
STALE, FRESH = "DOC-114", "DOC-207"

FINDINGS = {
    "web-research": f"The current return policy is {FRESH}.",
    "doc-research": "Returns are accepted within 60 days.",
}


class _CaptureSession(requests.Session):
    """Answers every request locally and keeps its body."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[bytes] = []

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        body = kwargs.get("data") or b""
        if hasattr(body, "to_string"):
            body = body.to_string()
        if isinstance(body, str):
            body = body.encode()
        if isinstance(body, bytes):
            self.bodies.append(body)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = url
        return response


class _ScriptedCoordinator(BaseChatModel):
    """Replays a script: the model whose choice the trace has to explain."""

    script: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedCoordinator":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _delegate(specialist: str, task: str, call_id: str) -> dict:
    return {
        "name": f"delegate_to_{specialist}",
        "args": {"task": task},
        "id": call_id,
        "type": "tool_call",
    }


def _spans(bodies: list[bytes]) -> list[dict]:
    """Reassemble runs from the multipart parts the client sent: each run
    arrives as `post.<id>` plus `post.<id>.inputs` / `.outputs` parts."""
    runs: dict[str, dict] = {}
    for body in bodies:
        boundary = body.split(b"\r\n", 1)[0]
        for part in body.split(boundary)[1:]:
            head, _, data = part.strip(b"\r\n").partition(b"\r\n\r\n")
            named = re.search(rb'name="\w+\.([\w-]+)(?:\.(\w+))?"', head)
            if not named:
                continue
            run = runs.setdefault(named.group(1).decode(), {})
            field = named.group(2)
            if field:
                run[field.decode()] = json.loads(data)
            else:
                run.update(json.loads(data))
    return list(runs.values())


@pytest.fixture
def planted_run(monkeypatch) -> tuple[dict, list[dict]]:
    monkeypatch.setenv("LANGSMITH_ENDPOINT", REFUSED)
    def specialist(**kw: Any) -> Any:
        reply = AIMessage(FINDINGS[kw["name"]])
        model = GenericFakeChatModel(messages=iter([reply]))
        return create_agent(model=model, tools=[], name=kw["name"])

    monkeypatch.setattr(research_module, "create_agent", specialist)
    coordinator = _ScriptedCoordinator(script=[
        AIMessage("", tool_calls=[_delegate(
            "web_research", "Find the document holding the current return policy.",
            "call_1",
        )]),
        AIMessage("", tool_calls=[_delegate(  # the regression: a stale ID
            "doc_research", f"Summarize the return window in document {STALE}.",
            "call_2",
        )]),
        AIMessage(f"Per {FRESH}, returns are accepted within 60 days."),
    ])
    session = _CaptureSession()
    client = Client(
        api_url=REFUSED, api_key="fake-key", session=session,
        info={"version": "0.10.0"},
    )
    ls.configure(client=client)
    history = [
        HumanMessage("What was the return window last year?"),
        AIMessage(f"Per {STALE}, 60 days."),
        HumanMessage("And what is it now?"),
    ]
    try:
        with ls.tracing_context(enabled=True):
            out = regressed_supervisor_graph(coordinator).invoke(
                {"messages": history, "handoffs": 0},
                trace_config("research", "thread-9", "cust-42"),
            )
        client.flush()
    finally:
        ls.configure(client=None)
    return out, _spans(session.bodies)


def test_the_regression_is_one_sentence_the_shipped_prompt_does_not_carry():
    sentence = "Always include the source document ID in the task."

    assert REGRESSED_PROMPT.endswith(f"Do not research yourself. {sentence}")
    assert "source document ID" not in inspect.getsource(research_module)


def test_the_planted_run_ships_a_brief_that_cites_the_fresh_document(planted_run):
    out, _ = planted_run

    assert out["messages"][-1].content == (
        f"Per {FRESH}, returns are accepted within 60 days."
    )


def test_click_one_the_root_carries_the_tags_and_thread_metadata(planted_run):
    _, spans = planted_run
    root = next(s for s in spans if s.get("parent_run_id") is None)

    assert root["name"] == "atlas-research"
    assert "research" in root["tags"]
    assert root["extra"]["metadata"]["thread_id"] == "thread-9"


def test_click_two_the_doc_research_span_input_carries_the_stale_id(planted_run):
    _, spans = planted_run
    doc_span = next(s for s in spans if s["name"] == "doc-research")
    received = json.dumps(doc_span["inputs"])

    assert STALE in received
    assert FRESH not in received


def test_click_three_a_sibling_supervisor_span_holds_both_ids(planted_run):
    _, spans = planted_run
    by_id = {s["id"]: s for s in spans}
    root = next(s for s in spans if s.get("parent_run_id") is None)
    doc_node = next(s for s in spans if s["name"] == "doc_research")
    supervisors = [
        s for s in spans
        if s["name"] == "supervisor" and s.get("parent_run_id") == root["id"]
    ]

    assert by_id[doc_node["parent_run_id"]] is root  # siblings, not parent/child
    assert any(
        STALE in json.dumps(s["inputs"]) and FRESH in json.dumps(s["inputs"])
        for s in supervisors
    )
