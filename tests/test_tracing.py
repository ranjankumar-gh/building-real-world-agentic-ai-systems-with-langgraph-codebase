"""Chapter 20, "Observability and Debugging with LangSmith" -
atlas/tracing.py.

See "Turning tracing on, by environment", "Naming the fleet", and
"Redacting the trace, not just the wire". Nothing here talks to LangSmith.
A `Client` starts a background thread that reaches for its API, so every
test that builds one points `LANGSMITH_ENDPOINT` at a refused local port,
or hands the client a session that captures each request instead of
sending it. The capture is the network boundary: whatever is in a captured
body is exactly what would have left the process.

The one test that needs a real account (submitting a run and reading it
back from a real project) is skip-guarded on `LANGSMITH_API_KEY` rather than
mocked: mocking the service itself would prove nothing about whether the
real integration works."""

import importlib.util
import os
import re

import langsmith as ls
import langsmith.run_trees
import pytest
import requests
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith import Client

from atlas.middleware import pii
from atlas.tracing import (
    configure_otel_export,
    configure_tracing,
    install_trace_client,
    mask_email,
    trace_config,
)

# Captured at import, BEFORE any test in this module runs, so the live test
# never picks up a fake key a neighbour left behind.
_REAL_LANGSMITH_KEY = os.environ.get("LANGSMITH_API_KEY")

requires_langsmith = pytest.mark.skipif(
    not _REAL_LANGSMITH_KEY,
    reason="requires a live LangSmith connection",
)

ADDRESS = "jane.doe@example.com"
ANY_ADDRESS = re.compile(rb"[\w.+-]+@[\w-]+\.[\w.-]+")
REFUSED = "http://127.0.0.1:9"  # nothing listens here; a stray send fails locally


@pytest.fixture
def offline_tracing_env(monkeypatch):
    """Record every variable the module writes, so monkeypatch restores it
    even though `configure_*` assign through os.environ directly; and put
    the global client back afterwards."""
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.setenv(name, "unset-by-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", REFUSED)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "unset-by-test")
    for name in (
        "LANGSMITH_TRACING_MODE",
        "LANGSMITH_OTEL_ENABLED",
        "LANGSMITH_OTEL_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch
    ls.configure(client=None)


def test_configure_tracing_sets_the_project_by_environment(offline_tracing_env):
    offline_tracing_env.setenv("ATLAS_LANGSMITH_KEY", "test-key-123")

    configure_tracing("dev")

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key-123"
    assert os.environ["LANGSMITH_PROJECT"] == "atlas-dev"


def test_configure_tracing_routes_prod_to_its_own_project(offline_tracing_env):
    """Dev and prod traces must never mix - the project name is derived
    from `env`, never inferred from a default."""
    offline_tracing_env.setenv("ATLAS_LANGSMITH_KEY", "test-key-123")

    configure_tracing("prod")

    assert os.environ["LANGSMITH_PROJECT"] == "atlas-prod"


def test_configure_tracing_fails_loudly_on_a_missing_key(offline_tracing_env):
    """A missing ATLAS_LANGSMITH_KEY must raise, not silently trace nothing."""
    offline_tracing_env.delenv("ATLAS_LANGSMITH_KEY", raising=False)

    with pytest.raises(KeyError):
        configure_tracing("dev")


def test_configure_tracing_installs_the_masking_client(offline_tracing_env):
    """The client tracing uses is the one installed with ls.configure, and
    it carries the anonymizer. A constructed-but-uninstalled client is the
    bug this replaces."""
    offline_tracing_env.setenv("ATLAS_LANGSMITH_KEY", "test-key-123")

    client = configure_tracing("dev")

    assert langsmith.run_trees.get_cached_client() is client
    assert client._anonymizer is mask_email


def test_configure_otel_export_sets_the_endpoint_and_installs_the_client(
    offline_tracing_env,
):
    client = configure_otel_export("http://localhost:4318")

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
    assert langsmith.run_trees.get_cached_client() is client
    assert client._anonymizer is mask_email  # OTel export is masked too


@pytest.mark.skipif(
    importlib.util.find_spec("opentelemetry") is not None,
    reason="pins the behavior WITHOUT the langsmith[otel] extra",
)
def test_otel_mode_without_the_extra_falls_back_to_langsmith(offline_tracing_env):
    """The pinned behavior the chapter states: without `langsmith[otel]`,
    `otel` mode warns and traces to LangSmith only. The mode is passed as
    an argument because the SDK caches LANGSMITH_TRACING_MODE once read."""
    with pytest.warns(UserWarning, match="requires OpenTelemetry"):
        client = configure_otel_export("http://localhost:4318")

    assert client._tracing_mode == "langsmith"


def test_mask_email_masks_every_string_with_piimiddlewares_token():
    payload = {"messages": [{"content": f"cc {ADDRESS}"}], "note": ADDRESS, "n": 3}

    masked = mask_email(payload)

    assert masked == {
        "messages": [{"content": "cc [REDACTED_EMAIL]"}],
        "note": "[REDACTED_EMAIL]",
        "n": 3,
    }


# --- What a traced run actually sends ---------------------------------------


class CaptureSession(requests.Session):
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
        if not isinstance(body, bytes):
            body = repr(body).encode()
        self.bodies.append(body)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = url
        return response


def _traced_turn(anonymizer) -> bytes:
    """One support turn through a mounted agent with Atlas's `pii`
    middleware, traced through a capturing client. Returns every byte the
    client would have sent."""
    session = CaptureSession()
    client = Client(
        api_url=REFUSED,
        api_key="fake-key",
        session=session,
        info={"version": "0.10.0"},
        anonymizer=anonymizer,
    )
    ls.configure(client=client)
    model = GenericFakeChatModel(messages=iter([AIMessage(f"I will email {ADDRESS}")]))
    agent = create_agent(model, tools=[], middleware=[pii], name="resolve-agent")

    def answer(state: MessagesState) -> dict:
        return agent.invoke(state)

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()

    with ls.tracing_context(enabled=True):
        graph.invoke(
            {"messages": [{"role": "user", "content": f"I am {ADDRESS}"}]},
            trace_config("support", "thread-1", "cust-42"),
        )
    client.flush()
    return b"\n".join(session.bodies)


def test_without_the_anonymizer_the_trace_carries_the_address(offline_tracing_env):
    """The control: PIIMiddleware redacts the stored messages, but the run's
    inputs and the model span's output reach the trace raw."""
    sent = _traced_turn(anonymizer=None)

    assert ANY_ADDRESS.search(sent)


def test_the_installed_anonymizer_sends_no_address(offline_tracing_env):
    sent = _traced_turn(anonymizer=mask_email)

    assert b"[REDACTED_EMAIL]" in sent  # the capture saw the run
    assert not ANY_ADDRESS.search(sent)


def test_trace_config_names_tags_and_attributes_the_run(offline_tracing_env):
    sent = _traced_turn(anonymizer=mask_email)

    for expected in (b'"atlas-support"', b'"atlas"', b'"support"', b'"cust-42"'):
        assert expected in sent


def test_trace_config_carries_customer_id_where_research_reads_it():
    """Chapter 18's research_namespace reads configurable["customer_id"];
    the same value is the trace's metadata."""
    config = trace_config("research", "thread-7", "cust-42")

    assert config["run_name"] == "atlas-research"
    assert config["tags"] == ["atlas", "research"]
    assert config["configurable"] == {"thread_id": "thread-7", "customer_id": "cust-42"}
    assert config["metadata"] == {"thread_id": "thread-7", "customer_id": "cust-42"}


def test_install_trace_client_returns_the_installed_client(offline_tracing_env):
    client = install_trace_client()

    assert langsmith.run_trees.get_cached_client() is client


@requires_langsmith
def test_a_traced_run_is_readable_back_from_a_live_langsmith_project(monkeypatch):
    """The one thing this chapter genuinely cannot verify offline: that a
    run submitted with tracing on actually lands in, and is readable back
    from, a real LangSmith project. Skipped without LANGSMITH_API_KEY.

    It submits a traced call to a per-run project, polls until ingestion
    catches up, and asserts on the run's name and outputs. Ingestion is
    asynchronous, so the poll is the test being honest about the service.
    """
    import time
    import uuid

    from langsmith import traceable
    from langsmith.run_helpers import tracing_context

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", _REAL_LANGSMITH_KEY)
    # An `otel` mode would route traces away from the LangSmith API, so the
    # poll would find nothing. Clear every mode switch explicitly.
    for name in ("LANGSMITH_TRACING_MODE", "LANGSMITH_OTEL_ONLY", "LANGSMITH_OTEL_ENABLED"):
        monkeypatch.delenv(name, raising=False)

    client = Client(api_key=_REAL_LANGSMITH_KEY)
    project = f"atlas-tracing-test-{uuid.uuid4().hex[:12]}"
    payload = uuid.uuid4().hex[:8]

    @traceable(name="atlas_traced_probe")
    def traced_probe(value: str) -> dict:
        return {"echoed": value}

    # Pass the client explicitly, and enable tracing through the context
    # variable that actually decides: a full-suite run can leave it disabled
    # while the environment says "true".
    with tracing_context(enabled=True):
        result = traced_probe(
            payload, langsmith_extra={"project_name": project, "client": client}
        )
    assert result == {"echoed": payload}  # the wrapper must not change behaviour

    client.flush()

    runs = []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            runs = list(client.list_runs(project_name=project))
        except Exception:
            runs = []  # project not visible yet; ingestion still catching up
        if runs:
            break
        time.sleep(3)

    try:
        assert runs, f"no run reached project {project} within 90s"

        run = runs[0]
        assert run.name == "atlas_traced_probe"
        assert run.outputs == {"echoed": payload}
        assert run.inputs.get("value") == payload
    finally:
        try:
            client.delete_project(project_name=project)
        except Exception:
            pass  # best effort; a stray empty test project is not a failure
