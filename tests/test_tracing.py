"""Chapter 20, "Observability and Debugging with LangSmith" -
atlas/tracing.py.

See "Turning tracing on, by environment" and "The PII redaction ordering
bug, made concrete". None of `configure_tracing`/`configure_otel_export`/
`redact_trace_outputs`/`langsmith_client` need a live LangSmith connection
to build or exercise locally - they only set environment variables or
transform a plain dict, matching the no-live-call convention already used
throughout this repo (see `tests/test_hello.py`). This chapter is the
first to depend on a REAL external service outside that convention -
LangSmith itself - so the one test that needs an actual live connection
(submitting a run and reading it back from a real project) is skip-guarded
on `LANGSMITH_API_KEY` rather than mocked: mocking the service itself would
prove nothing about whether the real integration works."""

import os

import pytest
from langsmith import Client

from atlas.middleware import redact_email
from atlas.tracing import (
    configure_otel_export,
    configure_tracing,
    langsmith_client,
    redact_trace_outputs,
)

# Captured at import, BEFORE any test in this module runs. Several of them
# drive configure_tracing, which assigns LANGSMITH_API_KEY through
# os.environ directly rather than through monkeypatch, so by the time the
# live test runs the ambient key can be a fake one left behind by a
# neighbour. The live test builds its client from this instead.
_REAL_LANGSMITH_KEY = os.environ.get("LANGSMITH_API_KEY")

requires_langsmith = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="requires a live LangSmith connection",
)


def test_configure_tracing_sets_the_project_by_environment(monkeypatch):
    monkeypatch.setenv("ATLAS_LANGSMITH_KEY", "test-key-123")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    configure_tracing("dev")

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key-123"
    assert os.environ["LANGSMITH_PROJECT"] == "atlas-dev"


def test_configure_tracing_routes_prod_to_its_own_project(monkeypatch):
    """Dev and prod traces must never mix - the project name is derived
    from `env`, never inferred from a default."""
    monkeypatch.setenv("ATLAS_LANGSMITH_KEY", "test-key-123")

    configure_tracing("prod")

    assert os.environ["LANGSMITH_PROJECT"] == "atlas-prod"


def test_configure_tracing_fails_loudly_on_a_missing_key(monkeypatch):
    """"Assert the requirement, don't degrade quietly" - the same
    discipline Chapter 7 applied to get_tools() returning zero MCP tools.
    A missing ATLAS_LANGSMITH_KEY must raise, not silently trace nothing."""
    monkeypatch.delenv("ATLAS_LANGSMITH_KEY", raising=False)

    with pytest.raises(KeyError):
        configure_tracing("dev")


def test_configure_otel_export_sets_the_endpoint_and_exclusivity_flag(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_OTEL_ONLY", raising=False)

    configure_otel_export("http://localhost:4318")

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
    assert os.environ["LANGSMITH_OTEL_ONLY"] == "true"


def test_redact_trace_outputs_redacts_email_without_dropping_other_keys():
    outputs = {
        "messages": [{"role": "assistant", "content": "reach me at a@b.com"}],
        "usage": {"total_tokens": 42},
    }

    result = redact_trace_outputs(outputs)

    assert result["messages"] == [
        {"role": "assistant", "content": "reach me at [EMAIL]"}
    ]
    assert result["usage"] == {"total_tokens": 42}  # untouched keys pass through


def test_redact_trace_outputs_redacts_every_message_not_just_the_first():
    outputs = {
        "messages": [
            {"role": "user", "content": "cc jane@example.com"},
            {"role": "assistant", "content": "cc john@example.com too"},
        ]
    }

    result = redact_trace_outputs(outputs)

    assert [m["content"] for m in result["messages"]] == [
        "cc [EMAIL]",
        "cc [EMAIL] too",
    ]


def test_redact_trace_outputs_agrees_with_the_wire_redaction():
    """The chapter's central claim: the wire (PIIMiddleware, via
    atlas.middleware.pii's detector) and the trace (this function) can
    never disagree about what "redacted" means, because both are built
    from the one EMAIL_PATTERN in atlas/middleware.py."""
    text = "cc jane@example.com please"

    outputs = redact_trace_outputs({"messages": [{"content": text}]})

    assert outputs["messages"][0]["content"] == redact_email(text)


def test_langsmith_client_is_constructed_locally_with_hide_outputs_wired():
    """Constructing a Client with hide_outputs= is a local operation - no
    network call, no API key needed to build it."""
    assert isinstance(langsmith_client, Client)
    assert langsmith_client._hide_outputs is redact_trace_outputs


@requires_langsmith
def test_a_traced_run_is_readable_back_from_a_live_langsmith_project(monkeypatch):
    """The one thing this chapter genuinely cannot verify offline: that a
    run submitted with tracing on actually lands in, and is readable back
    from, a real LangSmith project. Skipped without LANGSMITH_API_KEY.

    The previous version of this test listed projects and asserted the
    result was a list. `list(...)` is always a list, so it passed against an
    account with no traces in it at all - it never submitted a run and never
    read one back, which is the entire claim in the sentence above.

    This version submits a traced call to a per-run project, polls until
    ingestion catches up, and asserts on the run's name and outputs.
    Ingestion is asynchronous, so the poll is the test being honest about
    the service rather than flaky: a fixed sleep would either be too short
    on a slow day or waste time on a fast one.
    """
    import time
    import uuid

    from langsmith import traceable
    from langsmith.run_helpers import tracing_context

    # The API key alone submits nothing. LANGSMITH_TRACING=true is the
    # switch, which is this chapter's own point: "Nothing about a missing
    # LANGSMITH_TRACING=true crashes Atlas. The graph runs exactly the
    # same." The first run of this test proved it the hard way, waiting the
    # full 90 seconds for a trace that was never going to be sent.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", _REAL_LANGSMITH_KEY)
    # configure_otel_export sets LANGSMITH_OTEL_ONLY=true, which routes
    # traces EXCLUSIVELY through the OTel exporter - nothing reaches the
    # LangSmith API, so this test's poll would find nothing and report it as
    # "no run arrived". Clear it explicitly rather than depending on which
    # neighbour ran first and whether it cleaned up after itself.
    monkeypatch.delenv("LANGSMITH_OTEL_ONLY", raising=False)

    client = Client(api_key=_REAL_LANGSMITH_KEY)
    project = f"atlas-tracing-test-{uuid.uuid4().hex[:12]}"
    payload = uuid.uuid4().hex[:8]

    @traceable(name="atlas_traced_probe")
    def traced_probe(value: str) -> dict:
        return {"echoed": value}

    # Pass the client explicitly rather than letting @traceable pick up an
    # ambient one. Earlier tests in this module drive configure_tracing,
    # which assigns LANGSMITH_API_KEY through os.environ directly, and the
    # tracing background client caches whatever key it first saw. Running
    # this test after them without an explicit client submits with a stale
    # fake key and the ingest endpoint answers 403 Forbidden - which the
    # poll above can only report as "no run arrived". It passes alone and
    # fails in the full suite, which is the signature of exactly that.
    # LANGSMITH_TRACING=true in the environment is NOT sufficient here.
    # tracing_is_enabled() consults a context variable first, and something
    # earlier in a full-suite run leaves it disabled - measured, not
    # assumed: at this point the env said "true" while tracing_is_enabled()
    # returned False, which is why this test passed alone and failed in the
    # suite. tracing_context(enabled=True) sets the thing that actually
    # decides.
    with tracing_context(enabled=True):
        result = traced_probe(
            payload, langsmith_extra={"project_name": project, "client": client}
        )
    assert result == {"echoed": payload}  # the wrapper must not change behaviour

    # Push anything still buffered in the background sender before polling.
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()

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
