"""Chapter 20, "Observability and Debugging with LangSmith" - Atlas's
LangSmith tracing wiring.

See "Turning tracing on, by environment", "Naming the fleet", and
"Redacting the trace, not just the wire". Environment variables switch
tracing on and pick the project; LangChain/LangGraph runnables
self-instrument once `LANGSMITH_TRACING` is set. The client that records
the runs is installed in code, because that is where the trace-side PII
redaction lives.

`mask_email` is LangSmith's own anonymizer, built from `atlas/middleware.py`'s
`EMAIL_PATTERN` with PIIMiddleware's own replacement token, so the stored
messages, the stream (`atlas/stream.py`), and the trace all write the same
`[REDACTED_EMAIL]`. An anonymizer walks every string in a run's serialized
inputs AND outputs; `Client(hide_outputs=fn)` sees only outputs, and a
client you construct but never install is never used - tracing falls back
to a process-wide cached client unless `ls.configure(client=...)` sets one.

`trace_config` puts the run name, tags, and metadata on the config at the
entry point, once, so every span the run produces inherits them.

Building and exercising this module needs no live LangSmith account. A
`Client` does reach for its API in a background thread once constructed,
so `tests/test_tracing.py` points it at a refused local port or a capturing
session; no test in this repo calls LangSmith unless `LANGSMITH_API_KEY` is
set (the one live test, skip-guarded).
"""

import os
from typing import Literal

import langsmith as ls
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langsmith.anonymizer import create_anonymizer
from langsmith.client import TracingMode

from atlas.middleware import EMAIL_PATTERN

mask_email = create_anonymizer(
    [{"pattern": EMAIL_PATTERN, "replace": "[REDACTED_EMAIL]"}]  # <1>
)


def install_trace_client(mode: TracingMode | None = None) -> Client:
    """Build the client every traced run uses, and install it process-wide."""
    client = Client(anonymizer=mask_email, tracing_mode=mode)
    ls.configure(client=client)  # <2>
    return client


# 1. One pattern, one token: `pii` (atlas/middleware.py) is built from the
#    same EMAIL_PATTERN and writes `[REDACTED_EMAIL]`, so a support engineer
#    sees one token in the state, on the stream, and in the trace.
# 2. `Client(...)` alone changes nothing: LangChain's tracer uses the
#    process-wide client, and without `ls.configure` that is a default one
#    it builds itself.


def configure_tracing(env: str) -> Client:
    """Turn on LangSmith tracing, routed to a project by environment.

    Called once at process startup, before anything traces: the LangSmith
    SDK reads its environment variables once per process and caches them.
    `env` is 'dev' or 'prod' - never inferred from a default, because a wrong
    guess here means either no prod traces or dev noise landing in the
    project support engineers actually watch.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = os.environ["ATLAS_LANGSMITH_KEY"]  # <1>
    os.environ["LANGSMITH_PROJECT"] = f"atlas-{env}"
    return install_trace_client()


# 1. `LANGSMITH_API_KEY` is read from `ATLAS_LANGSMITH_KEY` rather than
#    assumed to already be set, so `configure_tracing` fails loudly (a
#    `KeyError`) on a missing key instead of silently tracing nothing - the
#    same "assert the requirement, don't degrade quietly" discipline Chapter
#    7 applied to `get_tools()` returning zero MCP tools.


def trace_config(
    kind: Literal["support", "research"], thread_id: str, customer_id: str
) -> RunnableConfig:
    """Name and tag a run once, at the entry point every run goes through."""
    return {
        "configurable": {"thread_id": thread_id, "customer_id": customer_id},
        "run_name": f"atlas-{kind}",
        "tags": ["atlas", kind],
        "metadata": {"thread_id": thread_id, "customer_id": customer_id},
    }


def configure_otel_export(endpoint: str) -> Client:
    """Route Atlas's traces through OpenTelemetry instead of LangSmith's own
    ingestion API. Requires `pip install "langsmith[otel]"`.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    return install_trace_client(mode="otel")  # <1>


# 1. `otel` sends traces only through the OTel exporter; `hybrid` sends them
#    to both LangSmith and the OTLP endpoint (a migration period, at double
#    ingestion cost); the default, `langsmith`, ignores the OTLP endpoint.
#    A deployment can set the same switch as LANGSMITH_TRACING_MODE, but only
#    before the process builds its first client - the SDK caches it. The
#    argument wins over the variable. Without the `otel` extra installed,
#    the client warns and falls back to `langsmith`.
