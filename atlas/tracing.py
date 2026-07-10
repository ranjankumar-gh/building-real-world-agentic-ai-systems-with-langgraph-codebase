"""Chapter 20, "Observability and Debugging with LangSmith" - Atlas's
LangSmith tracing wiring.

See "Turning tracing on, by environment" and "The PII redaction ordering
bug, made concrete". `configure_tracing` and `configure_otel_export` only
set environment variables - LangChain/LangGraph runnables self-instrument
once `LANGSMITH_TRACING` is set, no code-level import needed for the parts
of tracing this module doesn't touch directly. `@traceable` (see
`atlas/effects.py`'s `idempotency_key`) and `name=`/`trace()` (see
`atlas/agent.py`, `atlas/research.py`, `atlas/deep_research.py`) are the
other two pieces of this chapter's instrumentation - this module is just
the environment wiring plus the trace-side PII redaction.

Building and exercising this module needs no live LangSmith connection:
constructing a `Client`, setting environment variables, and transforming a
plain dict are all local operations that only reach the network once
`LANGSMITH_TRACING` is actually `"true"` and a real run happens. See
`tests/test_tracing.py` for the one test that DOES need a live account
(skip-guarded on `LANGSMITH_API_KEY`, per this book's companion-repo
convention for external services - see the module docstring in
`tests/test_tracing.py`).

`redact_trace_outputs` closes the gap "The PII redaction ordering bug, made
concrete" identifies: `PIIMiddleware`'s `apply_to_output` (`atlas/
middleware.py`) redacts what a `stream()`/`stream_events()` consumer sees;
LangSmith's tracing client ingests the run through a separate path
`apply_to_output` never reaches. `hide_outputs` is LangSmith's own
documented mechanism for this - `Client(hide_outputs=fn)` transforms a
run's outputs before they leave the process. `redact_email` is imported
from `atlas.middleware` rather than redefined here, and both it and
`PIIMiddleware`'s `detector=` are built from the one `EMAIL_PATTERN`, so the
wire and the trace can never disagree about what "redacted" means.
"""

import os

from langsmith import Client

from atlas.middleware import redact_email


def configure_tracing(env: str) -> None:
    """Turn on LangSmith tracing, routed to a project by environment.

    Called once at process startup. `env` is 'dev' or 'prod' - never inferred
    from a default, because a wrong guess here means either no prod traces or
    dev noise landing in the project support engineers actually watch.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = os.environ["ATLAS_LANGSMITH_KEY"]  # <1>
    os.environ["LANGSMITH_PROJECT"] = f"atlas-{env}"


# 1. `LANGSMITH_API_KEY` is read from `ATLAS_LANGSMITH_KEY` rather than
#    assumed to already be set, so `configure_tracing` fails loudly (a
#    `KeyError`) on a missing key instead of silently tracing nothing - the
#    same "assert the requirement, don't degrade quietly" discipline Chapter
#    7 applied to `get_tools()` returning zero MCP tools.


def configure_otel_export(endpoint: str) -> None:
    """Route Atlas's traces through OpenTelemetry instead of, or alongside,
    LangSmith's own ingestion API. Requires `pip install "langsmith[otel]"`.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_OTEL_ONLY"] = "true"  # <1>


# 1. `LANGSMITH_OTEL_ONLY=true` sends traces *exclusively* through the OTel
#    exporter - without it, traces go to both LangSmith's API and the OTel
#    endpoint, which is occasionally what you want (a migration period) and
#    often just double ingestion cost.


def redact_trace_outputs(outputs: dict) -> dict:
    """LangSmith's documented hook for this exact problem: transform what a
    run records before it leaves the process, using Atlas's own detector."""
    messages = outputs.get("messages", [])
    redacted = [{**m, "content": redact_email(m["content"])} for m in messages]
    return {**outputs, "messages": redacted}


langsmith_client = Client(hide_outputs=redact_trace_outputs)  # <1>

# 1. `hide_outputs` is LangSmith's own documented mechanism for redacting
#    trace content before it is sent - not a workaround improvised for this
#    chapter. Passing a function instead of `True` means selective
#    redaction rather than blacking out every run's output.
