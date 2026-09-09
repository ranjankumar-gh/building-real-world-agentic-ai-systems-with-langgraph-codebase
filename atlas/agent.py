"""Chapter 7, "Tools, Models, MCP, and create_agent" - the tool-using agent.
Chapter 8, "The Middleware System", adds the middleware stack.

See "Building Atlas's real tools" (binding the model) and "Reaching
external tools with MCP". `resolve_agent` is a standalone tool-calling
`create_agent`, built from the real tools in `atlas/tools.py` - it is not
yet wired into `atlas/graph.py`'s `retrieve`/`answer` nodes (that
integration is deferred to the chapters on sub-agents inside the larger
graph).

"Binding the model" shows two ways to pass the model to `create_agent`:
a bare string id (the common case; `create_agent` resolves it through
`init_chat_model`) and a configured `init_chat_model` instance (needed once
you set parameters like temperature or a token cap).
`build_resolve_agent_from_model_id` below is the string-shorthand builder;
`resolve_agent`, the module's live default, uses the configured instance.
Both are kept so neither binding style silently overwrites the other - the
agent code is otherwise identical either way. `build_resolve_agent` (async,
further down) is the chapter's own name for the MCP-connected builder from
"Reaching external tools with MCP" - kept exactly, since Exercise 2 refers
to it by that name.

Chapter 8, "Composing the stack", attaches `atlas/middleware.py`'s stack to
`resolve_agent` via the `middleware=` argument - PII redaction, history
summarization, the custom `AuthorityGate`, and the `HumanInTheLoopMiddleware`
placeholder, in that order (list order is nesting order; first = outermost).
The chapter's own code sample shows `model="claude-sonnet-4-6"` for brevity,
but `resolve_agent` keeps the Chapter 7 configured `init_chat_model` instance
below (`temperature=0, max_tokens=1024`) rather than silently dropping it -
the middleware argument is the only thing this chapter's increment changes
about `resolve_agent`.

Chapter 12, "Context Engineering", adds `ContextBudget` (see
"Compose with summarization"). The chapter's own snippet builds an
illustrative `resolve_agent` from just two middleware -
`ContextBudget(Budget(history=4000, retrieved=2000))` and a
`SummarizationMiddleware` configured identically to `atlas/middleware.py`'s
existing `summarizer` - to show the two composing in isolation. Folded into
the REAL, cumulative `resolve_agent` below, that reuses the existing
`summarizer` instance rather than constructing a second
`SummarizationMiddleware`: `create_agent` identifies middleware by class
alone for any type without a further discriminator (the same rule Chapter
8's `pii_type` collision illustrated for `PIIMiddleware`), so two separate
`SummarizationMiddleware` instances - even with identical config - collide
with `AssertionError: Please remove duplicate middleware instances.`
`ContextBudget` is the first `wrap_model_call` hook in the stack; it wraps
the model invocation itself, which always runs after every `before_model`
hook (`pii`'s and `summarizer`'s) regardless of its list position - so
summarization still compresses the durable history before `ContextBudget`
trims the per-call view of it, satisfying "Budget and summarization must
compose in the right order" structurally, not by list ordering.

Chapter 23, "Security, Privacy, Cost, and Governance", folds in the four
middleware from `atlas/security.py`, `atlas/cost.py`, and `atlas/audit.py`:
`RoleAuthorityGate`, `InjectionGuard`, `TenantBudgetGuard`, and `AuditGate`.
Before this chapter's increment they existed but were never attached to
`resolve_agent` - the prose argued for them, the stack did not carry them.
`RoleAuthorityGate` and `AuditGate` both read `request.runtime.context`, so
`resolve_agent` now also passes `context_schema=AtlasContext`. `context_schema`
only tells `create_agent` the shape to expect - it supplies no default: a
caller that omits `context=` at invocation time gets `runtime.context is
None`, and the first tool call raises `AttributeError: 'NoneType' object
has no attribute 'role'` from inside `RoleAuthorityGate`, not a quiet
fallback role. Verified directly against this build, not assumed. See
`RESOLVE_MIDDLEWARE`'s in-line annotation below for the two ordering
constraints this chapter adds.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.store.memory import InMemoryStore
from langsmith import trace

from atlas.audit import AuditGate
from atlas.context import Budget, ContextBudget
from atlas.cost import TenantBudgetGuard
from atlas.middleware import AuthorityGate, approval, pii, summarizer
from atlas.security import AtlasContext, InjectionGuard, RoleAuthorityGate
from atlas.tools import lookup_ticket, search_kb, set_ticket_status

RESOLVE_TOOLS = [search_kb, lookup_ticket, set_ticket_status]

RESOLVE_PROMPT = (
    "You are Atlas, a customer-support assistant. Use search_kb for factual "
    "questions and the ticket tools to read or change ticket state. Never "
    "claim an action succeeded unless a tool result confirmed it."
)


def build_resolve_agent_from_model_id(model_id: str = "claude-sonnet-4-6"):
    """The string-shorthand binding: create_agent resolves `model_id`
    through init_chat_model. See "Binding the model"."""
    return create_agent(
        model=model_id, tools=RESOLVE_TOOLS, system_prompt=RESOLVE_PROMPT
    )


# The configured-instance binding, used when model parameters matter
# (temperature, token caps, timeouts). This is the module's live default.
model = init_chat_model("claude-sonnet-4-6", temperature=0, max_tokens=1024)

# Chapter 12: the explicit per-turn context budget. See "Enforcing the
# budget" - history=4000, retrieved=2000 matches the chapter's own numbers.
context_budget = ContextBudget(Budget(history=4000, retrieved=2000))

# Chapter 23: the durable BaseStore the cumulative cost cap and the audit
# log both write to - one instance, shared, so a budget check and an audit
# entry for the same call land in the same store. See "A hard, cumulative
# cost ceiling" and "A durable audit log, deliberately separate from the
# trace".
store = InMemoryStore()

RESOLVE_MIDDLEWARE = [
    pii,
    InjectionGuard(),
    context_budget,
    summarizer,
    RoleAuthorityGate(),
    AuthorityGate(),
    TenantBudgetGuard(store),
    AuditGate(store),
    approval,
]  # <1>

resolve_agent = create_agent(
    model=model,
    tools=RESOLVE_TOOLS,
    system_prompt=RESOLVE_PROMPT,
    context_schema=AtlasContext,  # <2>
    middleware=RESOLVE_MIDDLEWARE,
    name="resolve-agent",  # Chapter 20: turns a generic AgentExecutor span
    # into "resolve-agent" in the trace tree - see "Naming the fleet".
)

# 1. Order is the whole point: the first entry is the outermost wrapper, so
#    PII redaction runs before summarization and before the model ever sees
#    the raw text. See "Composing the stack". `context_budget` sits between
#    `pii` and `summarizer`, mirroring the chapter's own ordering of the two
#    (ContextBudget, then SummarizationMiddleware) - though since one hooks
#    `wrap_model_call` and the other `before_model`, their relative list
#    position does not change execution order: every `before_model` hook
#    still runs before the wrapped model call. See "Compose with
#    summarization". Chapter 23 adds two more constraints, one per hook
#    type it uses. Among the `wrap_tool_call` gates - `InjectionGuard`,
#    `RoleAuthorityGate`, `AuthorityGate`, `AuditGate` - `InjectionGuard`
#    goes outermost (first) because its untrusted-content tag is applied
#    on the way back OUT of the handler chain, and outer wrappers finish
#    last: whatever `InjectionGuard` does to a tool result is the final
#    transformation before that result re-enters the conversation, not
#    something a gate closer to the tool can still see raw or a gate
#    further out can strip. `RoleAuthorityGate` comes next, still outside
#    `AuthorityGate` - both block by returning without calling `handler`,
#    so the one that runs first (the more outer one) decides first, and
#    `atlas/security.py`'s own docstring is explicit that an unauthorized
#    role must never reach `AuthorityGate`'s approval-required check at
#    all. `AuditGate` sits innermost of the four, closest to the real
#    tool call - which means a call either gate blocks never reaches it,
#    a real gap between "records every tool call" (its own docstring) and
#    what this order actually audits, flagged here rather than silently
#    accepted. Separately, among the `wrap_model_call` middleware,
#    `TenantBudgetGuard` sits inside `context_budget` (a different hook
#    from the tool-call gates above), so its token estimate reflects the
#    already-trimmed request `context_budget` hands it, not the raw
#    pre-trim history.
# 2. `RoleAuthorityGate.wrap_tool_call` reads `request.runtime.context.role`
#    and `AuditGate.wrap_tool_call` reads `request.runtime.context.customer_id`
#    (and `.role`) - both need `context_schema=` so `create_agent` populates
#    `runtime.context` from what a caller passes as `context=` per invocation,
#    the same mechanism Chapter 16 used for a handoff tool's `runtime.state`.
#    See `atlas/security.py`'s module docstring.


async def build_resolve_agent():
    """Reach an external tool server over MCP and fold its tools in
    alongside the in-process ones. See "Reaching external tools with MCP".

    `client.get_tools()` can fail open - if any configured server fails to
    connect it can silently return fewer tools, or none - so the guard
    below refuses to start rather than run with a silently shrunken
    authority surface. See "Production considerations"."""
    client = MultiServerMCPClient(
        {
            "atlas-status": {
                "transport": "stdio",
                "command": "python",
                "args": ["atlas/mcp_server.py"],
            }
        }
    )
    mcp_tools = await client.get_tools()
    if not mcp_tools:  # <1>
        raise RuntimeError("MCP server returned no tools; refusing to start.")
    return create_agent(
        model="claude-sonnet-4-6",
        tools=RESOLVE_TOOLS + mcp_tools,  # <2>
        system_prompt=RESOLVE_PROMPT,
    )


# 1. The guard matters more than it looks - see "Production considerations".
#    get_tools() can return an empty list when a server fails to connect,
#    silently shrinking the authority surface.
# 2. MCP tools and in-process tools are the same type from here on; the
#    agent does not distinguish them.


def run_resolve(inputs: dict, config: dict) -> dict:
    """Chapter 20: the attribution layer around every turn `resolve_agent`
    takes. See "Naming the fleet: attribution across the supervisor
    topology". `name="resolve-agent"` above turns the span itself into a
    labeled one; the `trace()` context here adds tags and metadata ONCE, at
    this single entry point, rather than scattering `tags=` across call
    sites where they could drift out of sync. `route`, `thread_id`, and
    `customer_id` are expected on `config["configurable"]` by the caller -
    reusing Chapter 6's routing decision and Chapter 9/13's existing
    identifiers rather than inventing new ones. Building/entering `trace()`
    needs no live LangSmith connection - it is a local context manager that
    only submits data once `LANGSMITH_TRACING` is actually "true"."""
    with trace(
        name="atlas-turn",
        tags=["atlas", config["configurable"]["route"]],
        metadata={
            "thread_id": config["configurable"]["thread_id"],
            "customer_id": config["configurable"]["customer_id"],
        },
    ):
        return resolve_agent.invoke(inputs, config)
