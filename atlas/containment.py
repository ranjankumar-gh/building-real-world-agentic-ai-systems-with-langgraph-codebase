"""Chapter 23, "Stopping a run that is already doing damage": the second
rung of the containment ladder.

The revocation record lives in the BaseStore Chapter 13 wired up, NOT in
AtlasState. That is the whole design. A revocation written into graph state
is restored along with the state when Chapter 9's time travel rewinds a
thread, which hands a run back the authority a human took away - see
"Authority never sits in state"."""

from collections.abc import Callable
from datetime import datetime, timezone

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.store.base import BaseStore


def revocation_ns(subject: str) -> tuple:
    """One namespace per subject. Outside every thread, so no checkpoint
    restore can reach it."""
    return ("containment", subject)


def revoke(store: BaseStore, subject: str, reason: str) -> None:
    """Take back authority already granted, persisted with whatever
    durability the caller's store provides - gone on restart with
    `atlas/memory.py`'s dev-default `build_dev_store()`, durable across
    one with `build_prod_store(db_uri)` in production. One-way in either
    case: nothing in this module reinstates a revocation, because
    reinstatement is a human decision made outside the run that tripped."""
    store.put(
        revocation_ns(subject),
        "revocation",
        {"reason": reason, "revoked_at": datetime.now(timezone.utc).isoformat()},
    )


def is_revoked(store: BaseStore, subject: str) -> bool:
    return store.get(revocation_ns(subject), "revocation") is not None


class RevocationGate(AgentMiddleware):
    """Fails every model call for a revoked subject. Placed outermost in the
    stack so a revoked run stops before any other middleware spends a token
    or writes an audit row.

    Raises rather than returning a `ModelResponse`, unlike
    `atlas/cost.py`'s `TenantBudgetGuard.degrade` - see the module's own
    inline note below `wrap_model_call` for why the two situations aren't
    the same shape."""

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        subject: str = request.runtime.context.customer_id
        if is_revoked(self.store, subject):
            raise RuntimeError(f"authority revoked for {subject}")  # <1>
        return handler(request)


# 1. `TenantBudgetGuard.wrap_model_call` (atlas/cost.py) refuses a call by
#    RETURNING a `ModelResponse` - a graceful decline that becomes a normal
#    AIMessage, saved to state, and carried on through every after_model
#    hook still ahead of it (PIIMiddleware's, `approval`'s). That is right
#    for a monthly cap: an expected, recurring business rule the caller
#    should see as an ordinary turn. Revocation is not that. Read against
#    `langchain/agents/factory.py`: `wrap_model_call` runs inside the
#    "model" node itself, and the after_model middleware chain is a
#    SEPARATE downstream node reached only if "model" returns without
#    raising. A returned `ModelResponse` here would let a revoked run's
#    synthetic refusal complete that node successfully and flow into
#    PII redaction, human-in-the-loop review, and back into state as an
#    unremarkable assistant turn - indistinguishable in shape from a run
#    that was never revoked at all, which is exactly the outcome
#    containment exists to prevent. Raising aborts the "model" node
#    outright: nothing downstream of it runs, and the exception propagates
#    out of `.invoke()` where the caller cannot mistake it for a completed
#    turn. `RoleAuthorityGate`/`AuthorityGate` (atlas/security.py,
#    atlas/middleware.py) refuse by returning an error `ToolMessage`
#    instead, but that is a different hook: `wrap_tool_call` sits inside
#    the "tools" node, where the graph's contract requires exactly one
#    `ToolMessage` per tool_call to proceed to the next model turn -
#    there is no "abort the whole run" primitive available, or desirable,
#    for refusing a single call among possibly several. `wrap_model_call`
#    carries no equivalent per-item contract, so raising is available here
#    and, verified against the graph wiring above, the only way to
#    guarantee that a revoked subject's run stops rather than politely
#    continues.
