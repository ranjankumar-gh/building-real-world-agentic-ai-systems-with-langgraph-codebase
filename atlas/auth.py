"""Chapter 23, "The role gate is only as good as the identity behind it":
server-side identity for the Agent Server Chapter 22 stands up.

`RoleAuthorityGate` (atlas/security.py) reads the caller's role from
`request.runtime.context.role`. In-process that is sound, because Atlas's own
code constructs the `AtlasContext` it passes to `resolve_agent`. Across a
network boundary it is not: `langgraph_sdk`'s `RunsClient.create` takes a
`context` argument, so an HTTP caller supplies the very role the gate then
checks. A gate that reads an attribute the caller controls is a convention,
not an authorization control.

This module is where the identity comes from instead. `@auth.authenticate`
runs server-side, before any graph does, and turns a credential into an
identity plus the permissions that identity actually holds. The role a
request may use is then derived from `ctx.user`, never read out of the
request body.

`context_for` is the other half: the served graph's resolve node
(`atlas/deploy/server.py`) builds the `AtlasContext` the gates read from
the identity this module proved - `runtime.server_info.user`, which the
Agent Server fills from `@auth.authenticate` - and from the ticket, never
from a `context` the caller sent. A run with no proved role gets the role
"anonymous", which `ROLE_TOOL_PERMISSIONS` grants nothing: default deny.

DELIBERATELY NOT WIRED INTO `langgraph.json`. Adding the `auth` key makes
every request to a locally running server need a token. Chapter 23 prints
the one-line config change to make when you deploy this for real:

    "auth": {"path": "./atlas/auth.py:auth"}

Without it every caller is anonymous, so the served graph's role gate
refuses every tool call: the server runs, and the gates hold.

TOKENS ARE SEEDED AND MOCKABLE, like every other backend in this repo. In a
real deployment `verify_token` calls your identity provider and this table
does not exist.
"""

from typing import Any
from uuid import UUID, uuid5

from langgraph_sdk import Auth

from atlas.security import APPROVER_ROLES, AtlasContext

auth = Auth()

# The seeded directory. Maps an opaque token to who holds it and what role
# that identity is entitled to use. `ROLE_TOOL_PERMISSIONS` in
# atlas/security.py is what a role then means in terms of tools.
DEV_IDENTITIES: dict[str, dict[str, str]] = {
    "dev-agent-token": {"identity": "agent-7", "role": "support_agent"},
    "dev-lead-token": {"identity": "lead-3", "role": "support_lead"},
    "dev-readonly-token": {"identity": "auditor-2", "role": "support_readonly"},
}


def verify_token(token: str) -> dict[str, str] | None:
    """The seam a real deployment replaces. Whatever is behind it - a JWT
    signature check, an introspection call, a session lookup - it must be
    something the caller cannot forge, which is the entire difference between
    this and reading a role out of the request."""
    return DEV_IDENTITIES.get(token)


@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Runs server-side on every request, before any graph is reached.

    The role is returned as a permission held BY the identity. That is the
    inversion this module exists for: `RoleAuthorityGate` can keep asking
    "what is this caller's role", and the answer now comes from a verified
    credential rather than from a field the caller filled in."""
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Unauthorized")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Unauthorized")

    record = verify_token(token)
    if record is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Unauthorized")

    return {
        "identity": record["identity"],
        "permissions": [f"role:{record['role']}"],
    }


def role_of(user: Any) -> str | None:
    """The authenticated role, read back off the identity rather than off the
    request. A deployment builds `AtlasContext` from this, so the role the
    gate checks is the role the credential proved."""
    for permission in getattr(user, "permissions", []) or []:
        if permission.startswith("role:"):
            return permission.removeprefix("role:")
    return None


def context_for(user: Any, customer_id: str | None) -> AtlasContext:
    """The `AtlasContext` a served run gets: the proved role, or none."""
    return AtlasContext(
        role=role_of(user) or "anonymous",  # no proved role: no tools
        customer_id=customer_id or "unknown",
    )


@auth.on
async def deny_by_default(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """Default deny, the same posture atlas/security.py takes on tools. A
    resource with no explicit handler is refused rather than allowed, so
    adding a new resource type to the server cannot silently open it."""
    return False


def _run_body(value: Any) -> dict:
    """The run a request asks for: a run's body arrives as `kwargs`, a
    cron's as `payload` (the SDK's RunsCreate and CronsCreate)."""
    return value.get("kwargs") or value.get("payload") or {}


def _steers_the_graph(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A run whose command carries `goto` or `update` moves the graph or
    writes its state from outside: only an approver may send one."""
    command = _run_body(value).get("command") or {}
    return bool(command.get("goto") or command.get("update")) and (
        role_of(ctx.user) not in APPROVER_ROLES
    )


GRAPH_OWNED_KEYS = frozenset({"approval", "refund_done"})  # only nodes write these


def _plants_state(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A run whose input sets a key only the graph's nodes write - the gate's
    `approval`, the refund's `refund_done`, anything named audit* - or a
    thread created with `supersteps` (state applied with no run at all),
    from a non-approver."""
    run_input = _run_body(value).get("input")
    keys = set(run_input) if isinstance(run_input, dict) else set()
    planted = keys & GRAPH_OWNED_KEYS or any(k.startswith("audit") for k in keys)
    return bool(planted or value.get("supersteps")) and (
        role_of(ctx.user) not in APPROVER_ROLES
    )


# The Agent Server names a graph's default assistant uuid5(NAMESPACE_GRAPH,
# graph_id) (langgraph-api 0.14.0, langgraph_api/graph.py). A run arrives at
# the hook with that UUID; a cron's payload still carries what the caller sent.
NAMESPACE_GRAPH = UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
APPROVER_ONLY_GRAPHS = frozenset({"sla-watch"})
_APPROVER_ONLY_IDS = APPROVER_ONLY_GRAPHS | {
    str(uuid5(NAMESPACE_GRAPH, graph_id)) for graph_id in APPROVER_ONLY_GRAPHS
}


def _canonical(assistant: Any) -> str:
    """A UUID in its one canonical spelling; any other value as given."""
    try:
        return str(UUID(str(assistant)))
    except ValueError:
        return str(assistant)


def _starts_an_approver_graph(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A run or cron on SLA Watch (Chapter 27) from a non-approver. Its
    drafts claim tickets, and only an approver may act on a check-in, so a
    run nobody can approve would only hold tickets back from the next scan.

    Keyed on assistant ids, not graph ids: neither a run's nor a cron's
    request names its graph, so an extra assistant on the sla-watch graph
    is not caught here. Callers cannot create one (assistants are
    read-only to them); the in-graph checks still refuse any send."""
    assistant = value.get("assistant_id") or _run_body(value).get("assistant_id")
    return _canonical(assistant) in _APPROVER_ONLY_IDS and (
        role_of(ctx.user) not in APPROVER_ROLES
    )


def _writes_state(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A threads "update" that is a state write, from a non-approver.

    The hook never sees the values written: the in-memory runtime passes a
    state write as ThreadsUpdate(thread_id=...) alone, so a write that would
    touch `ticket`, `approval` or `refund_done` cannot be told from any other.
    A metadata patch carries `metadata`, a run cancel carries `action`; any
    other update is a state write, and only an approver may make one. That
    includes `threads.create(supersteps=...)`: the runtime authorizes the
    thread as a "create" and then applies the supersteps as an "update" of
    this same shape.

    This test rests on the runtime's request shape, verified by probe on
    langgraph-api 0.14.0 with langgraph-runtime-inmem 0.34.2 only. Any value
    that carries a `metadata` key passes it - a state write that arrives as
    `{"thread_id": ..., "metadata": {}}` is let through as if it were a
    metadata patch - so a runtime that sends `metadata` with a state write
    gets past it. It is defense in depth only: `refund` verifies the gate's
    own audit row, so a state write that got through would still charge
    nothing."""
    return (
        getattr(ctx, "resource", "threads") == "threads"  # a cron update is not one
        and ctx.action == "update"
        and "metadata" not in value
        and not value.get("action")
        and role_of(ctx.user) not in APPROVER_ROLES
    )


def _refuse(detail: str) -> Auth.exceptions.HTTPException:
    return Auth.exceptions.HTTPException(status_code=403, detail=detail)


def _owned(ctx: Auth.types.AuthContext, value: Any) -> dict[str, str] | None:
    """Stamp a new resource with its owner, then filter every access to it.

    An approver is not filtered: a support lead reads and resumes whichever
    thread holds the refund waiting for them (role-based access). Everyone
    else sees only what they own. Ownership is the server's to set - it
    stamps `owner` on create - and no caller, approver or not, may change it
    afterwards."""
    sends_a_run = ctx.action in ("create", "create_run") or "payload" in value
    if sends_a_run and _steers_the_graph(ctx, value):
        raise _refuse("only an approver may send goto or update")  # <1>
    if sends_a_run and _plants_state(ctx, value):
        raise _refuse("only an approver may set graph-owned state")
    if sends_a_run and _starts_an_approver_graph(ctx, value):
        raise _refuse("only an approver may run sla-watch")  # <4>
    if ctx.action == "update" and "owner" in (value.get("metadata") or {}):
        raise _refuse("thread ownership is set by the server")  # <2>
    if _writes_state(ctx, value):
        raise _refuse("only an approver may write thread state")
    if ctx.action in ("create", "create_run"):
        value.setdefault("metadata", {})["owner"] = ctx.user.identity
    if role_of(ctx.user) in APPROVER_ROLES:
        return None  # <3>
    return {"owner": ctx.user.identity}


# 1. Defense in depth, not the control: the gate binds a decision to what
#    the approver was shown, and `refund` charges only against the gate's
#    own audit row (atlas/graph.py). The in-memory runtime passes the run's
#    request body to this hook as value["kwargs"] (langgraph-runtime-inmem
#    0.34.2, checked by probe); a runtime that does not leaves `kwargs`
#    empty, and the checks inside the graph still hold. The same goes for
#    `_plants_state` and `_writes_state`. A cron carries its run as
#    `payload` (crons are stamped and filtered by `_owned` too), so both
#    checks read it there as well.
# 2. Without this, a thread owner could hand a thread to someone else by
#    patching `metadata.owner`, and the thread would show up in their view.
# 3. No filter at all. The owner stamp still records who opened the thread.
# 4. Defense in depth again: SLA Watch's claims lapse after a day
#    (atlas/sla_watch.py), so even a run that got past this hook could hold
#    a ticket back for one day, not indefinitely.


@auth.on.threads
async def threads_are_scoped_to_their_owner(
    ctx: Auth.types.AuthContext, value: Any
) -> dict[str, str] | None:
    """Returning a dict makes it a metadata filter: the caller only sees, and
    only touches, threads stamped with their own identity, and a thread it
    creates is stamped with that identity (the pinned SDK's own pattern).
    An approver gets no filter (see `_owned`).

    This is the tenancy boundary Chapter 13's per-customer namespace draws
    inside the store, drawn again at the API. Without it, an authenticated
    caller is authenticated to everyone's conversations at once."""
    return _owned(ctx, value)


@auth.on.crons
async def crons_are_scoped_to_their_owner(
    ctx: Auth.types.AuthContext, value: Any
) -> dict[str, str] | None:
    """Chapter 22's scheduled monitor is a cron. Without this handler the
    default deny above refuses `crons.create`; with it, a cron is owned the
    way a thread is, and its runs carry the owner's identity."""
    return _owned(ctx, value)


@auth.on.assistants
async def assistants_are_read_only(
    ctx: Auth.types.AuthContext, value: Any
) -> bool:
    """Graphs are deployment artifacts, not customer data. Reading which
    assistants exist is fine; creating or mutating one from a support
    request is not, whatever role the caller holds."""
    return ctx.action in ("read", "search")


@auth.on.store
async def the_store_is_the_graphs(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """The HTTP store API (`client.store.*`), which the graph's own
    `runtime.store` never passes through.

    The audit log is the record `refund` charges against, so no caller may
    read or write any namespace under "audit" - nor search from an empty
    prefix, which would reach it. A customer's namespaces ("customer", id,
    ...) are readable only by an identity holding `customer:<id>` (no seeded
    token does); nothing else is open. Every refusal is a 403."""
    namespace = tuple(value.get("namespace") or ())
    if not namespace or namespace[0] == "audit":
        raise _refuse("the audit log is written by the graph only")
    permissions = getattr(ctx.user, "permissions", None) or []
    return (
        namespace[0] == "customer"
        and len(namespace) > 1
        and ctx.action in ("get", "search")
        and f"customer:{namespace[1]}" in permissions
    )
