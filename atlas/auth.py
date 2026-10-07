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


def _steers_the_graph(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A run whose command carries `goto` or `update` moves the graph or
    writes its state from outside: only an approver may send one."""
    command = (value.get("kwargs") or {}).get("command") or {}
    return bool(command.get("goto") or command.get("update")) and (
        role_of(ctx.user) not in APPROVER_ROLES
    )


def _writes_state(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """A threads "update" that is a state write, from a non-approver.

    The hook never sees the values written: the in-memory runtime passes a
    state write as ThreadsUpdate(thread_id=...) alone, so a write that would
    touch `ticket`, `approval` or `refund_done` cannot be told from any other.
    A metadata patch carries `metadata`, a run cancel carries `action`; any
    other update is a state write, and only an approver may make one."""
    return (
        ctx.action == "update"
        and "metadata" not in value
        and not value.get("action")
        and role_of(ctx.user) not in APPROVER_ROLES
    )


def _owned(ctx: Auth.types.AuthContext, value: Any) -> dict[str, str]:
    """Stamp a new resource with its owner, then filter every access to it."""
    if ctx.action == "create_run" and _steers_the_graph(ctx, value):  # <1>
        raise Auth.exceptions.HTTPException(
            status_code=403, detail="only an approver may send goto or update"
        )
    if _writes_state(ctx, value):
        raise Auth.exceptions.HTTPException(
            status_code=403, detail="only an approver may write thread state"
        )
    if ctx.action in ("create", "create_run"):
        value.setdefault("metadata", {})["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}


# 1. Defense in depth, not the control: the gate binds a decision to what
#    the approver was shown and `refund` refuses a non-approver on its own
#    (atlas/graph.py). The in-memory runtime passes the run's request body to
#    this hook as value["kwargs"] (langgraph-runtime-inmem 0.34.2, checked by
#    probe); a runtime that does not leaves `kwargs` empty, and the checks
#    inside the graph still hold. The same goes for `_writes_state`.


@auth.on.threads
async def threads_are_scoped_to_their_owner(
    ctx: Auth.types.AuthContext, value: Any
) -> dict[str, str]:
    """Returning a dict makes it a metadata filter: the caller only sees, and
    only touches, threads stamped with their own identity, and a thread it
    creates is stamped with that identity (the pinned SDK's own pattern).

    This is the tenancy boundary Chapter 13's per-customer namespace draws
    inside the store, drawn again at the API. Without it, an authenticated
    caller is authenticated to everyone's conversations at once."""
    return _owned(ctx, value)


@auth.on.crons
async def crons_are_scoped_to_their_owner(
    ctx: Auth.types.AuthContext, value: Any
) -> dict[str, str]:
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
