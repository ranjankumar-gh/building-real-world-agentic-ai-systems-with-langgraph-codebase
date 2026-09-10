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

DELIBERATELY NOT WIRED INTO `langgraph.json`. Adding the `auth` key makes
every request to a locally running `langgraph up` need a token, which would
break the book's promise that a reader can run everything with no accounts
and no setup. Chapter 23 prints the one-line config change to make when you
deploy this for real:

    "auth": {"path": "./atlas/auth.py:auth"}

TOKENS ARE SEEDED AND MOCKABLE, like every other backend in this repo. In a
real deployment `verify_token` calls your identity provider and this table
does not exist.
"""

from typing import Any

from langgraph_sdk import Auth

auth = Auth()

# The seeded directory. Maps an opaque token to who holds it and what role
# that identity is entitled to use. `ROLE_TOOL_PERMISSIONS` in
# atlas/security.py is what a role then means in terms of tools.
DEV_IDENTITIES: dict[str, dict[str, str]] = {
    "dev-agent-token": {"identity": "agent-7", "role": "support_agent"},
    "dev-readonly-token": {"identity": "auditor-2", "role": "read_only"},
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


@auth.on
async def deny_by_default(ctx: Auth.types.AuthContext, value: Any) -> bool:
    """Default deny, the same posture atlas/security.py takes on tools. A
    resource with no explicit handler is refused rather than allowed, so
    adding a new resource type to the server cannot silently open it."""
    return False


@auth.on.threads
async def threads_are_scoped_to_their_owner(
    ctx: Auth.types.AuthContext, value: Any
) -> dict[str, str]:
    """Returning a dict makes it a metadata filter: the caller only sees, and
    only touches, threads stamped with their own identity.

    This is the tenancy boundary Chapter 13's per-customer namespace draws
    inside the store, drawn again at the API. Without it, an authenticated
    caller is authenticated to everyone's conversations at once."""
    return {"owner": ctx.user.identity}


@auth.on.assistants
async def assistants_are_read_only(
    ctx: Auth.types.AuthContext, value: Any
) -> bool:
    """Graphs are deployment artifacts, not customer data. Reading which
    assistants exist is fine; creating or mutating one from a support
    request is not, whatever role the caller holds."""
    return ctx.action in ("read", "search")
