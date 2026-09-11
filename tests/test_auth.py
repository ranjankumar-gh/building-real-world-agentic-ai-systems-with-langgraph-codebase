"""Chapter 23, "The role gate is only as good as the identity behind it" -
atlas/auth.py.

No server needed. `Auth` handlers are plain async functions with a declared
shape, so each one is callable directly with a stand-in context, the same way
tests/test_security.py drives middleware without an agent loop. That keeps
this suite offline, which is the point: the identity layer is the last place
that should only be exercisable against a deployment.
"""

import asyncio
from typing import Any

import pytest
from langgraph_sdk import Auth

from atlas.auth import (
    assistants_are_read_only,
    authenticate,
    deny_by_default,
    role_of,
    threads_are_scoped_to_their_owner,
)


class _User:
    """The shape `ctx.user` presents to a handler: an identity and the
    permissions the credential proved."""

    def __init__(self, identity: str, permissions: list[str] | None = None) -> None:
        self.identity = identity
        self.permissions = permissions or []


class _Ctx:
    def __init__(self, user: _User, action: str = "create") -> None:
        self.user = user
        self.action = action


def _authenticate(authorization: str | None) -> Any:
    return asyncio.run(authenticate(authorization))


def test_a_valid_bearer_token_yields_an_identity_and_its_role() -> None:
    result = _authenticate("Bearer dev-agent-token")

    assert result["identity"] == "agent-7"
    assert result["permissions"] == ["role:support_agent"]


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "dev-agent-token",  # no scheme
        "Basic dev-agent-token",  # wrong scheme
        "Bearer ",  # scheme, no token
        "Bearer not-a-real-token",
    ],
)
def test_every_way_of_not_proving_identity_is_a_401(authorization: str | None) -> None:
    """Default deny at the door. A missing, malformed, or unknown credential
    all end the same way, because "could not prove it" and "proved it wrong"
    are the same answer to a caller."""
    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _authenticate(authorization)

    assert excinfo.value.status_code == 401


def test_the_role_is_read_off_the_identity_not_off_the_request() -> None:
    """The whole point of the module. `RoleAuthorityGate` asks what role the
    caller has; `role_of` answers from the permissions the credential proved,
    so the answer cannot be supplied by the caller."""
    proved = _authenticate("Bearer dev-readonly-token")

    assert role_of(_User(proved["identity"], proved["permissions"])) == "read_only"


def test_an_identity_with_no_role_permission_has_no_role() -> None:
    """No silent default. An identity carrying no role permission returns
    None rather than falling back to something permissive."""
    assert role_of(_User("agent-7", ["threads:write"])) is None
    assert role_of(_User("agent-7", [])) is None


def test_two_tokens_map_to_two_different_roles() -> None:
    agent = _authenticate("Bearer dev-agent-token")
    auditor = _authenticate("Bearer dev-readonly-token")

    assert role_of(_User(agent["identity"], agent["permissions"])) == "support_agent"
    assert role_of(_User(auditor["identity"], auditor["permissions"])) == "read_only"


def test_an_unhandled_resource_is_denied_rather_than_allowed() -> None:
    """The catch-all. Adding a new resource type to the server cannot
    silently open it, the same default-deny posture atlas/security.py takes
    on tools."""
    assert asyncio.run(deny_by_default(_Ctx(_User("agent-7")), {})) is False


def test_threads_are_filtered_to_the_calling_identity() -> None:
    """Returning a dict makes it a metadata filter. Without it, an
    authenticated caller is authenticated to everyone's conversations."""
    result = asyncio.run(
        threads_are_scoped_to_their_owner(_Ctx(_User("agent-7")), {})
    )

    assert result == {"owner": "agent-7"}


def test_two_identities_get_two_different_thread_filters() -> None:
    """The tenancy boundary Chapter 13 draws inside the store, drawn again at
    the API."""
    mine = asyncio.run(threads_are_scoped_to_their_owner(_Ctx(_User("agent-7")), {}))
    theirs = asyncio.run(
        threads_are_scoped_to_their_owner(_Ctx(_User("auditor-2")), {})
    )

    assert mine != theirs


@pytest.mark.parametrize("action", ["read", "search"])
def test_assistants_may_be_read(action: str) -> None:
    ctx = _Ctx(_User("agent-7"), action=action)

    assert asyncio.run(assistants_are_read_only(ctx, {})) is True


@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_assistants_may_not_be_mutated_whatever_the_role(action: str) -> None:
    """Graphs are deployment artifacts, not customer data. No role reachable
    over this API gets to change one."""
    ctx = _Ctx(_User("agent-7", ["role:support_agent"]), action=action)

    assert asyncio.run(assistants_are_read_only(ctx, {})) is False
