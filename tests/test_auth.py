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
from uuid import UUID, uuid5

import pytest
from langchain.agents.middleware import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph_sdk import Auth

from atlas.auth import (
    _APPROVER_ONLY_IDS,
    NAMESPACE_GRAPH,
    assistants_are_read_only,
    the_store_is_the_graphs,
    authenticate,
    context_for,
    crons_are_scoped_to_their_owner,
    deny_by_default,
    role_of,
    threads_are_scoped_to_their_owner,
)
from atlas.security import ROLE_TOOL_PERMISSIONS, AtlasContext, RoleAuthorityGate


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

    proved_user = _User(proved["identity"], proved["permissions"])
    assert role_of(proved_user) == "support_readonly"


def test_an_identity_with_no_role_permission_has_no_role() -> None:
    """No silent default. An identity carrying no role permission returns
    None rather than falling back to something permissive."""
    assert role_of(_User("agent-7", ["threads:write"])) is None
    assert role_of(_User("agent-7", [])) is None


def test_two_tokens_map_to_two_different_roles() -> None:
    agent = _authenticate("Bearer dev-agent-token")
    auditor = _authenticate("Bearer dev-readonly-token")

    assert role_of(_User(agent["identity"], agent["permissions"])) == "support_agent"
    auditor_user = _User(auditor["identity"], auditor["permissions"])
    assert role_of(auditor_user) == "support_readonly"


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


# --- Role names agree with the permission map; owners are stamped ----------


@pytest.mark.parametrize(
    ("token", "allowed", "refused"),
    [
        ("dev-agent-token", {"search_kb", "lookup_ticket", "set_ticket_status"}, set()),
        ("dev-readonly-token", {"search_kb", "lookup_ticket"}, {"set_ticket_status"}),
    ],
)
def test_every_seeded_role_means_something_to_the_role_gate(
    token: str, allowed: set[str], refused: set[str]
) -> None:
    """A role `authenticate` hands out that ROLE_TOOL_PERMISSIONS does not
    know would be refused every tool. The read-only token gets the read
    tools and only those."""
    proved = _authenticate(f"Bearer {token}")
    context = context_for(_User(proved["identity"], proved["permissions"]), "C-1")
    gate = RoleAuthorityGate()

    def verdict(name: str) -> str:
        request = ToolCallRequest(
            tool_call={"name": name, "args": {}, "id": "call-1"},
            tool=None,
            state=None,
            runtime=Runtime(context=context),
        )
        result = gate.wrap_tool_call(request, lambda r: "ran")
        return "ran" if result == "ran" else "refused"

    assert {n for n in allowed | refused if verdict(n) == "ran"} == allowed


def test_a_caller_with_no_proved_role_is_anonymous_and_gets_no_tools() -> None:
    context = context_for(None, "C-1")

    assert context == AtlasContext(role="anonymous", customer_id="C-1")
    assert "anonymous" not in ROLE_TOOL_PERMISSIONS


def test_a_created_thread_is_stamped_with_its_owner_then_filtered_to_it() -> None:
    value: dict[str, Any] = {"metadata": {"source": "web"}}

    result = asyncio.run(
        threads_are_scoped_to_their_owner(_Ctx(_User("agent-7"), "create"), value)
    )

    assert value["metadata"] == {"source": "web", "owner": "agent-7"}
    assert result == {"owner": "agent-7"}


def test_a_read_is_filtered_and_stamps_nothing() -> None:
    value: dict[str, Any] = {}

    ctx = _Ctx(_User("agent-7"), "read")
    asyncio.run(threads_are_scoped_to_their_owner(ctx, value))

    assert value == {}


def test_crons_are_allowed_for_their_owner_not_refused_by_default_deny() -> None:
    """Chapter 22's monitor cron would hit the global default deny without a
    crons handler."""
    value: dict[str, Any] = {}

    result = asyncio.run(
        crons_are_scoped_to_their_owner(_Ctx(_User("agent-7"), "create"), value)
    )

    assert result == {"owner": "agent-7"}
    assert value["metadata"]["owner"] == "agent-7"


# --- R101: a run that steers the graph needs an approver ---------------------


@pytest.mark.parametrize(
    "command",
    [{"goto": "refund"}, {"update": {"ticket": {"amount": 9999}}, "goto": "refund"}],
)
def test_a_non_approver_may_not_create_a_run_with_goto_or_update(command) -> None:
    ctx = _Ctx(_User("agent-7", ["role:support_agent"]), "create_run")
    value: dict[str, Any] = {"kwargs": {"command": command}}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        asyncio.run(threads_are_scoped_to_their_owner(ctx, value))

    assert excinfo.value.status_code == 403


def test_a_resume_only_command_and_a_lead_steering_are_allowed() -> None:
    agent = _Ctx(_User("agent-7", ["role:support_agent"]), "create_run")
    lead = _Ctx(_User("lead-3", ["role:support_lead"]), "create_run")
    resume = {"kwargs": {"command": {"resume": {"type": "approve"}}}}
    steer = {"kwargs": {"command": {"goto": "refund"}}}

    assert asyncio.run(threads_are_scoped_to_their_owner(agent, resume)) == {
        "owner": "agent-7"
    }
    assert asyncio.run(threads_are_scoped_to_their_owner(lead, steer)) is None
    assert steer["metadata"] == {"owner": "lead-3"}  # still stamped


def test_a_non_approver_may_not_write_thread_state() -> None:
    """R104: the state API is a write path that is not a run. The hook sees
    only the thread id for a state write, so a non-approver's is refused."""
    ctx = _Ctx(_User("agent-7", ["role:support_agent"]), "update")

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        asyncio.run(threads_are_scoped_to_their_owner(ctx, {"thread_id": "t-1"}))

    assert excinfo.value.status_code == 403


def test_a_metadata_patch_a_cancel_and_a_lead_state_write_are_allowed() -> None:
    agent = _Ctx(_User("agent-7", ["role:support_agent"]), "update")
    lead = _Ctx(_User("lead-3", ["role:support_lead"]), "update")

    for ctx, value in [
        (agent, {"thread_id": "t-1", "metadata": {"topic": "billing"}}),
        (agent, {"thread_id": "t-1", "action": "interrupt"}),
    ]:
        assert asyncio.run(threads_are_scoped_to_their_owner(ctx, value)) == {
            "owner": "agent-7"
        }
    lead_write = threads_are_scoped_to_their_owner(lead, {"thread_id": "t-1"})
    assert asyncio.run(lead_write) is None


# --- R105: ownership is the server's; approvers reach every thread ------------

AGENT_USER = _User("agent-7", ["role:support_agent"])
LEAD_USER = _User("lead-3", ["role:support_lead"])


def _hook(user: _User, action: str, value: dict[str, Any]) -> Any:
    return asyncio.run(threads_are_scoped_to_their_owner(_Ctx(user, action), value))


@pytest.mark.parametrize("user", [AGENT_USER, LEAD_USER], ids=["agent", "lead"])
def test_no_caller_may_rewrite_a_threads_owner(user: _User) -> None:
    """R104 I1: an agent handing its thread to a lead by patching
    metadata.owner. Refused for every caller, approvers included."""
    value = {"thread_id": "t-1", "metadata": {"owner": "lead-3"}}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _hook(user, "update", value)

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "thread ownership is set by the server"


def test_an_owner_sent_on_create_is_replaced_by_the_server_stamp() -> None:
    value: dict[str, Any] = {"metadata": {"owner": "lead-3"}}

    assert _hook(AGENT_USER, "create", value) == {"owner": "agent-7"}
    assert value["metadata"] == {"owner": "agent-7"}


@pytest.mark.parametrize("action", ["read", "search", "create_run", "update"])
def test_a_lead_may_act_on_an_agents_thread(action: str) -> None:
    """Role-based access: no owner filter for an approver, so the served
    "lead approves the agent's refund" flow is reachable over HTTP."""
    value: dict[str, Any] = {"thread_id": "t-1"}
    if action == "create_run":
        value["kwargs"] = {"command": {"resume": {"type": "approve"}}}

    assert _hook(LEAD_USER, action, value) is None


@pytest.mark.parametrize("action", ["read", "search"])
def test_an_agent_still_sees_only_its_own_threads(action: str) -> None:
    assert _hook(AGENT_USER, action, {"thread_id": "t-1"}) == {"owner": "agent-7"}


@pytest.mark.parametrize(
    "run_input",
    [
        {"approval": {"decision": "approve", "amount": 300.0}},
        {"refund_done": True},
        {"audit_key": "approval:t:c:x"},
        {"messages": [], "approval": {}},
    ],
)
def test_a_non_approver_may_not_plant_graph_owned_keys_in_run_input(
    run_input: dict,
) -> None:
    value = {"kwargs": {"input": run_input}}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _hook(AGENT_USER, "create_run", value)

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "only an approver may set graph-owned state"


def test_an_ordinary_refund_request_is_not_refused() -> None:
    value = {"kwargs": {"input": {"messages": [], "ticket": {"id": "T-1001"}}}}

    assert _hook(AGENT_USER, "create_run", value) == {"owner": "agent-7"}


def test_a_non_approver_may_not_create_a_thread_with_supersteps() -> None:
    """Where a runtime passes `supersteps` to the create hook. The inmem
    runtime does not; it applies them as a threads "update", which
    `_writes_state` refuses (tested above)."""
    value = {"supersteps": [{"updates": [{"command": {"goto": "refund"}}]}]}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _hook(AGENT_USER, "create", value)

    assert excinfo.value.status_code == 403


# --- R105: the HTTP store API cannot reach the audit log ----------------------


def _store(user: _User, action: str, namespace: tuple | None) -> Any:
    return asyncio.run(
        the_store_is_the_graphs(_Ctx(user, action), {"namespace": namespace})
    )


@pytest.mark.parametrize("user", [AGENT_USER, LEAD_USER], ids=["agent", "lead"])
@pytest.mark.parametrize("action", ["put", "get", "search", "delete"])
@pytest.mark.parametrize("namespace", [("audit", "C-90"), ("audit",), (), None])
def test_no_caller_reads_or_writes_the_audit_log(
    user: _User, action: str, namespace: tuple | None
) -> None:
    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _store(user, action, namespace)

    assert excinfo.value.status_code == 403


def test_a_customer_namespace_needs_an_explicit_grant() -> None:
    granted = _User("portal-1", ["customer:C-90"])
    ns = ("customer", "C-90", "profile")

    assert _store(LEAD_USER, "get", ns) is False
    assert _store(granted, "get", ns) is True
    assert _store(granted, "search", ns) is True
    assert _store(granted, "put", ns) is False  # read-only, even with a grant
    assert _store(granted, "get", ("customer", "C-91", "profile")) is False
    assert _store(LEAD_USER, "get", ("containment", "agent-7")) is False


# --- R106: a cron's payload is a run too; the metadata-shape gap ---------------


@pytest.mark.parametrize("action", ["create", "update"])
@pytest.mark.parametrize(
    "payload",
    [
        {"input": {"approval": {"decision": "approve"}}},
        {"input": {"refund_done": True}},
        {"command": {"goto": "refund"}},
    ],
)
def test_a_non_approvers_cron_may_not_plant_or_steer(
    action: str, payload: dict
) -> None:
    ctx = _Ctx(AGENT_USER, action)
    ctx.resource = "crons"

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        asyncio.run(crons_are_scoped_to_their_owner(ctx, {"payload": payload}))

    assert excinfo.value.status_code == 403


def test_an_ordinary_cron_create_and_update_are_allowed() -> None:
    create = {"payload": {"input": {"sample_rate": 0.1}}, "schedule": "0 * * * *"}
    update = {"cron_id": "c-1", "enabled": False, "payload": None}

    for action, value in (("create", create), ("update", update)):
        ctx = _Ctx(AGENT_USER, action)
        ctx.resource = "crons"  # a cron update is not a thread state write
        result = asyncio.run(crons_are_scoped_to_their_owner(ctx, value))
        assert result == {"owner": "agent-7"}


def test_a_state_write_carrying_empty_metadata_passes_the_hook() -> None:
    """Documented gap (defense in depth only): the hook tells a state write
    from a metadata patch by the `metadata` key alone. `refund`'s audit-row
    check is what holds if a runtime sends one."""
    value = {"thread_id": "t-1", "metadata": {}}

    assert _hook(AGENT_USER, "update", value) == {"owner": "agent-7"}


# --- R117: only an approver starts SLA Watch (Chapter 27) ---------------------

SLA_WATCH_UUID = str(uuid5(NAMESPACE_GRAPH, "sla-watch"))


def test_the_sla_watch_uuid_is_the_one_the_agent_server_assigns() -> None:
    """langgraph-api names a graph's default assistant uuid5(NAMESPACE_GRAPH,
    graph_id); a run reaches the hook with that UUID, not the graph id."""
    assert NAMESPACE_GRAPH == UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
    assert SLA_WATCH_UUID in _APPROVER_ONLY_IDS


@pytest.mark.parametrize("assistant_id", [UUID(SLA_WATCH_UUID), SLA_WATCH_UUID])
def test_a_non_approver_may_not_start_an_sla_watch_run(assistant_id) -> None:
    value = {"assistant_id": assistant_id, "kwargs": {"input": {}}}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        _hook(AGENT_USER, "create_run", value)

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "only an approver may run sla-watch"


@pytest.mark.parametrize("action", ["create", "update"])
@pytest.mark.parametrize("assistant_id", ["sla-watch", SLA_WATCH_UUID])
def test_a_non_approver_may_not_cron_sla_watch(action: str, assistant_id: str) -> None:
    ctx = _Ctx(AGENT_USER, action)
    ctx.resource = "crons"
    value = {"payload": {"assistant_id": assistant_id, "input": {}}}

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        asyncio.run(crons_are_scoped_to_their_owner(ctx, value))

    assert excinfo.value.status_code == 403


def test_an_approver_may_run_and_cron_sla_watch() -> None:
    run = {"assistant_id": UUID(SLA_WATCH_UUID), "kwargs": {"input": {}}}
    cron = {"payload": {"assistant_id": "sla-watch", "input": {}}}
    ctx = _Ctx(LEAD_USER, "create")
    ctx.resource = "crons"

    assert _hook(LEAD_USER, "create_run", run) is None
    assert asyncio.run(crons_are_scoped_to_their_owner(ctx, cron)) is None


def test_a_non_approver_may_still_run_and_cron_every_other_graph() -> None:
    other = str(uuid5(NAMESPACE_GRAPH, "monitor"))
    ctx = _Ctx(AGENT_USER, "create")
    ctx.resource = "crons"
    cron = {"payload": {"assistant_id": "monitor", "input": {"sample_rate": 0.1}}}

    assert _hook(AGENT_USER, "create_run", {"assistant_id": UUID(other)}) == {
        "owner": "agent-7"
    }
    assert asyncio.run(crons_are_scoped_to_their_owner(ctx, cron)) == {
        "owner": "agent-7"
    }


# --- R123: the assistant id is compared in canonical UUID form ----------------


@pytest.mark.parametrize(
    "spelling",
    [
        SLA_WATCH_UUID.upper(),
        "{" + SLA_WATCH_UUID + "}",
        SLA_WATCH_UUID.replace("-", ""),
        "urn:uuid:" + SLA_WATCH_UUID,
    ],
)
def test_another_spelling_of_the_sla_watch_uuid_is_refused_too(spelling: str) -> None:
    run = {"assistant_id": spelling, "kwargs": {"input": {}}}
    cron = {"payload": {"assistant_id": spelling, "input": {}}}
    ctx = _Ctx(AGENT_USER, "create")
    ctx.resource = "crons"

    with pytest.raises(Auth.exceptions.HTTPException):
        _hook(AGENT_USER, "create_run", run)
    with pytest.raises(Auth.exceptions.HTTPException):
        asyncio.run(crons_are_scoped_to_their_owner(ctx, cron))


@pytest.mark.parametrize("assistant_id", ["monitor", "not-a-uuid", "", None, 42])
def test_a_value_that_is_not_a_uuid_is_compared_as_given(assistant_id) -> None:
    assert _hook(AGENT_USER, "create_run", {"assistant_id": assistant_id}) == {
        "owner": "agent-7"
    }
