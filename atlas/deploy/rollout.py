"""Chapter 22, "Deployment and Scaling" - the drain-migrate-deploy script.

See "The drain-migrate-deploy script". Chapter 10 built
`RunControl.request_drain()` for one process: stop accepting new work, let
the current superstep finish, leave a resumable checkpoint - it raises
`GraphDrained` inside the process that owns the run, not to anyone watching
from outside. An external rollout script can't poll a LangGraph object
across process boundaries, so each replica needs its own small readiness
endpoint reporting whether it has finished draining - the orchestration
piece this chapter adds on top of Chapter 10's primitive, not a replacement
for it.

`load_balancer` here is Atlas's own thin wrapper (not a LangGraph API, and
not implemented in this repo - the companion repo has no real fleet to
front): it is expected to expose `remove`, `add`, `request_drain`, and
`is_drained`, where `request_drain` calls the replica's readiness endpoint,
which in turn calls the real `RunControl.request_drain()` inside that
process. `tests/test_rollout.py` exercises `drain_replica`/`rolling_deploy`
against a small fake that implements exactly that interface, since the
orchestration logic itself - remove before drain, one replica at a time,
raise on a drain that never finishes - needs no live fleet to verify."""

import time
from typing import Callable, Protocol


class LoadBalancer(Protocol):
    """Minimal surface a rollout script needs from a load balancer client."""

    def remove(self, replica_id: str) -> None: ...
    def request_drain(self, replica_id: str) -> None: ...
    def is_drained(self, replica_id: str) -> bool: ...
    def add(self, replica_id: str) -> None: ...


def drain_replica(
    replica_id: str,
    load_balancer: LoadBalancer,
    max_wait_s: float = 60.0,
) -> None:
    """Take one replica out of rotation, drain it, wait for it to finish."""
    load_balancer.remove(replica_id)  # <1>
    load_balancer.request_drain(replica_id)  # <2>
    deadline = time.monotonic() + max_wait_s
    while not load_balancer.is_drained(replica_id) and time.monotonic() < deadline:
        time.sleep(1.0)
    if not load_balancer.is_drained(replica_id):
        raise TimeoutError(f"{replica_id} did not drain within {max_wait_s}s")


def rolling_deploy(
    replica_ids: list[str],
    load_balancer: LoadBalancer,
    deploy_one: Callable[[str], None],
) -> None:
    """Replace replicas ONE AT A TIME - never more than one out of rotation."""
    for replica_id in replica_ids:
        drain_replica(replica_id, load_balancer)
        deploy_one(replica_id)  # <3>
        load_balancer.add(replica_id)
