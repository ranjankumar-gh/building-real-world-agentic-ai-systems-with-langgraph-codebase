"""Chapter 22, "Deployment and Scaling" - the drain-and-deploy script.

See "The drain-and-deploy script". The Agent Server drains itself: on
SIGTERM it stops taking work from the shared run queue and lets in-flight
runs finish for up to `BG_JOB_SHUTDOWN_GRACE_PERIOD_SECS` (180 s in the
server version this chapter checked, langgraph-api 0.14.0). A run still
going when the grace period ends is cancelled and set back to pending for
another replica, up to `BG_JOB_MAX_RETRIES` (3). What the server cannot
survive is a hard kill before the grace period is over - Docker's default
stop timeout is 10 s - so the orchestrator's stop timeout (`stop_grace_period`
in Compose, `terminationGracePeriodSeconds` in Kubernetes) must be longer
than the server's grace. `docker-compose.override.yml` sets both.

Chapter 10's `RunControl.request_drain()` is the same idea for a runner you
own: one `RunControl` per run, passed as `ainvoke(..., control=...)`. The
Agent Server passes none and needs none, so this script never calls it.

What the script adds is the order: take one replica out of the load
balancer (so no new HTTP request lands on it), send SIGTERM and wait for it
to stop, replace it, wait for the new one to be ready, put it back. One at a
time, so the schema straddle stays one replica wide.

`fleet` is Atlas's own thin wrapper over the load balancer and the
orchestrator (not a LangGraph API, and not implemented in this repo - the
companion repo has no real fleet to front). `tests/test_rollout.py`
exercises the ordering against a small fake that implements exactly that
interface."""

import time
from typing import Callable, Protocol

SERVER_GRACE_S = 180.0  # BG_JOB_SHUTDOWN_GRACE_PERIOD_SECS, server default


class Fleet(Protocol):
    """What a rollout script needs from the load balancer and orchestrator."""

    def remove(self, replica_id: str) -> None: ...
    def terminate(self, replica_id: str) -> None: ...
    def is_stopped(self, replica_id: str) -> bool: ...
    def is_ready(self, replica_id: str) -> bool: ...
    def add(self, replica_id: str) -> None: ...


def wait_for(check: Callable[[], bool], max_wait_s: float, what: str) -> None:
    deadline = time.monotonic() + max_wait_s
    while not check() and time.monotonic() < deadline:
        time.sleep(1.0)
    if not check():
        raise TimeoutError(f"{what} within {max_wait_s}s")


def drain_replica(
    replica_id: str, fleet: Fleet, max_wait_s: float = SERVER_GRACE_S + 20
) -> None:
    """Take one replica out of rotation and let the server drain it."""
    fleet.remove(replica_id)  # <1>
    fleet.terminate(replica_id)  # <2>
    wait_for(lambda: fleet.is_stopped(replica_id), max_wait_s,
             f"{replica_id} did not stop")


def rolling_deploy(
    replica_ids: list[str],
    fleet: Fleet,
    deploy_one: Callable[[str], None],
    ready_wait_s: float = 120.0,
) -> None:
    """Replace replicas ONE AT A TIME - never more than one out of rotation."""
    for replica_id in replica_ids:
        drain_replica(replica_id, fleet)
        deploy_one(replica_id)  # <3>
        wait_for(lambda: fleet.is_ready(replica_id), ready_wait_s,
                 f"{replica_id} was not ready")
        fleet.add(replica_id)


# 1. Out of the load balancer first, so no new HTTP request lands on a
#    replica about to stop. It does not stop background runs: every replica
#    also runs queue workers that pull from the shared queue. SIGTERM does.
# 2. SIGTERM, not a kill. The server stops taking queue work and finishes
#    what it has for up to its grace period; a run cut off at the end of it
#    goes back on the queue for another replica.
# 3. `deploy_one` is the container replacement: pull the new image, start
#    it. The new replica rejoins only after a readiness check passes.
