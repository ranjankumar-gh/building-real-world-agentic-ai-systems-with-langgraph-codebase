"""Chapter 22, "Deployment and Scaling" - atlas/deploy/rollout.py.

See "The drain-migrate-deploy script". `drain_replica`/`rolling_deploy` are
orchestration logic over a `fleet` interface (`remove`/`terminate`/
`is_stopped`/`is_ready`/`add`), not a live fleet - there is no real load
balancer or orchestrator in this repo to drive, so these tests exercise the
ordering itself against a small fake: out of rotation before SIGTERM, one
replica at a time, back in only once ready, and a `TimeoutError` when a
replica never stops or never comes up."""

import pytest

from atlas.deploy.rollout import SERVER_GRACE_S, drain_replica, rolling_deploy


class FakeFleet:
    """Records calls in order; a test controls when replicas stop and start."""

    def __init__(self, stops: bool = True, ready: bool = True) -> None:
        self.events: list[tuple[str, str]] = []
        self._stops = stops
        self._ready = ready
        self._stopped: set[str] = set()

    def remove(self, replica_id: str) -> None:
        self.events.append(("remove", replica_id))

    def terminate(self, replica_id: str) -> None:
        self.events.append(("terminate", replica_id))
        if self._stops:
            self._stopped.add(replica_id)

    def is_stopped(self, replica_id: str) -> bool:
        return replica_id in self._stopped

    def is_ready(self, replica_id: str) -> bool:
        return self._ready

    def add(self, replica_id: str) -> None:
        self.events.append(("add", replica_id))


def test_drain_replica_leaves_the_load_balancer_before_sigterm():
    """Out of rotation first, so no new HTTP request lands on a replica the
    server is about to drain."""
    fleet = FakeFleet()

    drain_replica("r1", fleet)

    assert fleet.events == [("remove", "r1"), ("terminate", "r1")]


def test_the_default_wait_outlasts_the_servers_grace_period():
    import inspect

    default = inspect.signature(drain_replica).parameters["max_wait_s"].default
    assert default > SERVER_GRACE_S == 180.0


def test_drain_replica_times_out_if_the_replica_never_stops(monkeypatch):
    """A replica still running past the wait raises rather than deploying on
    top of work that never finished."""
    fleet = FakeFleet(stops=False)
    monkeypatch.setattr("time.sleep", lambda _: None)  # no real waiting in tests

    with pytest.raises(TimeoutError, match="r1 did not stop"):
        drain_replica("r1", fleet, max_wait_s=0.0)


def test_rolling_deploy_replaces_replicas_one_at_a_time_in_order():
    fleet = FakeFleet()
    deployed: list[str] = []

    def deploy_one(replica_id: str) -> None:
        deployed.append(replica_id)
        fleet.events.append(("deploy", replica_id))

    rolling_deploy(["r1", "r2", "r3"], fleet, deploy_one=deploy_one)

    assert deployed == ["r1", "r2", "r3"]
    per_replica = ["remove", "terminate", "deploy", "add"]
    assert fleet.events == [(e, r) for r in ("r1", "r2", "r3") for e in per_replica]


def test_a_replica_that_never_becomes_ready_is_not_put_back(monkeypatch):
    fleet = FakeFleet(ready=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="r1 was not ready"):
        rolling_deploy(["r1"], fleet, deploy_one=lambda r: None, ready_wait_s=0.0)
    assert ("add", "r1") not in fleet.events
