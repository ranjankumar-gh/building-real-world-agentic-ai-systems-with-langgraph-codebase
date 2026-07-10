"""Chapter 22, "Deployment and Scaling" - atlas/deploy/rollout.py.

See "The drain-migrate-deploy script". `drain_replica`/`rolling_deploy` are
orchestration logic over a `load_balancer` interface (`remove`/`add`/
`request_drain`/`is_drained`), not a live fleet - there is no real load
balancer or Agent Server in this repo to front, so these tests exercise the
logic itself against a small fake that implements exactly that interface:
remove-before-drain, one replica at a time, and a `TimeoutError` when a
drain never finishes."""

import pytest

from atlas.deploy.rollout import drain_replica, rolling_deploy


class FakeLoadBalancer:
    """Records calls and lets a test control when a replica reports drained."""

    def __init__(self, drains_immediately: bool = True) -> None:
        self.removed: list[str] = []
        self.added: list[str] = []
        self.drain_requested: list[str] = []
        self._drains_immediately = drains_immediately
        self._drained: set[str] = set()

    def remove(self, replica_id: str) -> None:
        self.removed.append(replica_id)

    def add(self, replica_id: str) -> None:
        self.added.append(replica_id)

    def request_drain(self, replica_id: str) -> None:
        self.drain_requested.append(replica_id)
        if self._drains_immediately:
            self._drained.add(replica_id)

    def is_drained(self, replica_id: str) -> bool:
        return replica_id in self._drained


def test_drain_replica_removes_before_requesting_drain():
    """Removing from rotation before draining is what stops new work from
    landing on a replica that's about to disappear."""
    lb = FakeLoadBalancer()

    drain_replica("r1", lb)

    assert lb.removed == ["r1"]
    assert lb.drain_requested == ["r1"]


def test_drain_replica_returns_once_is_drained_reports_true():
    lb = FakeLoadBalancer(drains_immediately=True)

    drain_replica("r1", lb, max_wait_s=5.0)  # would hang were it not drained

    assert lb.is_drained("r1") is True


def test_drain_replica_times_out_if_never_drained(monkeypatch):
    """A replica stuck mid-run past max_wait_s raises rather than deploying
    on top of work that never finished."""
    lb = FakeLoadBalancer(drains_immediately=False)
    monkeypatch.setattr("time.sleep", lambda _: None)  # no real waiting in tests

    with pytest.raises(TimeoutError, match="r1"):
        drain_replica("r1", lb, max_wait_s=0.0)


def test_rolling_deploy_replaces_replicas_one_at_a_time_in_order():
    lb = FakeLoadBalancer(drains_immediately=True)
    deployed: list[str] = []

    rolling_deploy(["r1", "r2", "r3"], lb, deploy_one=deployed.append)

    assert deployed == ["r1", "r2", "r3"]
    assert lb.removed == ["r1", "r2", "r3"]
    assert lb.added == ["r1", "r2", "r3"]


def test_rolling_deploy_only_deploys_after_the_prior_replica_is_back_in_rotation():
    """Never more than one replica out of rotation: deploy_one for r2 must
    not run until r1 has already been re-added."""
    lb = FakeLoadBalancer(drains_immediately=True)
    order: list[str] = []

    def deploy_one(replica_id: str) -> None:
        order.append(f"deploy:{replica_id}")
        # r1 must already be back in rotation before r2's deploy starts.
        if replica_id == "r2":
            assert lb.added == ["r1"]

    rolling_deploy(["r1", "r2"], lb, deploy_one=deploy_one)

    assert order == ["deploy:r1", "deploy:r2"]
