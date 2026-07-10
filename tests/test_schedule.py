"""Chapter 22, "Deployment and Scaling" - atlas/deploy/schedule.py.

See "Cron and webhooks: making the online monitor real infrastructure".
`schedule_quality_monitor`/`notify_on_research_complete` both need a live
`langgraph up` Agent Server (real infrastructure, the same external-service
exception as Chapter 9's Postgres or Chapter 20's LangSmith - see
`requires_live_agent_server` below) to actually schedule anything, so the
default tests here monkeypatch `get_client` with a fake SDK client and check
only the call shape: which assistant, which schedule/input, which webhook -
not a live connection."""

import asyncio
import os

import pytest

from atlas.deploy import schedule as schedule_module
from atlas.deploy.schedule import (
    notify_on_research_complete,
    schedule_quality_monitor,
    schedule_sla_watch,
)

requires_live_agent_server = pytest.mark.skipif(
    not os.environ.get("ATLAS_AGENT_SERVER_URL"),
    reason="requires a live `langgraph up` Agent Server",
)


class _FakeCrons:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"cron_id": "cron-1"}


class _FakeRuns:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"run_id": "run-1"}


class _FakeClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.crons = _FakeCrons()
        self.runs = _FakeRuns()


def test_schedule_quality_monitor_targets_the_resolve_assistant_every_15_minutes(
    monkeypatch,
):
    fake = _FakeClient("http://localhost:8123")
    monkeypatch.setattr(schedule_module, "get_client", lambda url: fake)

    result = asyncio.run(schedule_quality_monitor())

    assert result == {"cron_id": "cron-1"}
    assert fake.crons.calls == [
        {
            "assistant_id": "resolve",
            "schedule": "*/15 * * * *",
            "input": {"sample_rate": 0.05},
        }
    ]


def test_schedule_quality_monitor_uses_the_given_url_sample_rate_and_schedule(
    monkeypatch,
):
    seen_urls = []
    fake = _FakeClient("http://example.internal:8123")
    monkeypatch.setattr(
        schedule_module,
        "get_client",
        lambda url: seen_urls.append(url) or fake,
    )

    asyncio.run(
        schedule_quality_monitor(
            url="http://example.internal:8123", schedule="0 * * * *", sample_rate=0.1
        )
    )

    assert seen_urls == ["http://example.internal:8123"]
    assert fake.crons.calls[0]["schedule"] == "0 * * * *"
    assert fake.crons.calls[0]["input"] == {"sample_rate": 0.1}


def test_notify_on_research_complete_targets_the_research_assistant_with_a_webhook(
    monkeypatch,
):
    fake = _FakeClient("http://localhost:8123")
    monkeypatch.setattr(schedule_module, "get_client", lambda url: fake)
    message = {"role": "user", "content": "Compare our SLA to three competitors'."}

    result = asyncio.run(
        notify_on_research_complete(
            thread_id="thread-1",
            message=message,
            webhook="https://internal.atlas.example.com/hooks/research-complete",
        )
    )

    assert result == {"run_id": "run-1"}
    assert fake.runs.calls == [
        {
            "thread_id": "thread-1",
            "assistant_id": "research",
            "input": {"messages": [message]},
            "webhook": "https://internal.atlas.example.com/hooks/research-complete",
        }
    ]


def test_schedule_sla_watch_targets_the_sla_watch_assistant_hourly(monkeypatch):
    fake = _FakeClient("http://localhost:8123")
    monkeypatch.setattr(schedule_module, "get_client", lambda url: fake)

    result = asyncio.run(schedule_sla_watch())

    assert result == {"cron_id": "cron-1"}
    assert fake.crons.calls == [
        {"assistant_id": "sla-watch", "schedule": "0 * * * *", "input": {}}
    ]


def test_schedule_sla_watch_uses_the_given_url_and_schedule(monkeypatch):
    seen_urls = []
    fake = _FakeClient("http://example.internal:8123")
    monkeypatch.setattr(
        schedule_module,
        "get_client",
        lambda url: seen_urls.append(url) or fake,
    )

    asyncio.run(
        schedule_sla_watch(url="http://example.internal:8123", schedule="0 */2 * * *")
    )

    assert seen_urls == ["http://example.internal:8123"]
    assert fake.crons.calls[0]["schedule"] == "0 */2 * * *"


# --- External-service exception: a live Agent Server -----------------------


@requires_live_agent_server
def test_schedule_quality_monitor_against_a_real_agent_server():
    """Skipped by default - see `requires_live_agent_server` above. Run
    `langgraph up --port 8123 --wait` (or point ATLAS_AGENT_SERVER_URL at a
    running one) to exercise this for real."""
    asyncio.run(schedule_quality_monitor(url=os.environ["ATLAS_AGENT_SERVER_URL"]))
