"""Chapter 22, "Deployment and Scaling" - cron and webhook wiring against a
live self-hosted Agent Server.

See "Cron and webhooks: making the online monitor real infrastructure".
`schedule_quality_monitor` converts Chapter 21's `run_quality_monitor` (an
ad-hoc script someone had to remember to run) into a stateless cron on the
Agent Server itself, targeting the `monitor` graph `atlas/deploy/server.py`
serves (input `{"sample_rate": float}`) - a fresh thread per trigger,
deleted when the run finishes unless `on_run_completed="keep"`, the right
shape for a monitor that scores an independent sample each run and has no
reason to remember the last one. The schedule is UTC unless `timezone=` is
set. `notify_on_research_complete` covers the opposite
direction for the Chapter 17 research fan-out: not "run this on a schedule"
but "tell me when a run I already started finishes," via a webhook instead
of a caller polling a long-running brief. The research graph's input is
`{"sources": [...]}` (Chapter 17's map-reduce reads `state["sources"]`), and
the run goes on a thread created first, so the caller can read the findings
back from it.

Under Chapter 23's `auth` entry, `atlas/auth.py` must allow crons
explicitly (its default deny refuses any resource without a handler); it
does, scoped to the owner the way threads are.

Both need a live `langgraph up` Agent Server reachable at `url` - there is
no seeded, mockable stand-in for the Agent Server itself in this repo (it is
real infrastructure, the same category as Chapter 20's LangSmith or Chapter
9's Postgres), so `tests/test_schedule.py` only exercises argument shape
(what gets passed to `langgraph_sdk`) and skip-guards anything that would
actually dial a server.

Chapter 27, "Capstone", adds `schedule_sla_watch` - the same stateless-cron
shape as `schedule_quality_monitor`, targeting the `sla-watch` assistant
hourly instead of `monitor` every 15 minutes."""

from langgraph_sdk import get_client


async def schedule_quality_monitor(
    url: str = "http://localhost:8123",
    schedule: str = "*/15 * * * *",  # every 15 minutes, UTC
    sample_rate: float = 0.05,
) -> dict:
    """Turn Chapter 21's online quality monitor into a stateless cron."""
    client = get_client(url=url)
    return await client.crons.create(  # <1>
        assistant_id="monitor",
        schedule=schedule,
        input={"sample_rate": sample_rate},
    )


async def schedule_sla_watch(
    url: str = "http://localhost:8123",
    schedule: str = "0 * * * *",  # hourly
) -> dict:
    """Chapter 27, "Capstone": SLA Watch's own stateless cron - a fresh
    thread per hourly trigger, the same shape as
    `schedule_quality_monitor` above (each triggered run is independent of
    the last one, mirroring Chapter 21's online monitor). "Stateless"
    describes the cron's own scheduling, not the run it triggers: any GIVEN
    triggered run still gets a normal `thread_id` and Chapter 9's full
    checkpointer, and can sit suspended at the approval gate for as long as
    a reviewer takes."""
    client = get_client(url=url)
    return await client.crons.create(
        assistant_id="sla-watch",
        schedule=schedule,
        input={},
    )


async def notify_on_research_complete(
    sources: list[str],
    webhook: str,
    url: str = "http://localhost:8123",
) -> dict:
    """Fire a webhook when a long-running research run finishes, instead of
    making the caller poll for it."""
    client = get_client(url=url)
    thread = await client.threads.create()
    return await client.runs.create(
        thread["thread_id"],
        assistant_id="research",
        input={"sources": sources},  # what Chapter 17's fan-out reads
        webhook=webhook,  # called once the run finishes, whatever it produced
    )
