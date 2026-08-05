"""Chapter 20, "Observability" - the detection half.

See "What to page on: turning traces into detection". Chapters 19-21 gave
Atlas reconstruction: a trace explains any run you already know went wrong.
None of it tells you a run went wrong. `atlas/monitor.py` scores a sample and
writes the score back to LangSmith, where it becomes a trend line someone has
to remember to look at.

This module is the missing comparison. It holds the signals worth paging on,
the threshold for each, and a pure function that turns observed values into
breaches. Deliberately pure: deciding "is this bad" is separable from
gathering the numbers and from delivering the page, and only the decision
needs to be pinned by a test.

The thresholds below are the ones Chapter 1 said the team would set, written
down at last. They are Atlas's, not universal - a support assistant that
escalates 25% of tickets may be healthy where one that escalates 25% of
refunds is not. Pick yours from what the product promises, then let an
alert defend that promise.
"""

from dataclasses import dataclass
from typing import Literal

Direction = Literal["above", "below"]


@dataclass(frozen=True)
class Signal:
    """One thing worth waking someone up for.

    `source` names where the number comes from, because a threshold nobody
    can compute is decoration. Every source below is something Atlas already
    records by Chapter 21 - no new instrumentation, only the comparison
    nobody had written.
    """

    name: str
    threshold: float
    direction: Direction
    source: str
    why: str


@dataclass(frozen=True)
class Breach:
    signal: Signal
    observed: float

    def __str__(self) -> str:
        arrow = ">" if self.signal.direction == "above" else "<"
        return (
            f"{self.signal.name}: {self.observed:g} {arrow} {self.signal.threshold:g}"
            f" ({self.signal.why})"
        )


# Chapter 1 set these as Atlas's success criteria and then nothing measured
# them for twenty-six chapters. This is where they stop being aspirations.
SIGNALS: tuple[Signal, ...] = (
    Signal(
        name="resolution_rate",
        threshold=0.80,
        direction="below",
        source="share of runs ending at `answer` rather than `escalate`, "
        "from the routing decision already on every trace (Ch20)",
        why="Chapter 1's headline promise; a slide here is the agent quietly "
        "handing work back to humans",
    ),
    Signal(
        name="tool_error_rate",
        threshold=0.02,
        direction="above",
        source="share of runs with a non-null `error` channel, written by "
        "`retrieve` when the KB is unavailable (Ch6)",
        why="the failure Atlas is designed to survive, and the one that turns "
        "into escalations if it persists",
    ),
    Signal(
        name="p95_time_to_first_token_s",
        threshold=2.0,
        direction="above",
        source="first `messages` chunk on the stream, per run (Ch19)",
        why="the only latency the customer actually feels; total runtime can "
        "double without anyone noticing if the first token is fast",
    ),
    Signal(
        name="cost_per_resolution_usd",
        threshold=0.05,
        direction="above",
        source="tenant token spend (Ch23) divided by resolutions in the window",
        why="the number that decides whether the agent is cheaper than the "
        "human it replaced - the one an invoice reveals too late",
    ),
    Signal(
        name="oldest_pending_interrupt_h",
        threshold=24.0,
        direction="above",
        source="age of the oldest thread suspended at an approval gate (Ch11)",
        why="suspended runs are the one kind of state nothing reclaims; a "
        "growing approval backlog is invisible until a customer asks",
    ),
    Signal(
        name="online_quality_score",
        threshold=0.85,
        direction="below",
        source="`atlas/monitor.py`'s sampled judge score (Ch21)",
        why="catches the silent failure mode: confident, fluent, wrong - which "
        "no exception handler will ever fire on",
    ),
)


def breaches(observed: dict[str, float]) -> list[Breach]:
    """Every signal the observed window violates.

    A signal missing from `observed` is skipped rather than treated as
    healthy: not measuring something is a different state from measuring it
    and finding it fine, and silently reading the first as the second is how
    a dashboard ends up green through an outage.
    """
    found: list[Breach] = []
    for signal in SIGNALS:
        if signal.name not in observed:
            continue
        value = observed[signal.name]
        bad = (
            value > signal.threshold
            if signal.direction == "above"
            else value < signal.threshold
        )
        if bad:
            found.append(Breach(signal=signal, observed=value))
    return found


def unmeasured(observed: dict[str, float]) -> list[str]:
    """Signals with no observed value this window.

    Worth reporting alongside the breaches: a signal that stops arriving is
    usually a broken collector, and a broken collector looks exactly like
    good news.
    """
    return [s.name for s in SIGNALS if s.name not in observed]
