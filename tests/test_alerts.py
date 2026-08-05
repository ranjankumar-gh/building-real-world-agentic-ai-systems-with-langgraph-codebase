"""Chapter 20, "Observability" - atlas/alerts.py.

See "What to page on: turning traces into detection". `breaches` is a pure
function over observed values, so the decision it makes is testable without a
metrics backend, a scheduler, or a live LangSmith project - which is the
reason the deciding was separated from the gathering in the first place.
"""

from atlas.alerts import SIGNALS, Breach, breaches, unmeasured


def _healthy() -> dict[str, float]:
    """A window where every signal sits on the good side of its threshold."""
    return {
        "resolution_rate": 0.91,
        "tool_error_rate": 0.004,
        "p95_time_to_first_token_s": 1.1,
        "cost_per_resolution_usd": 0.021,
        "oldest_pending_interrupt_h": 3.0,
        "online_quality_score": 0.93,
    }


def test_a_healthy_window_pages_nobody():
    assert breaches(_healthy()) == []


def test_every_signal_can_actually_fire():
    """A threshold nobody can cross is decoration. Each signal is pushed to
    its bad side one at a time, so no signal is silently unreachable."""
    for signal in SIGNALS:
        observed = _healthy()
        observed[signal.name] = (
            signal.threshold * 2 if signal.direction == "above" else signal.threshold / 2
        )

        fired = breaches(observed)

        assert [b.signal.name for b in fired] == [signal.name]


def test_resolution_slipping_below_chapter_ones_promise_fires():
    """Chapter 1 promised at least 80% resolved without a human handoff, then
    nothing measured it for twenty-six chapters."""
    fired = breaches({**_healthy(), "resolution_rate": 0.72})

    assert len(fired) == 1
    assert fired[0].signal.name == "resolution_rate"
    assert fired[0].observed == 0.72


def test_a_value_exactly_on_the_threshold_does_not_fire():
    """Thresholds are inclusive of the healthy side. A signal that pages at
    exactly its target trains people to ignore it."""
    for signal in SIGNALS:
        observed = {**_healthy(), signal.name: signal.threshold}

        assert breaches(observed) == []


def test_several_signals_can_breach_at_once():
    fired = breaches(
        {**_healthy(), "resolution_rate": 0.5, "tool_error_rate": 0.4}
    )

    assert {b.signal.name for b in fired} == {"resolution_rate", "tool_error_rate"}


def test_a_missing_signal_is_not_treated_as_healthy():
    """Not measuring something is a different state from measuring it and
    finding it fine. Reading the first as the second is how a dashboard stays
    green through an outage."""
    assert breaches({}) == []
    assert set(unmeasured({})) == {s.name for s in SIGNALS}


def test_unmeasured_reports_only_what_is_actually_absent():
    assert unmeasured(_healthy()) == []
    assert unmeasured({"resolution_rate": 0.9}) == [
        s.name for s in SIGNALS if s.name != "resolution_rate"
    ]


def test_a_breach_renders_the_number_the_threshold_and_the_reason():
    """The page has to say why it woke someone up, not just that it did."""
    fired = breaches({**_healthy(), "resolution_rate": 0.5})

    rendered = str(fired[0])
    assert "resolution_rate" in rendered
    assert "0.5" in rendered and "0.8" in rendered
    assert "Chapter 1" in rendered


def test_every_signal_names_where_its_number_comes_from():
    """A threshold with no source is unactionable - the on-call engineer has
    to be able to go compute it."""
    for signal in SIGNALS:
        assert signal.source.strip()
        assert signal.why.strip()
