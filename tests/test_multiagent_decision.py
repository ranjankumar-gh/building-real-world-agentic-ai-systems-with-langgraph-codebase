"""Chapter 15, "When and Why to Go Multi-Agent" - atlas/multiagent_decision.py.

See "The decision: a framework and a cost model". `should_split` needs no
live model or graph - it is a pure predicate over `Task`'s four shape
flags. `single_agent_cost`/`multi_agent_cost` are pure arithmetic; the
sequential-support-task numbers in the chapter's prose (4 calls, 32K
tokens, 1.8s for the single agent) are reproduced directly as a test."""

from atlas.multiagent_decision import (
    CostEstimate,
    Task,
    multi_agent_cost,
    should_split,
    single_agent_cost,
)


def test_should_split_stays_single_by_default():
    """None of the three qualifying shapes present - stay single."""
    task = Task(
        parallel_independent=False,
        heterogeneous_tools=False,
        isolation_boundary=False,
        fits_one_context=False,
    )

    assert should_split(task) is False


def test_should_split_true_when_a_qualifying_shape_does_not_fit_one_context():
    task = Task(
        parallel_independent=True,
        heterogeneous_tools=False,
        isolation_boundary=False,
        fits_one_context=False,
    )

    assert should_split(task) is True


def test_should_split_false_when_qualifying_shape_still_fits_one_context():
    """The guard most teams skip: a task that fits cleanly in one context
    budget stays single even if it could technically be split."""
    task = Task(
        parallel_independent=True,
        heterogeneous_tools=True,
        isolation_boundary=True,
        fits_one_context=True,
    )

    assert should_split(task) is False


def test_should_split_true_for_each_qualifying_shape_alone():
    base = dict(parallel_independent=False, heterogeneous_tools=False, isolation_boundary=False)
    for shape in ("parallel_independent", "heterogeneous_tools", "isolation_boundary"):
        flags = dict(base, **{shape: True})
        task = Task(fits_one_context=False, **flags)
        assert should_split(task) is True, shape


def test_single_agent_cost_matches_the_chapters_worked_example():
    """Four steps, 8,000 tokens of context, 0.45s per call - the chapter's
    own sequential-support-task numbers: 4 calls, 32K tokens, 1.8s."""
    estimate = single_agent_cost(steps=4, ctx_tokens=8000, s_per_call=0.45)

    assert estimate == CostEstimate(calls=4, tokens=32000, latency_s=1.8)


def test_multi_agent_cost_pays_a_tax_over_the_single_agent_equivalent():
    """Same per-call cost and context size as the single-agent example,
    fanned across three agents - several times the tokens, well over
    double the latency, matching the chapter's "several times the tokens"
    and "easily double the latency" claims."""
    single = single_agent_cost(steps=4, ctx_tokens=8000, s_per_call=0.45)
    multi = multi_agent_cost(
        agents=3, steps_each=4, ctx_tokens=8000, s_per_call=0.45, handoff_tokens=500
    )

    assert multi.tokens > 3 * single.tokens
    assert multi.latency_s > 2 * single.latency_s
    assert multi.calls > single.calls


def test_multi_agent_cost_accounts_for_coordinator_routing_calls():
    """calls = agents * steps_each + agents (the "+ coordinator routing
    calls" the chapter's code comment calls out) - not just agents *
    steps_each."""
    estimate = multi_agent_cost(
        agents=3, steps_each=4, ctx_tokens=1000, s_per_call=0.1, handoff_tokens=0
    )

    assert estimate.calls == 3 * 4 + 3
