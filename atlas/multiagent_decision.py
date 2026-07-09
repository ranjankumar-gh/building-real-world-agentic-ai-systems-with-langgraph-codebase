"""Chapter 15, "When and Why to Go Multi-Agent" - the split decision.

See "The decision: a framework and a cost model". Before any orchestration
code exists, Atlas can answer two questions on paper: is a split even
*permitted* for this task's shape (`should_split`), and if so, does it
*pay* (`single_agent_cost` vs `multi_agent_cost`)? Both are deliberately
simple - the point is not precision, it is forcing the comparison before
the rebuild, while changing your mind is still free.

`Task.fits_one_context` is the guard most teams skip: a task that already
fits comfortably in one context budget (Chapter 12) does not need
splitting even if it *could* be split, because the single agent keeps the
whole context intact and skips the coordination tax entirely."""

from dataclasses import dataclass


@dataclass
class Task:
    parallel_independent: bool  # subtasks that can run at once, without each other's context
    heterogeneous_tools: bool  # subtasks needing genuinely different tools or models
    isolation_boundary: bool  # a hard safety/audit/compliance wall between subtasks
    fits_one_context: bool  # the whole job fits comfortably in one context budget


def should_split(task: Task) -> bool:
    """Stay single by default. A split is justified only when the work has a
    shape that benefits AND does not already fit cleanly in one agent."""
    has_qualifying_shape = (
        task.parallel_independent or task.heterogeneous_tools or task.isolation_boundary
    )
    return has_qualifying_shape and not task.fits_one_context


@dataclass
class CostEstimate:
    calls: int
    tokens: int
    latency_s: float


def single_agent_cost(steps: int, ctx_tokens: int, s_per_call: float) -> CostEstimate:
    """One agent: each step re-sends the shared context once."""
    return CostEstimate(steps, steps * ctx_tokens, steps * s_per_call)


def multi_agent_cost(
    agents: int,
    steps_each: int,
    ctx_tokens: int,
    s_per_call: float,
    handoff_tokens: int,
) -> CostEstimate:
    """N agents plus a coordinator: each agent re-establishes context, and
    every handoff duplicates a slice of it. Latency is serial across the
    coordinator's hops (parallel subtasks divide the step latency instead)."""
    calls = agents * steps_each + agents  # + coordinator routing calls
    tokens = calls * ctx_tokens + agents * handoff_tokens
    latency = (agents * steps_each + agents) * s_per_call
    return CostEstimate(calls, tokens, latency)
