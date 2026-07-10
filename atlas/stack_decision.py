"""Chapter 25, "Choosing Your Stack (and When Not to Use LangGraph)".

See "The architecture decision record". `ProjectShape` captures the five
ADR questions as four flags (the fifth - what the team already operates -
is a non-technical factor the function deliberately does not encode).
`recommend` checks DURABILITY NEED first, then UNPREDICTABLE BRANCHING,
as a 2x2 rather than a sequential chain - an earlier draft checked
branching first and misrouted the durable-but-predictable batch-pipeline
case (Case 3) to a lighter-weight recommendation than it needed. The
order of the two checks is the whole point of this module."""

from dataclasses import dataclass


@dataclass
class ProjectShape:
    survives_restart: bool
    unpredictable_branching: bool
    crosses_checkpoint_membrane: bool = False
    genuinely_multi_agent: bool = False


def recommend(shape: ProjectShape) -> str:
    """The ADR's first two questions ARE the decision; 3 and 4 explain why,
    once you're already in LangGraph's territory."""
    if not shape.survives_restart:
        if not shape.unpredictable_branching:
            return "nothing needed, or a single-agent SDK"
        return "LangGraph without a checkpointer, or a lighter agent library"
    if not shape.unpredictable_branching:
        return "Temporal, or a plain workflow engine"
    return "LangGraph - the durability tax earns its keep"
