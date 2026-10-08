"""Chapter 25, "Choosing Your Stack (and When Not to Use LangGraph)" -
atlas/stack_decision.py.

See "The architecture decision record". The three worked cases from the
chapter (stateless FAQ bot, Atlas-shaped durable agent, LLM-assisted
batch pipeline) are reproduced directly as tests, plus the two
"deserves its own branch" edge cases the 2x2 shape exists to get right:
a project needing durability without unpredictable branching (Case 3 -
the one a chain that answers after the branching check alone misroutes),
and a project needing unpredictable branching without durability."""

from atlas.stack_decision import ProjectShape, recommend


def test_case_1_stateless_faq_bot_needs_nothing_durable():
    shape = ProjectShape(survives_restart=False, unpredictable_branching=False)

    assert recommend(shape) == "nothing needed, or a single-agent SDK"


def test_case_2_atlas_shaped_durable_agent_earns_the_durability_tax():
    shape = ProjectShape(
        survives_restart=True,
        unpredictable_branching=True,
        crosses_checkpoint_membrane=True,
        genuinely_multi_agent=True,
    )

    assert recommend(shape) == "LangGraph - the durability tax earns its keep"


def test_case_3_batch_pipeline_is_durable_but_not_unpredictable():
    """The tell a chain that answers after one question gets wrong:
    durability needed, branching predictable, is a different problem
    (durable execution of known steps) than the FAQ bot's "needs
    nothing" case - not a smaller version of it."""
    shape = ProjectShape(survives_restart=True, unpredictable_branching=False)

    assert recommend(shape) == "Temporal, or a plain workflow engine"


def test_unpredictable_branching_without_durability_stays_off_the_full_stack():
    """Branches on model output but nothing outlives the request - a
    lighter LangGraph usage or a lighter library, not the full checkpointed
    stack Case 2 needs."""
    shape = ProjectShape(survives_restart=False, unpredictable_branching=True)

    assert recommend(shape) == "LangGraph without a checkpointer, or a lighter agent library"


def test_recommend_asks_both_questions_before_answering():
    """The misroute the 2x2 prevents: a chain that returns after the
    branching question alone. Which question is asked first does not
    matter; answering before both are asked does. Two shapes that disagree
    only on unpredictable_branching, both needing durability, must NOT
    fall into the "nothing needed" bucket that early return would give the
    non-branching one."""
    durable_predictable = ProjectShape(survives_restart=True, unpredictable_branching=False)
    durable_unpredictable = ProjectShape(survives_restart=True, unpredictable_branching=True)

    assert recommend(durable_predictable) != "nothing needed, or a single-agent SDK"
    assert recommend(durable_unpredictable) != "nothing needed, or a single-agent SDK"
    assert recommend(durable_predictable) != recommend(durable_unpredictable)


def test_project_shape_defaults_for_the_two_refining_questions():
    """crosses_checkpoint_membrane and genuinely_multi_agent default to
    False - the ADR's questions 3 and 4 refine *why*, they don't change
    *whether*, so the two worked cases that omit them (Cases 1 and 3)
    should construct cleanly."""
    shape = ProjectShape(survives_restart=False, unpredictable_branching=False)

    assert shape.crosses_checkpoint_membrane is False
    assert shape.genuinely_multi_agent is False
