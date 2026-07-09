"""Chapter 2: atlas/hello.py - the create_agent baseline, no hand-rolled loop."""

from atlas import hello


def test_kb_lookup_answers_a_matching_query():
    result = hello.kb_lookup.invoke({"query": "What's the refund window?"})

    assert result == "Refunds are available within 30 days of purchase."


def test_kb_lookup_reports_no_match_for_an_unrelated_query():
    result = hello.kb_lookup.invoke({"query": "Where is my order?"})

    assert result == "No knowledge-base article matched."


def test_agent_compiles_to_an_invokable_graph_without_calling_the_model():
    # Constructing the agent must not require a live API key - it is only
    # needed if the graph is actually invoked, which this test deliberately
    # avoids (no external accounts, per the book's own local-only promise).
    assert hasattr(hello.agent, "invoke")
