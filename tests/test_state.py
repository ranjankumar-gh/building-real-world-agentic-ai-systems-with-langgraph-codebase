"""Chapter 5: atlas/state.py - the fully reducer-annotated AtlasState and its
custom `dedup_by_id` reducer."""

from typing import Annotated, get_args, get_origin

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from atlas.state import AtlasState, Doc, dedup_by_id


def _doc(doc_id: str, text: str = "") -> Doc:
    return {"id": doc_id, "text": text, "score": 0.0}


def test_dedup_by_id_drops_duplicate_ids_and_preserves_order():
    current = [_doc("1", "a"), _doc("2", "b")]
    update = [_doc("2", "b-dup"), _doc("3", "c")]

    merged = dedup_by_id(current, update)

    assert [doc["id"] for doc in merged] == ["1", "2", "3"]
    # the existing copy wins; the duplicate from `update` is dropped, not merged
    assert merged[1]["text"] == "b"


def test_dedup_by_id_yields_same_set_for_disjoint_inputs_in_either_order():
    current = [_doc("1"), _doc("2")]

    merged_ab = dedup_by_id(current, [_doc("3"), _doc("4")])
    merged_ba = dedup_by_id(current, [_doc("4"), _doc("3")])

    assert {doc["id"] for doc in merged_ab} == {doc["id"] for doc in merged_ba}
    assert len(merged_ab) == len(merged_ba) == 4


def test_dedup_by_id_on_empty_current_just_returns_the_update_deduped():
    update = [_doc("1"), _doc("1"), _doc("2")]

    merged = dedup_by_id([], update)

    assert [doc["id"] for doc in merged] == ["1", "2"]


def test_atlas_state_messages_channel_uses_add_messages():
    annotation = AtlasState.__annotations__["messages"]

    assert get_origin(annotation) is Annotated
    assert get_args(annotation)[1] is add_messages


def test_atlas_state_retrieved_channel_uses_dedup_by_id():
    annotation = AtlasState.__annotations__["retrieved"]

    assert get_origin(annotation) is Annotated
    assert get_args(annotation)[1] is dedup_by_id


def test_atlas_state_ticket_and_route_have_no_reducer_annotation():
    # LastValue channels: plain types, no Annotated reducer metadata.
    assert get_origin(AtlasState.__annotations__["ticket"]) is not Annotated
    assert get_origin(AtlasState.__annotations__["route"]) is not Annotated


def test_atlas_state_retrieve_attempts_and_error_have_no_reducer_annotation():
    """Chapter 6's loop-guard counter and recovery flag are also LastValue -
    only atlas/graph.py's retrieve node ever writes either one."""
    assert get_origin(AtlasState.__annotations__["retrieve_attempts"]) is not Annotated
    assert get_origin(AtlasState.__annotations__["error"]) is not Annotated


def test_atlas_state_refund_done_is_additive_and_has_no_reducer_annotation():
    """Chapter 10's new field: a plain LastValue bool, added additively so a
    checkpoint written before this chapter (which has no `refund_done` key
    at all) still resumes - see atlas/graph.py's `refund_already_done`."""
    assert "refund_done" in AtlasState.__annotations__
    assert get_origin(AtlasState.__annotations__["refund_done"]) is not Annotated


def test_retrieved_channel_merges_parallel_fan_out_without_duplicates():
    """Reproduces the hook's incident directly on AtlasState: three sources
    fan out from START in the same superstep and write `retrieved`. Because
    the channel carries `dedup_by_id`, the same-step write does not raise
    `InvalidUpdateError` - it merges, with the overlapping doc kept once."""

    def source_a(state: AtlasState) -> dict:
        return {"retrieved": [_doc("1", "product-docs")]}

    def source_b(state: AtlasState) -> dict:
        return {"retrieved": [_doc("1", "policy-docs-dup"), _doc("2", "policy-docs")]}

    def source_c(state: AtlasState) -> dict:
        return {"retrieved": [_doc("3", "past-tickets")]}

    builder = StateGraph(AtlasState)
    builder.add_node("a", source_a)
    builder.add_node("b", source_b)
    builder.add_node("c", source_c)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge(START, "c")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    builder.add_edge("c", END)
    graph = builder.compile()

    result = graph.invoke(
        {"messages": [], "retrieved": [], "ticket": None, "route": ""}
    )

    assert {doc["id"] for doc in result["retrieved"]} == {"1", "2", "3"}
    assert len(result["retrieved"]) == 3


def test_atlas_state_declares_the_chapter_11_approval_record_channel():
    """Chapter 11: the gate's audit record needs a declared channel - a
    write to an undeclared key is dropped (Chapter 6)."""
    assert "approval" in AtlasState.__annotations__
    assert get_origin(AtlasState.__annotations__["approval"]) is not Annotated
