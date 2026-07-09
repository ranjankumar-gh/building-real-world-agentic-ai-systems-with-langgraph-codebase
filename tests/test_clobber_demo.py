"""Chapter 5: atlas/clobber_demo.py - the loud, same-superstep clobber and
its one-line fix.

See "Reproducing and fixing the clobber". `bad_graph`'s `hits` channel has no
reducer, so the two same-superstep writers race and the runtime raises
`InvalidUpdateError` rather than silently drop one. `good_graph` annotates
the same channel with `operator.add`, so the two writes concatenate."""

import pytest
from langgraph.errors import InvalidUpdateError

from atlas.clobber_demo import bad_graph, good_graph


def test_bad_graph_raises_invalid_update_error_on_the_same_step_write():
    with pytest.raises(InvalidUpdateError):
        bad_graph.invoke({"hits": []})


def test_good_graph_merges_both_writes_via_operator_add():
    result = good_graph.invoke({"hits": []})

    assert sorted(result["hits"]) == sorted(["a1", "a2", "b1"])
    assert len(result["hits"]) == 3
