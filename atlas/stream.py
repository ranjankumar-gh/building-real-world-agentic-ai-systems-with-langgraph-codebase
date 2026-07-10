"""Chapter 19, "Streaming" - multiplex Atlas's tokens, step status, and
Chapter 17/18 fan-out progress into one tagged stream.

See "Building the multiplexed stream". `stream_atlas` wraps `atlas/graph.py`'s
compiled `graph` (the Chapter 9 checkpointer-backed default) with
`graph.stream(..., stream_mode=["updates", "messages", "custom"],
subgraphs=True, version="v2")` - the production default from "The stream_mode
taxonomy": step status, answer tokens, and tool-reported progress
(`atlas/deep_research.py`'s `source_lookup`, via `get_stream_writer()`),
multiplexed onto one iterator.

`version="v2"` is passed explicitly on purpose - `graph.stream()`'s own
default is still the older `v1`, which yields a raw chunk for a single
`stream_mode` but `(mode, chunk)` tuples for a list of modes (and changes
shape again with `subgraphs=True`). Under `v2`, every combination yields the
same `{"type": mode, "ns": namespace_tuple, "data": payload}` dict - see
"stream() has its own quiet version split - and the default is the awkward
one".

`chunk["ns"]` is `()` for the main graph and a non-empty namespace tuple for
anything inside Chapter 17's `research` subgraph or a Chapter 18 sub-agent.
Normalizing the empty case to `("main",)` here means every downstream
consumer branches on `event["source"]` the same way regardless of where the
event came from - it never has to special-case the root graph.
"""

from typing import Iterator

from atlas.graph import graph


def stream_atlas(inputs: dict, config: dict) -> Iterator[dict]:
    """Multiplex step status, tokens, and tool progress into one tagged stream."""
    for chunk in graph.stream(
        inputs,
        config,
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True,
        version="v2",
    ):
        event = {
            "kind": chunk["type"],  # "updates" | "messages" | "custom"
            "source": chunk["ns"] or ("main",),
            "data": chunk["data"],
        }
        yield event
