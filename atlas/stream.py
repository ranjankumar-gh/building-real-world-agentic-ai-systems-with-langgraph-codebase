"""Chapter 19, "Streaming" - multiplex Atlas's tokens, step status, and
tool-reported progress into one tagged, redacted stream.

See "Building the multiplexed stream". `stream_atlas` wraps a compiled graph
(by default `atlas/graph.py`'s `graph`, Atlas's main support graph) with
`graph.stream(..., stream_mode=["updates", "messages", "custom"],
subgraphs=True, version="v2")` - the production default from "The stream_mode
taxonomy". Research and the Deep Research Agent are their own graphs
(Chapters 17-18); pass one as `graph=` and its workers and sub-agents arrive
on the same loop with non-empty namespaces.

`version="v2"` is passed explicitly on purpose - `graph.stream()`'s own
default is still the older `v1`, whose chunk shape changes with the
`stream_mode` argument. Under `v2`, every combination yields the same
`{"type": mode, "ns": namespace_tuple, "data": payload}` dict.

`chunk["ns"]` is `()` for the streamed graph's own nodes and the path of the
node that started a subgraph otherwise. Normalizing the empty case to
`("main",)` means every consumer branches on `event["source"]` the same way.

Redaction runs HERE, on the stream, because nothing upstream does it for the
wire: `PIIMiddleware(apply_to_output=True)` rewrites the stored message in
`after_model`, after the tokens have already streamed, and a transformer
registered at `compile(transformers=...)` runs only under
`stream_events(version="v3")`, never under `stream()`. Token deltas are held
back to the last whitespace before they are redacted, so an address the
tokenizer split across deltas is whole when the pattern sees it.

`stream_detached` is "A disconnect is not `interrupt()`": in process, the
loop that reads `graph.stream()` is the loop that drives the run, so closing
it stops the run at the next step. A run that must finish gets its own
worker, and the live channel reads a bounded buffer the worker fills.
"""

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.pregel import Pregel
from langgraph.types import Interrupt

from atlas.graph import graph as atlas_graph
from atlas.middleware import EMAIL_PATTERN

REDACTED = "[REDACTED_EMAIL]"  # PIIMiddleware's own token, so both sinks agree


def redact(value: Any) -> Any:
    """Mask every email address in an event payload, however deep it sits."""
    if isinstance(value, str):
        return EMAIL_PATTERN.sub(REDACTED, value)
    if isinstance(value, BaseMessage):
        update: dict[str, Any] = {"content": redact(value.content)}
        if isinstance(value, AIMessage):
            update["tool_calls"] = redact(value.tool_calls)
        return value.model_copy(update=update)
    if isinstance(value, Interrupt):
        return Interrupt(value=redact(value.value), id=value.id)
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def hold_back(held: dict[str, str], chunk: AIMessageChunk) -> AIMessageChunk:
    """Release a token stream only up to its last whitespace, so an address
    split across deltas is redacted whole. The model's final chunk
    (`chunk_position="last"`) releases the rest. Tool-call chunks pass
    through, redacted fragment by fragment; the whole call, redacted as one,
    arrives on `updates`."""
    key = chunk.id or ""
    text = held.pop(key, "") + chunk.text
    if chunk.chunk_position == "last":
        cut = len(text)
    else:
        cut = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t")) + 1
    if cut < len(text):
        held[key] = text[cut:]
    return AIMessageChunk(
        content=redact(text[:cut]),
        id=chunk.id,
        chunk_position=chunk.chunk_position,
        tool_call_chunks=redact(chunk.tool_call_chunks),
    )


def stream_atlas(
    inputs: dict, config: dict, graph: Pregel = atlas_graph
) -> Iterator[dict]:
    """Multiplex step status, tokens, and tool progress into one tagged,
    redacted stream."""
    held: dict[str, str] = {}  # per model call: text not yet safe to send
    for chunk in graph.stream(
        inputs,
        config,
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True,
        version="v2",
    ):
        data = chunk["data"]
        if chunk["type"] == "messages" and isinstance(data[0], AIMessageChunk):
            data = (hold_back(held, data[0]), redact(data[1]))
        else:
            data = redact(data)  # updates, custom, whole messages
        yield {
            "kind": chunk["type"],  # "updates" | "messages" | "custom"
            "source": chunk["ns"] or ("main",),
            "data": data,
        }


DONE = object()  # the worker's end-of-run marker


def offer(buffer: queue.Queue, item: object) -> None:
    """Put without ever blocking the worker: a full buffer drops its oldest
    event. The client that missed it catches up from the checkpoint."""
    while True:
        try:
            buffer.put_nowait(item)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                buffer.get_nowait()


def stream_detached(
    inputs: dict, config: dict, graph: Pregel = atlas_graph, maxsize: int = 256
) -> Iterator[dict]:
    """Run the graph in its own worker; the live channel reads a bounded
    buffer. Closing this iterator closes the live channel, not the run. A
    run that fails raises its error here, so the reader can tell failure
    from completion."""
    buffer: queue.Queue = queue.Queue(maxsize=maxsize)

    def work() -> None:
        end: object = DONE
        try:
            for event in stream_atlas(inputs, config, graph):
                offer(buffer, event)
        except Exception as exc:  # the run failed: say so, don't just stop
            end = exc
        finally:
            offer(buffer, end)  # one terminal item, so dropping can't lose it

    threading.Thread(target=work, daemon=True).start()
    while (event := buffer.get()) is not DONE:
        if isinstance(event, Exception):
            raise event
        yield event
