"""Chapter 12, "Context Engineering" - the context budget.

See "Enforcing the budget". Atlas's context window is now a resource it
manages on purpose, not a buffer that fills until it overflows. `Budget` is
a per-turn allocation of the model's context window. `trim_history` is the
one trim every bounded model call shares: the most recent turns that fit the
history slice, starting on a human turn. It returns a new list and never
touches the durable history in state or the checkpoint.

Two callers apply it. `atlas.graph`'s `triage` node hands its classifier
`trim_history(state["messages"], BUDGET.history)`, so the graph's own model
call is bounded. `ContextBudget` is the `AgentMiddleware` that applies it to
every call `resolve_agent` (atlas/agent.py) makes, via `wrap_model_call`,
rewriting the *view* the model sees this turn (that distinction is the seam:
`wrap_model_call` rewrites the request, `before_model` would mutate
persisted state).

`select_docs` is the hot-path half of *select*: rank retrieved documents by
score and cap them to the retrieved slice, instead of handing the model
every document the search returned. `atlas.graph`'s `retrieve` node calls it
on the search results with `BUDGET.retrieved`, before `route_after_retrieve`
reads the list, so an article larger than the whole slice leaves the list
empty and takes Chapter 6's path (retry, then escalate) instead of reaching
`answer`. The richer, long-term-memory half of select is Chapters 13-14.

`BUDGET` is Atlas's one allocation (history=4000, retrieved=2000), shared by
`triage`, by `ContextBudget` on `resolve_agent`, and by `retrieve`, so the
slices cannot drift apart.

`ContextBudget`'s trim lives in `_view`, shared by `wrap_model_call` and its
async twin `awrap_model_call` (Chapter 8 states the rule for
`wrap_tool_call`; Chapter 12 extends it to `wrap_model_call`: under
`ainvoke`/`astream` the model node calls the async hook, and a middleware
with only the sync one raises `NotImplementedError` there).

The trim passes `include_system=True`, which keeps a SystemMessage at index 0
of the list it is given. The agent's own system prompt is never in that list:
`ModelRequest.messages` excludes it (`create_agent` carries it as
`request.system_message` and prepends it only when it calls the model), so
the prompt is never trimmed and never counts against `history`. What the flag
keeps is a system message the *caller* put first in the conversation:
Chapter 17's mount adapter (`atlas/resolve.py`) opens the mounted agent's
input with one built from the capped documents and the customer profile, and
without the flag `start_on="human"` would drop it. With no leading system
message - `triage`, and any plain conversation - the trim is unchanged.

`atlas/state.py`'s `Doc` has carried `score: float` since Chapter 5 (set
by the retriever); `select_docs` is its first reader."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from atlas.state import Doc


@dataclass
class Budget:
    """A per-turn allocation of the model's context window, in tokens.
    System prompt, scratch, and headroom take the remainder."""

    history: int  # reserved for conversation history
    retrieved: int  # reserved for retrieved documents


BUDGET = Budget(history=4000, retrieved=2000)  # Atlas's allocation


def trim_history(messages: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """The most recent turns that fit the history slice, as a new list.
    The one trim every bounded model call shares."""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",  # keep the most recent turns
        start_on="human",
        include_system=True,  # a leading SystemMessage survives the cut
    )


class ContextBudget(AgentMiddleware):
    """Enforce the budget on every model call - without deleting anything
    from persisted state."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget

    def _view(self, request: ModelRequest) -> ModelRequest:
        trimmed = trim_history(request.messages, self.budget.history)
        return request.override(messages=trimmed)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._view(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._view(request))


def select_docs(docs: list[Doc], max_tokens: int) -> list[Doc]:
    """Take the highest-scored docs that fit the retrieved slice.
    Include the best few that fit, not everything fetched."""
    kept, used = [], 0
    for doc in sorted(docs, key=lambda d: d["score"], reverse=True):
        cost = count_tokens_approximately([HumanMessage(doc["text"])])
        if used + cost > max_tokens:
            continue
        kept.append(doc)
        used += cost
    return kept
