"""Chapter 12, "Context Engineering" - the context budget.

See "Enforcing the budget". Atlas's context window is now a resource it
manages on purpose, not a buffer that fills until it overflows. `Budget` is
a per-turn allocation of the model's context window; `ContextBudget` is the
`AgentMiddleware` that enforces it on every call via `wrap_model_call` -
trimming the *view* the model sees this turn, never the durable history in
state or the checkpoint (that distinction is the seam: `wrap_model_call`
rewrites the request, `before_model` would mutate persisted state).

`select_docs` is the hot-path half of *select*: rank retrieved documents by
score and cap them to the retrieved slice, instead of handing the model
every document `atlas.graph`'s `retrieve` node fetched. The richer,
long-term-memory half of select is Chapters 13-14.

`atlas/state.py`'s `Doc` gained a `score: float` field this chapter -
additive, the same migration discipline Chapter 10 used for `refund_done`:
existing code that only reads `doc["id"]`/`doc["text"]` is unaffected, and
`select_docs` is the first reader of the new field."""

from collections.abc import Callable
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from atlas.state import Doc


@dataclass
class Budget:
    """A per-turn allocation of the model's context window, in tokens.
    System prompt, scratch, and headroom take the remainder."""

    history: int  # reserved for conversation history
    retrieved: int  # reserved for retrieved documents


class ContextBudget(AgentMiddleware):
    """Enforce the budget on every model call - without deleting anything
    from persisted state."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        trimmed = trim_messages(
            request.messages,
            max_tokens=self.budget.history,
            token_counter=count_tokens_approximately,
            strategy="last",  # keep the most recent turns
            start_on="human",  # never start on a dangling ToolMessage
            include_system=True,  # always keep the system prompt
        )
        return handler(request.override(messages=trimmed))


def select_docs(docs: list[Doc], max_tokens: int) -> list[Doc]:
    """Take the highest-scored docs until the retrieved slice is spent.
    Include the best few that fit, not everything fetched."""
    kept, used = [], 0
    for doc in sorted(docs, key=lambda d: d["score"], reverse=True):
        cost = count_tokens_approximately([HumanMessage(doc["text"])])
        if used + cost > max_tokens:
            break
        kept.append(doc)
        used += cost
    return kept
