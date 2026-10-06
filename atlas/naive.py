"""Atlas v0: a deliberately fragile agent - a hand-rolled while loop around an LLM.

See Chapter 1 ("The Agent Reliability Problem"). This is the naive orchestration
this book spends the rest of its chapters replacing - do not build on this module.
"""

from typing import Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

# Seeded, mockable backend - ships in the companion repo, no external account.
_REFUNDS: dict[str, dict] = {
    "T-1001": {"status": "pending", "amount": 49.0},
}
refund_calls: list[str] = []   # a stand-in for observability: what actually ran


@tool
def issue_refund(ticket_id: str) -> str:
    """Issue a refund for a ticket. This is a side-effecting write."""
    refund_calls.append(ticket_id)
    record = _REFUNDS[ticket_id]
    record["status"] = "refunded"
    return f"Refund of ${record['amount']:.2f} issued for {ticket_id}."


def text_of(message: BaseMessage) -> str:
    """Plain text from a message whose content may be a string or a list
    of content blocks (model-agnostic; see Chapter 7)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
    )


SYSTEM_PROMPT = (
    "You are Atlas, a customer-support assistant. Resolve the user's request "
    "using the available tools, then reply with a short confirmation."
)
TOOLS_BY_NAME = {"issue_refund": issue_refund}


def build_model() -> Runnable:
    """The real path: Claude via LangChain, with tools bound. Swap the model
    id to change providers; the loop below does not change."""
    model = init_chat_model("claude-sonnet-4-6", temperature=0)
    return model.bind_tools(list(TOOLS_BY_NAME.values()))


class ChatModel(Protocol):
    """All run() needs from a model: .invoke(messages) returning a message.
    The bound model from build_model() fits, and so does a scripted stand-in."""

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage: ...


def run(user_text: str, model: ChatModel) -> str:
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_text)]
    while True:                                       # <1>
        ai = model.invoke(messages)
        messages.append(ai)
        if ai.tool_calls:                             # <2>
            for call in ai.tool_calls:
                result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])  # <3>
                messages.append(ToolMessage(result, tool_call_id=call["id"]))
            continue
        return text_of(ai)                            # <4>
