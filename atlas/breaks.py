"""Breaking atlas/naive.py three ways, deterministically and offline.

See Chapter 1, "Watching it break". A ScriptedModel stands in for a live LLM so
each failure mode - non-termination, lost state, a swallowed exception - can be
reproduced with no API key and no flakiness.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from atlas.naive import SYSTEM_PROMPT, TOOLS_BY_NAME, refund_calls, run, text_of


class ScriptedModel:
    """Returns pre-baked AI messages in order; repeats the last one forever.
    Lets us trigger each failure mode deterministically, with no live LLM."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages: list) -> AIMessage:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def run_with_watchdog(user_text, model, limit=25) -> str:
    """The same loop as run(), capped only so a non-terminating agent can be
    observed rather than hang. The cap is a diagnostic, not a solution."""
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_text)]
    for step in range(1, limit + 1):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return f"stopped after {step} step(s): {text_of(ai)}"
        for call in ai.tool_calls:
            TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage("ok", tool_call_id=call["id"]))
    return f"NEVER TERMINATED: still looping at step {limit}"


def run_swallowing(user_text, model) -> str:
    """The naive 'fix' for tool errors: catch and continue. It converts a loud
    failure into a silent, confident wrong answer."""
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_text)]
    while True:
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return text_of(ai)
        for call in ai.tool_calls:
            try:
                result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            except Exception as exc:                  # <1>
                result = f"Tool error: {exc}"         # <2>
            messages.append(ToolMessage(result, tool_call_id=call["id"]))


@tool
def issue_refund_broken(ticket_id: str) -> str:
    """Issue a refund - but the payment backend is down."""
    raise ConnectionError("payment gateway timeout")


if __name__ == "__main__":
    # Failure 1 - it runs forever.
    stuck = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "issue_refund", "args": {"ticket_id": "T-1001"},
             "id": "call_1", "type": "tool_call"},
        ]),
    ])
    print(run_with_watchdog("refund my order", stuck))
    # -> NEVER TERMINATED: still looping at step 25

    # Failure 2 - state dies on restart.
    two_turns = ScriptedModel([
        AIMessage(content="Sure - what is your order number?"),
        AIMessage(content="I don't have any record of a previous request. "
                          "Could you tell me what you need?"),
    ])
    print("turn 1:", run("I'd like a refund.", two_turns))
    # ... imagine a deploy restarts the process here ...
    print("turn 2:", run("It's T-1001.", two_turns))
    # turn 2: I don't have any record of a previous request...

    # Failure 3 - a swallowed exception, and a confident lie.
    confirm = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "issue_refund", "args": {"ticket_id": "T-1001"},
             "id": "c1", "type": "tool_call"}]),
        AIMessage(content="All set - your refund has been issued. Anything else?"),
    ])

    TOOLS_BY_NAME["issue_refund"] = issue_refund_broken  # the backend is down
    print(run_swallowing("Refund T-1001 please", confirm))
    # -> All set - your refund has been issued. Anything else?
    print("refund actually issued?", "T-1001" in refund_calls)
    # -> refund actually issued? False
