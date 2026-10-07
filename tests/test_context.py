"""Chapter 12, "Context Engineering" - atlas/context.py.

See "Enforcing the budget". `ContextBudget.wrap_model_call` needs no live
model call to test - it only rewrites the `ModelRequest` before handing it
to `handler`, so a dummy model object standing in for `BaseChatModel` (the
same no-live-call convention `tests/test_middleware.py` and
`tests/test_agent.py` use) is enough to exercise it directly, without
`create_agent` or a real chat model."""

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from atlas.context import Budget, ContextBudget, select_docs


def _request(messages: list) -> ModelRequest:
    return ModelRequest(
        model=object(),  # stand-in for BaseChatModel; never invoked in these tests
        messages=messages,
        system_message=SystemMessage("You are Atlas."),
    )


def _long_conversation(turns: int) -> list:
    """A plain-chat history with no tool calls, long enough that a small
    history budget forces a trim."""
    messages = []
    for i in range(turns):
        messages.append(HumanMessage(f"question number {i} about my order"))
        messages.append(AIMessage(f"answer number {i} about your order"))
    return messages


def test_budget_holds_the_two_named_slices():
    budget = Budget(history=4000, retrieved=2000)

    assert budget.history == 4000
    assert budget.retrieved == 2000


def test_context_budget_trims_the_view_not_the_request_object_in_place():
    """wrap_model_call must call handler with an OVERRIDDEN request, not
    mutate the one it was given - the durable history is untouched."""
    budget = ContextBudget(Budget(history=50, retrieved=500))
    original = _request(_long_conversation(turns=20))
    seen = {}

    def handler(request: ModelRequest):
        seen["request"] = request
        return "ok"

    result = budget.wrap_model_call(original, handler)

    assert result == "ok"
    assert seen["request"] is not original
    assert len(seen["request"].messages) < len(original.messages)


def test_context_budget_keeps_the_view_under_the_history_slice():
    budget = ContextBudget(Budget(history=200, retrieved=500))
    request = _request(_long_conversation(turns=30))
    captured = []

    budget.wrap_model_call(request, lambda r: captured.append(r) or "ok")

    trimmed = captured[0].messages
    # The system prompt is not in request.messages, so it is not counted:
    # the trimmed history alone stays under the history slice.
    assert count_tokens_approximately(trimmed) <= 200
    assert len(trimmed) < len(request.messages)


def test_context_budget_never_lets_the_trimmed_window_start_on_an_orphaned_tool_message():
    """The "trimming can sever a tool-call pair" guard, reproduced directly:
    a budget tight enough to fit only the trailing `ToolMessage` would, with
    a plain "keep the last N tokens" trim, keep that tool result while
    dropping the `AIMessage` that issued its call - the malformed sequence
    the callout warns providers reject. `start_on="human"` is the fix: at
    the same tight budget it refuses to start on anything but a human
    message, so it returns an empty view instead of a broken one."""
    ends_on_tool_result = [
        HumanMessage("what is the refund window for my order"),
        AIMessage(
            "let me look that up for you right now",
            tool_calls=[
                {
                    "name": "search_kb",
                    "args": {"query": "refund window policy details extended"},
                    "id": "call-1",
                }
            ],
        ),
        ToolMessage("30 days", tool_call_id="call-1"),
    ]
    request = _request(ends_on_tool_result)

    # Without the guard, this exact budget keeps the orphaned ToolMessage
    # alone - the bug the chapter's callout describes.
    from langchain_core.messages import trim_messages

    unguarded = trim_messages(
        request.messages,
        max_tokens=8,
        token_counter=count_tokens_approximately,
        strategy="last",
    )
    assert len(unguarded) == 1
    assert isinstance(unguarded[0], ToolMessage)  # the broken sequence

    # With ContextBudget's start_on="human" guard, the same tight budget
    # never produces that broken sequence.
    budget = ContextBudget(Budget(history=8, retrieved=500))
    captured = []

    budget.wrap_model_call(request, lambda r: captured.append(r) or "ok")

    assert not any(isinstance(m, ToolMessage) for m in captured[0].messages)


def test_context_budget_leaves_the_system_message_field_untouched():
    """ModelRequest.messages excludes the system message by construction
    (it lives on request.system_message) - wrap_model_call only overrides
    `messages`, so the system prompt survives any budget, including one too
    tight to keep any conversation turns at all."""
    budget = ContextBudget(Budget(history=1, retrieved=500))
    request = _request(_long_conversation(turns=10))
    captured = []

    budget.wrap_model_call(request, lambda r: captured.append(r) or "ok")

    assert captured[0].system_message == request.system_message


def test_select_docs_keeps_the_highest_scored_docs_first():
    docs = [
        {"id": "1", "text": "short", "score": 0.2},
        {"id": "2", "text": "also short", "score": 0.9},
        {"id": "3", "text": "medium relevance", "score": 0.5},
    ]

    kept = select_docs(docs, max_tokens=1000)

    assert [d["id"] for d in kept] == ["2", "3", "1"]


def test_select_docs_stops_at_the_first_doc_that_does_not_fit():
    """`select_docs` walks the sorted-by-score list and `break`s on the
    first doc that would blow the slice - it does not skip ahead to a
    smaller, lower-ranked doc that might still fit. A budget that is
    entirely spent by the top-ranked doc yields nothing further."""
    docs = [
        {"id": "big", "text": "word " * 200, "score": 0.9},  # far over budget
        {"id": "small", "text": "brief", "score": 0.1},  # would fit alone
    ]

    kept = select_docs(docs, max_tokens=5)

    assert kept == []  # breaks on "big" before ever reaching "small"


def test_select_docs_on_empty_input_returns_empty():
    assert select_docs([], max_tokens=1000) == []


def test_select_docs_never_exceeds_the_token_budget():
    docs = [{"id": str(i), "text": f"document number {i} " * 5, "score": float(i)} for i in range(10)]

    kept = select_docs(docs, max_tokens=40)

    total = sum(count_tokens_approximately([HumanMessage(d["text"])]) for d in kept)
    assert total <= 40


def test_the_system_prompt_travels_outside_request_messages_in_a_real_agent():
    """Why ContextBudget passes no `include_system`: inside a real
    `create_agent` run, `request.messages` holds no SystemMessage - the
    prompt rides on `request.system_message` and is prepended only when the
    model is called. So the trim can neither drop it nor count it."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    seen = {}

    class Record(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen["messages"] = list(request.messages)
            seen["system"] = request.system_message
            return handler(request)

    model = FakeMessagesListChatModel(responses=[AIMessage("done")])
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are Atlas.",
        middleware=[ContextBudget(Budget(history=40, retrieved=500)), Record()],
    )
    agent.invoke({"messages": _long_conversation(turns=10) + [HumanMessage("hi")]})

    assert not any(isinstance(m, SystemMessage) for m in seen["messages"])
    assert seen["system"].content == "You are Atlas."
    assert count_tokens_approximately(seen["messages"]) <= 40
