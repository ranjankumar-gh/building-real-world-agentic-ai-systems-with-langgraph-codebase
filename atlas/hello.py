"""Hello-world Atlas: the smallest correct baseline, built with create_agent.

See Chapter 2, "Hello-world Atlas". Direct replacement for Chapter 1's
atlas/naive.py - same job, no hand-rolled while loop. Still not durable
(no checkpointer; Chapter 9), not supervised (no HITL; Chapter 11), and not
traced (Chapter 20). create_agent hands the loop to the runtime, but its
only loop bound is a 9,999-step recursion limit and a raising tool still
ends the run (Chapter 7 adds the error policy).

Run it with `uv run python -m atlas.hello` (needs ANTHROPIC_API_KEY).
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

# A stand-in for retrieval. The real knowledge base sits behind a tool
# interface (Appendix G); Chapter 7 builds the real thing.
_KB = {"refund window": "Refunds are available within 30 days of purchase."}


@tool
def kb_lookup(query: str) -> str:
    """Look up an answer in the support knowledge base."""
    for key, answer in _KB.items():
        if key in query.lower():
            return answer
    return "No knowledge-base article matched."


SYSTEM_PROMPT = (
    "You are Atlas, a customer-support assistant. Use the knowledge base when "
    "it is relevant, and keep replies short."
)

agent = create_agent(                       # <1>
    model="claude-sonnet-4-6",              # <2>
    tools=[kb_lookup],
    system_prompt=SYSTEM_PROMPT,            # <3>
)

# 1. create_agent returns a compiled LangGraph graph - the runtime is
#    underneath, even though it is never named directly.
# 2. The model is passed as a string id; create_agent resolves it through
#    init_chat_model, so switching providers is a one-line change.
# 3. system_prompt=, not prompt=. This is the 1.x rename; the old keyword is
#    the 0.x tell.


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the refund window?"}]}
    )
    print(result["messages"][-1].content)
    # -> Refunds are available within 30 days of purchase.
