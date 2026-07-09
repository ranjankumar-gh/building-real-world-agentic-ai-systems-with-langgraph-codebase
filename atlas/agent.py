"""Chapter 7, "Tools, Models, MCP, and create_agent" - the tool-using agent.

See "Building Atlas's real tools" (binding the model) and "Reaching
external tools with MCP". `resolve_agent` is a standalone tool-calling
`create_agent`, built from the real tools in `atlas/tools.py` - it is not
yet wired into `atlas/graph.py`'s `retrieve`/`answer` nodes (that
integration is deferred to the chapters on sub-agents inside the larger
graph).

"Binding the model" shows two ways to pass the model to `create_agent`:
a bare string id (the common case; `create_agent` resolves it through
`init_chat_model`) and a configured `init_chat_model` instance (needed once
you set parameters like temperature or a token cap).
`build_resolve_agent_from_model_id` below is the string-shorthand builder;
`resolve_agent`, the module's live default, uses the configured instance.
Both are kept so neither binding style silently overwrites the other - the
agent code is otherwise identical either way. `build_resolve_agent` (async,
further down) is the chapter's own name for the MCP-connected builder from
"Reaching external tools with MCP" - kept exactly, since Exercise 2 refers
to it by that name.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from atlas.tools import lookup_ticket, search_kb, set_ticket_status

RESOLVE_TOOLS = [search_kb, lookup_ticket, set_ticket_status]

RESOLVE_PROMPT = (
    "You are Atlas, a customer-support assistant. Use search_kb for factual "
    "questions and the ticket tools to read or change ticket state. Never "
    "claim an action succeeded unless a tool result confirmed it."
)


def build_resolve_agent_from_model_id(model_id: str = "claude-sonnet-4-6"):
    """The string-shorthand binding: create_agent resolves `model_id`
    through init_chat_model. See "Binding the model"."""
    return create_agent(model=model_id, tools=RESOLVE_TOOLS, system_prompt=RESOLVE_PROMPT)


# The configured-instance binding, used when model parameters matter
# (temperature, token caps, timeouts). This is the module's live default.
model = init_chat_model("claude-sonnet-4-6", temperature=0, max_tokens=1024)
resolve_agent = create_agent(
    model=model, tools=RESOLVE_TOOLS, system_prompt=RESOLVE_PROMPT
)


async def build_resolve_agent():
    """Reach an external tool server over MCP and fold its tools in
    alongside the in-process ones. See "Reaching external tools with MCP".

    `client.get_tools()` can fail open - if any configured server fails to
    connect it can silently return fewer tools, or none - so the guard
    below refuses to start rather than run with a silently shrunken
    authority surface. See "Production considerations"."""
    client = MultiServerMCPClient(
        {
            "atlas-status": {
                "transport": "stdio",
                "command": "python",
                "args": ["atlas/mcp_server.py"],
            }
        }
    )
    mcp_tools = await client.get_tools()
    if not mcp_tools:  # <1>
        raise RuntimeError("MCP server returned no tools; refusing to start.")
    return create_agent(
        model="claude-sonnet-4-6",
        tools=RESOLVE_TOOLS + mcp_tools,  # <2>
        system_prompt=RESOLVE_PROMPT,
    )


# 1. The guard matters more than it looks - see "Production considerations".
#    get_tools() can return an empty list when a server fails to connect,
#    silently shrinking the authority surface.
# 2. MCP tools and in-process tools are the same type from here on; the
#    agent does not distinguish them.
