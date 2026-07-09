"""Chapter 7, "Tools, Models, MCP, and create_agent" - a minimal MCP server.

See "Reaching external tools with MCP". `FastMCP` builds a server from
decorated functions the same way `@tool` builds a LangChain tool: the
function name becomes the tool name, the docstring becomes the
description. `atlas/agent.py`'s `build_resolve_agent` connects to this
server with `MultiServerMCPClient` over the `stdio` transport (it launches
this file as a subprocess) and loads `service_status` as an ordinary tool.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("atlas-status")


@mcp.tool()
def service_status(component: str) -> str:
    """Report the current operational status of an Atlas backend component."""
    statuses = {"kb": "operational", "ticketing": "operational"}
    return statuses.get(component, "unknown")


if __name__ == "__main__":
    mcp.run(transport="stdio")  # local subprocess transport
