"""Chapter 7, "Tools, Models, MCP, and create_agent" - atlas/mcp_server.py.

See "Reaching external tools with MCP". A real end-to-end test would spawn
this module as a subprocess and talk to it over the `stdio` transport via
`MultiServerMCPClient` - that is what `atlas/agent.py`'s `build_resolve_agent`
does, and it is exercised (with a fake client) in `tests/test_agent.py`.
Spinning up a real MCP client/server pair here would mean shelling out to a
subprocess in the test suite, which is exactly the kind of network/process
dependency this book's tests avoid - so this file instead exercises the
server object and its registered tool directly, in-process, the way
`FastMCP` builds and would dispatch to it."""

import asyncio

from atlas.mcp_server import mcp, service_status


def test_service_status_reports_operational_for_known_components():
    assert service_status("kb") == "operational"
    assert service_status("ticketing") == "operational"


def test_service_status_reports_unknown_for_an_unrecognized_component():
    assert service_status("nonexistent") == "unknown"


def test_the_server_registers_service_status_with_its_docstring_as_contract():
    """Same contract discipline as `@tool`: the function name becomes the
    tool name, the docstring becomes the description."""
    tools = asyncio.run(mcp.list_tools())

    assert len(tools) == 1
    assert tools[0].name == "service_status"
    assert tools[0].description == (
        "Report the current operational status of an Atlas backend component."
    )
