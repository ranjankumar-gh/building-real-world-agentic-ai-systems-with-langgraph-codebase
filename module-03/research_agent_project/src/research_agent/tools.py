"""
Search tools for the Research Agent.

This module provides web search capabilities using DuckDuckGo.
Can be extended to support other search providers.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from typing import Protocol


class SearchTool(Protocol):
    """Protocol for search tools."""
    
    def run(self, query: str) -> str:
        """Execute a search query and return results."""
        ...


def create_search_tool() -> SearchTool:
    """
    Create a DuckDuckGo search tool.
    
    DuckDuckGo is used because:
    - No API key required
    - Free to use
    - Reasonable rate limits
    - Good for development/testing
    
    For production, consider:
    - SerpAPI (more reliable, better results)
    - Google Custom Search (official Google results)
    - Bing Search API (Microsoft results)
    
    Returns:
        SearchTool instance
    
    Example:
        >>> search = create_search_tool()
        >>> results = search.run("Python programming")
        >>> print(results[:100])
    """
    return DuckDuckGoSearchRun()


def create_search_tools() -> list[SearchTool]:
    """
    Create list of all available search tools.
    
    Currently only DuckDuckGo is configured.
    Add more tools here as needed.
    
    Returns:
        List of search tools
    """
    return [create_search_tool()]


# Global search tool instance
search_tool = create_search_tool()
tools = create_search_tools()
