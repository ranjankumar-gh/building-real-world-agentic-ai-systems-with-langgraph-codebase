"""
State definition for the Research Agent.

This module defines the complete state schema that the agent uses
to track conversation, task progress, search results, and control flow.
"""

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ResearchAgentState(TypedDict):
    """
    Complete state for the research agent.
    
    This state is the single source of truth for the agent's execution.
    Every piece of information needed for decisions is stored here.
    
    Attributes:
        messages: Conversation history with LLM (auto-appended using add_messages)
        research_query: The original research question
        research_plan: Generated plan from planning node
        search_queries: List of search queries to execute
        search_results: Results from web searches (success or error)
        key_findings: Extracted findings from search results
        report: Final generated research report
        current_stage: Current execution stage (used for routing)
        retry_count: Number of retry attempts
        max_retries: Maximum allowed retries before giving up
        error_message: Error details if failure occurred
    """
    
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Task
    research_query: str
    research_plan: str
    
    # Search
    search_queries: list[str]
    search_results: list[dict]
    
    # Processing
    key_findings: list[str]
    report: str
    
    # Control flow
    current_stage: Literal[
        "planning",
        "searching", 
        "validating",
        "processing",
        "generating",
        "complete",
        "error"
    ]
    retry_count: int
    max_retries: int
    error_message: str | None


def create_initial_state(
    research_query: str,
    max_retries: int = 2
) -> ResearchAgentState:
    """
    Create initial state for a research task.
    
    Args:
        research_query: The research question to investigate
        max_retries: Maximum number of retry attempts (default: 2)
    
    Returns:
        ResearchAgentState with all fields initialized
    
    Example:
        >>> state = create_initial_state("Latest AI developments")
        >>> print(state["research_query"])
        Latest AI developments
    """
    return {
        "research_query": research_query,
        "messages": [],
        "search_queries": [],
        "search_results": [],
        "key_findings": [],
        "report": "",
        "current_stage": "planning",
        "retry_count": 0,
        "max_retries": max_retries,
        "error_message": None,
        "research_plan": ""
    }
