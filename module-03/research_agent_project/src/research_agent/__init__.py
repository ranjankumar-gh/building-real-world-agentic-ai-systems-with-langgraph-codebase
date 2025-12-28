"""
Research Agent with LangGraph

A production-ready research agent demonstrating explicit state management,
deterministic flow, checkpointing, and retry logic.
"""

from .state import ResearchAgentState, create_initial_state
from .config import AgentConfig, DEFAULT_CONFIG
from .graph import create_research_agent
from .nodes import (
    plan_research,
    execute_search,
    validate_results,
    process_results,
    generate_report,
    handle_error
)

__version__ = "1.0.0"

__all__ = [
    # Main functions
    "create_research_agent",
    "create_initial_state",
    
    # State and config
    "ResearchAgentState",
    "AgentConfig",
    "DEFAULT_CONFIG",
    
    # Individual nodes (for testing/customization)
    "plan_research",
    "execute_search",
    "validate_results",
    "process_results",
    "generate_report",
    "handle_error",
]
