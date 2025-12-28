"""
Graph construction and routing for the Research Agent.

This module builds the agent workflow graph with nodes and edges,
defining the complete execution flow including retry logic.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import ResearchAgentState
from .nodes import (
    plan_research,
    execute_search,
    validate_results,
    process_results,
    generate_report,
    handle_error
)

logger = logging.getLogger(__name__)


def route_after_validation(state: ResearchAgentState) -> str:
    """
    Router function after validation node.
    
    Routes based on validation outcome:
    - If processing: results are valid, proceed to processing
    - If error: insufficient results, route to error handler
    
    Args:
        state: Current agent state
    
    Returns:
        Next node name
    """
    stage = state["current_stage"]
    logger.debug(f"Routing after validation: stage={stage}")
    
    if stage == "processing":
        return "process"
    elif stage == "error":
        return "handle_error"
    return "process"


def route_after_error(state: ResearchAgentState) -> str:
    """
    Router function after error handler node.
    
    Routes based on retry decision:
    - If searching: retry the search
    - Otherwise: give up and end
    
    Args:
        state: Current agent state
    
    Returns:
        Next node name or END
    """
    stage = state["current_stage"]
    logger.debug(f"Routing after error: stage={stage}")
    
    if stage == "searching":
        return "search"  # Retry
    return "end"


def create_workflow() -> StateGraph:
    """
    Create the research agent workflow graph.
    
    The workflow follows this structure:
    START → plan → search → validate → process → generate → END
                              ↓           ↓
                            error ← → retry_search
    
    Returns:
        StateGraph instance (not yet compiled)
    """
    logger.info("Creating workflow graph")
    
    # Create graph
    workflow = StateGraph(ResearchAgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("search", execute_search)
    workflow.add_node("validate", validate_results)
    workflow.add_node("process", process_results)
    workflow.add_node("generate", generate_report)
    workflow.add_node("handle_error", handle_error)
    
    logger.debug("Added 6 nodes to graph")
    
    # Set entry point
    workflow.set_entry_point("plan")
    
    # Add static edges (unconditional flow)
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "validate")
    workflow.add_edge("process", "generate")
    workflow.add_edge("generate", END)
    
    logger.debug("Added static edges")
    
    # Add conditional edges (decision points)
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "process": "process",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "search": "search",
            "end": END
        }
    )
    
    logger.debug("Added conditional edges")
    logger.info("Workflow graph created successfully")
    
    return workflow


def create_research_agent(
    checkpointer: BaseCheckpointSaver | None = None
):
    """
    Create a compiled research agent.
    
    This is the main factory function for creating the agent.
    Use this in your application code.
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
                     If None, uses MemorySaver (in-memory, for development)
    
    Returns:
        Compiled agent ready for execution
    
    Example:
        >>> agent = create_research_agent()
        >>> result = agent.invoke(initial_state, config=config)
        
        >>> # With persistent checkpointing
        >>> from langgraph.checkpoint.sqlite import SqliteSaver
        >>> checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        >>> agent = create_research_agent(checkpointer=checkpointer)
    """
    logger.info("Creating research agent")
    
    # Use memory saver if no checkpointer provided
    if checkpointer is None:
        logger.info("No checkpointer provided, using MemorySaver")
        checkpointer = MemorySaver()
    else:
        logger.info(f"Using checkpointer: {type(checkpointer).__name__}")
    
    # Build and compile graph
    workflow = create_workflow()
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Research agent created successfully")
    return app
