"""
Agent nodes for the Research Agent.

Each node is a function that transforms state. Nodes are:
1. plan_research - Generate search plan
2. execute_search - Run web searches
3. validate_results - Check result quality
4. process_results - Extract key findings
5. generate_report - Create final report
6. handle_error - Manage retry logic
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .state import ResearchAgentState
from .config import AgentConfig, DEFAULT_CONFIG
from .tools import search_tool

logger = logging.getLogger(__name__)


def create_llm(config: AgentConfig = DEFAULT_CONFIG) -> ChatOllama:
    """
    Create LLM instance with configuration.

    Args:
        config: Agent configuration

    Returns:
        Configured ChatOllama instance
    """
    return ChatOllama(
        model=config.llm_model,
        temperature=config.llm_temperature,
        base_url=config.ollama_base_url
    )


def plan_research(
    state: ResearchAgentState
) -> dict:
    """
    Node 1: Planning - Create research plan and search queries.

    This node analyzes the research query and breaks it down into
    specific search queries that will yield useful results.

    Args:
        state: Current agent state

    Returns:
        State updates with research plan and search queries
    """
    logger.info(f"Planning research for: {state['research_query']}")

    llm = create_llm(DEFAULT_CONFIG)
    query = state["research_query"]
    
    planning_prompt = f"""Create a research plan for: {query}
    
    Output:
    1. List of 3-5 specific search queries
    2. Key aspects to investigate
    
    Be specific and focused."""
    
    messages = [
        SystemMessage(content="You are a research planning assistant."),
        HumanMessage(content=planning_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Extract search queries (simplified - in practice, use structured output)
    queries = [
        q.strip() 
        for q in response.content.split("\n") 
        if q.strip() and not q.startswith("#")
    ][:5]
    
    logger.info(f"Generated {len(queries)} search queries")
    
    return {
        "research_plan": response.content,
        "search_queries": queries,
        "current_stage": "searching",
        "messages": [response]
    }


def execute_search(
    state: ResearchAgentState
) -> dict:
    """
    Node 2: Search Execution - Execute web searches.

    Runs each search query and collects results. Handles errors
    gracefully by recording them in results rather than crashing.

    Args:
        state: Current agent state

    Returns:
        State updates with search results
    """
    queries = state["search_queries"]
    logger.info(f"Executing {len(queries[:DEFAULT_CONFIG.search_limit])} searches")
    
    all_results = []
    for query in queries[:DEFAULT_CONFIG.search_limit]:
        try:
            logger.debug(f"Searching: {query}")
            result = search_tool.run(query)
            all_results.append({
                "query": query,
                "result": result
            })
            logger.debug(f"Search successful: {query}")
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {str(e)}")
            all_results.append({
                "query": query,
                "error": str(e)
            })
    
    valid_count = sum(1 for r in all_results if "error" not in r)
    logger.info(f"Completed searches: {valid_count}/{len(all_results)} successful")
    
    return {
        "search_results": all_results,
        "current_stage": "validating"
    }


def validate_results(
    state: ResearchAgentState
) -> dict:
    """
    Node 3: Validation - Check if search results are sufficient.

    Validates that we have enough good results to proceed with
    processing. If not, routes to error handler for retry.

    Args:
        state: Current agent state

    Returns:
        State updates with validation result
    """
    results = state["search_results"]
    valid_results = [r for r in results if "error" not in r]

    logger.info(f"Validating results: {len(valid_results)} valid out of {len(results)}")

    if len(valid_results) >= DEFAULT_CONFIG.min_valid_results:
        logger.info("Validation passed - proceeding to processing")
        return {
            "current_stage": "processing"
        }
    else:
        logger.warning(
            f"Validation failed - only {len(valid_results)} valid results "
            f"(need {DEFAULT_CONFIG.min_valid_results})"
        )
        return {
            "current_stage": "error",
            "retry_count": state["retry_count"] + 1,
            "error_message": "Insufficient search results"
        }


def process_results(
    state: ResearchAgentState
) -> dict:
    """
    Node 4: Processing - Extract key findings from search results.

    Uses LLM to analyze search results and extract the most
    important, relevant findings related to the research query.

    Args:
        state: Current agent state

    Returns:
        State updates with extracted findings
    """
    logger.info("Processing search results")

    llm = create_llm(DEFAULT_CONFIG)
    results = state["search_results"]
    query = state["research_query"]
    
    # Log search results summary
    valid_results = [r for r in results if "error" not in r]
    logger.info(f"Processing {len(valid_results)} valid results out of {len(results)} total")
    
    # Prepare results summary
    results_text = "\n\n".join([
        f"Query: {r['query']}\nResults: {r.get('result', 'N/A')}"
        for r in results
        if "error" not in r
    ])
    
    logger.debug(f"Results text length: {len(results_text)} characters")
    
    processing_prompt = f"""Based on these search results, identify 5 key findings related to: {query}
    
    Search Results:
    {results_text}
    
    Extract concise, factual key findings (one sentence each).
    Format each finding as a bullet point starting with a dash (-)."""
    
    messages = [
        SystemMessage(content="You are a research analyst."),
        HumanMessage(content=processing_prompt)
    ]
    
    logger.info("Invoking LLM to extract findings")
    response = llm.invoke(messages)
    logger.debug(f"LLM response length: {len(response.content)} characters")
    
    # Extract findings - improved parsing
    findings = []
    for line in response.content.split("\n"):
        line = line.strip()
        # Check if line starts with common bullet formats
        if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
            # Remove bullet marker and clean
            finding = line.lstrip("-•* ").strip()
            if finding:  # Only add non-empty findings
                findings.append(finding)
                logger.debug(f"Extracted finding {len(findings)}: {finding[:60]}...")
    
    # Limit to 5 findings as requested
    findings = findings[:5]
    
    logger.info(f"Successfully extracted {len(findings)} key findings")
    
    # Log each finding for observability
    if findings:
        logger.info("Key findings extracted:")
        for i, finding in enumerate(findings, 1):
            logger.info(f"  Finding {i}: {finding[:100]}{'...' if len(finding) > 100 else ''}")
    else:
        logger.warning("No findings could be extracted from LLM response")
        logger.debug(f"LLM response was: {response.content[:500]}...")
    
    return {
        "key_findings": findings,
        "current_stage": "generating",
        "messages": [response]
    }

def generate_report(
    state: ResearchAgentState
) -> dict:
    """
    Node 5: Report Generation - Create final research report.

    Synthesizes key findings into a structured, professional
    research report with executive summary and conclusion.

    Args:
        state: Current agent state

    Returns:
        State updates with final report
    """
    logger.info("Generating final report")

    llm = create_llm(DEFAULT_CONFIG)
    query = state["research_query"]
    findings = state["key_findings"]
    
    report_prompt = f"""Create a concise research report on: {query}
    
    Key Findings:
    {chr(10).join(f"- {f}" for f in findings)}
    
    Structure:
    1. Executive Summary (2-3 sentences)
    2. Key Findings (bullet points)
    3. Conclusion (1-2 sentences)
    
    Keep it professional and factual."""
    
    messages = [
        SystemMessage(content="You are a research report writer."),
        HumanMessage(content=report_prompt)
    ]
    
    response = llm.invoke(messages)
    
    logger.info("Report generated successfully")
    
    return {
        "report": response.content,
        "current_stage": "complete",
        "messages": [response]
    }


def handle_error(
    state: ResearchAgentState
) -> dict:
    """
    Node 6: Error Handler - Manage retry logic.

    Decides whether to retry failed operations or fail gracefully.
    Implements retry counter logic to prevent infinite loops.

    Args:
        state: Current agent state

    Returns:
        State updates with retry decision
    """
    error = state["error_message"]
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    
    logger.warning(f"Error handler invoked: {error} (retry {retry_count}/{max_retries})")
    
    if retry_count < max_retries:
        logger.info(f"Retrying - attempt {retry_count + 1}/{max_retries}")
        return {
            "current_stage": "searching",  # Retry search
            "error_message": None
        }
    else:
        logger.error(f"Max retries exceeded - failing gracefully")
        return {
            "report": f"Failed to complete research: {error}",
            "current_stage": "complete"
        }
