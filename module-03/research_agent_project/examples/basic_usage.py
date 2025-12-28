"""
Basic usage example for the Research Agent.

This example shows the simplest way to use the agent:
1. Create initial state
2. Create agent
3. Run research
4. Print report
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import create_research_agent, create_initial_state


def main():
    """Run a basic research query."""
    
    # Create initial state
    initial_state = create_initial_state(
        research_query="Latest developments in quantum computing",
        max_retries=2
    )
    
    # Create agent
    print("Creating research agent...")
    agent = create_research_agent()
    
    # Configure execution
    config = {"configurable": {"thread_id": "research-001"}}
    
    # Run research
    print(f"\nResearching: {initial_state['research_query']}\n")
    print("=" * 60)
    
    result = agent.invoke(initial_state, config=config)
    
    # Print final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result["report"])
    print("\n" + "=" * 60)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  - Search queries: {len(result['search_queries'])}")
    print(f"  - Valid results: {len([r for r in result['search_results'] if 'error' not in r])}")
    print(f"  - Key findings: {len(result['key_findings'])}")
    print(f"  - Retry count: {result['retry_count']}")
    print(f"  - Final stage: {result['current_stage']}")


if __name__ == "__main__":
    main()
