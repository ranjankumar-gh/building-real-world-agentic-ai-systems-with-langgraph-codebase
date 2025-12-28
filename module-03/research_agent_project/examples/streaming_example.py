"""
Streaming execution example for the Research Agent.

This example demonstrates real-time streaming of agent execution:
- See each node as it executes
- Monitor progress in real-time
- Display state changes
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import create_research_agent, create_initial_state


def print_step(node_name: str, node_output: dict):
    """Pretty print execution step."""
    print(f"\n{'=' * 60}")
    print(f"NODE: {node_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Stage: {node_output.get('current_stage', 'N/A')}")
    print(f"Retry count: {node_output.get('retry_count', 0)}")
    
    if "error_message" in node_output and node_output["error_message"]:
        print(f"⚠️  Error: {node_output['error_message']}")
    
    # Show search queries if present
    if "search_queries" in node_output and node_output["search_queries"]:
        print(f"\nGenerated {len(node_output['search_queries'])} search queries:")
        for i, query in enumerate(node_output["search_queries"][:3], 1):
            print(f"  {i}. {query}")
    
    # Show search results summary if present
    if "search_results" in node_output and node_output["search_results"]:
        total = len(node_output["search_results"])
        valid = sum(1 for r in node_output["search_results"] if "error" not in r)
        print(f"\nSearch results: {valid}/{total} successful")
    
    # Show findings if present
    if "key_findings" in node_output and node_output["key_findings"]:
        print(f"\nExtracted {len(node_output['key_findings'])} key findings")


def main():
    """Run research with streaming output."""
    
    # Create initial state
    initial_state = create_initial_state(
        research_query="Impact of artificial intelligence on healthcare",
        max_retries=2
    )
    
    # Create agent
    print("🔬 Starting Research Agent with Streaming")
    print("=" * 60)
    print(f"Query: {initial_state['research_query']}")
    print("=" * 60)
    
    agent = create_research_agent()
    config = {"configurable": {"thread_id": "research-streaming"}}
    
    # Stream execution
    for step in agent.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        print_step(node_name, node_output)
    
    # Get final state
    final_state = agent.get_state(config)
    
    # Print final report
    print(f"\n{'=' * 60}")
    print("📄 FINAL REPORT")
    print(f"{'=' * 60}\n")
    print(final_state.values["report"])
    print(f"\n{'=' * 60}")
    
    # Summary
    print("\n✅ Research completed successfully!")
    print(f"Total retry attempts: {final_state.values['retry_count']}")


if __name__ == "__main__":
    main()
