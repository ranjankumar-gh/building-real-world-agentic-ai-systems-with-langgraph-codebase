#!/usr/bin/env python
"""
Quick run script for the Research Agent.

Usage:
    python run.py "Your research question here"
    
Or just run without arguments for interactive mode:
    python run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent import create_research_agent, create_initial_state


def run_research(query: str, max_retries: int = 2, verbose: bool = True):
    """
    Run a research query.
    
    Args:
        query: Research question
        max_retries: Maximum retry attempts
        verbose: Print progress
    
    Returns:
        Final state with report
    """
    # Create initial state
    initial_state = create_initial_state(query, max_retries=max_retries)
    
    # Create agent
    if verbose:
        print("🔬 Research Agent")
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60 + "\n")
    
    agent = create_research_agent()
    config = {"configurable": {"thread_id": f"research-{hash(query)}"}}
    
    # Stream execution if verbose
    if verbose:
        for step in agent.stream(initial_state, config=config):
            node_name = list(step.keys())[0]
            print(f"✓ {node_name}")
        
        # Get final state
        final_state = agent.get_state(config)
        result = final_state.values
    else:
        # Just invoke
        result = agent.invoke(initial_state, config=config)
    
    # Print report
    if verbose:
        print("\n" + "=" * 60)
        print("📄 RESEARCH REPORT")
        print("=" * 60 + "\n")
    
    print(result["report"])
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"✅ Completed (retries: {result['retry_count']})")
        print("=" * 60)
    
    return result


def interactive_mode():
    """Run in interactive mode."""
    print("🔬 Research Agent - Interactive Mode")
    print("=" * 60)
    print("Enter your research questions (or 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("\nResearch query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            print()
            run_research(query, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or check your API key.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Query provided as argument
        query = " ".join(sys.argv[1:])
        run_research(query, verbose=True)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
