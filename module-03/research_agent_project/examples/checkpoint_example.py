"""
Checkpoint and resume example for the Research Agent.

This example demonstrates:
1. Running research with persistent checkpointing
2. Simulating a failure
3. Resuming from the last checkpoint
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import create_research_agent, create_initial_state
from langgraph.checkpoint.sqlite import SqliteSaver

# Configure logging to see agent progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    """Demonstrate checkpointing and resume."""
    
    # Create persistent checkpointer
    db_path = "research_checkpoints.db"
    print(f"📦 Using SQLite checkpointer: {db_path}")
    checkpointer = SqliteSaver.from_conn_string(db_path)
    
    # Create agent with checkpointing
    agent = create_research_agent(checkpointer=checkpointer)
    
    # Initial state
    initial_state = create_initial_state(
        research_query="Renewable energy trends in 2024",
        max_retries=2
    )
    
    config = {"configurable": {"thread_id": "checkpoint-demo"}}
    
    print("\n" + "=" * 60)
    print("PHASE 1: Initial Execution")
    print("=" * 60)
    
    # Run until completion
    print("\nExecuting research agent...")
    try:
        step_count = 0
        for step in agent.stream(initial_state, config=config):
            step_count += 1
            node_name = list(step.keys())[0]
            print(f"  Step {step_count}: {node_name}")
            
            # Simulate failure after 3 steps (optional - comment out for full run)
            # if step_count == 3:
            #     raise Exception("Simulated failure!")
        
        print("✅ Initial execution completed")
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        print("💾 State was checkpointed before failure")
    
    # Inspect checkpoint
    print("\n" + "=" * 60)
    print("PHASE 2: Inspecting Checkpoint")
    print("=" * 60)
    
    state = agent.get_state(config)
    print(f"\nCheckpoint information:")
    print(f"  - Current stage: {state.values.get('current_stage')}")
    print(f"  - Retry count: {state.values.get('retry_count')}")
    print(f"  - Search queries: {len(state.values.get('search_queries', []))}")
    print(f"  - Search results: {len(state.values.get('search_results', []))}")
    
    # Resume from checkpoint
    print("\n" + "=" * 60)
    print("PHASE 3: Resuming from Checkpoint")
    print("=" * 60)
    
    print("\nResuming execution (invoke with None = use saved state)...")
    result = agent.invoke(None, config=config)
    
    print("✅ Execution resumed and completed")
    
    # Print final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result["report"])
    
    # Show checkpoint history
    print("\n" + "=" * 60)
    print("PHASE 4: Checkpoint History")
    print("=" * 60)
    
    print("\nCheckpoint history for this thread:")
    history = agent.get_state_history(config)
    for i, checkpoint_state in enumerate(list(history)[:5], 1):  # Show last 5
        print(f"\n  Checkpoint {i}:")
        print(f"    Stage: {checkpoint_state.values.get('current_stage')}")
        print(f"    Timestamp: {checkpoint_state.config.get('configurable', {}).get('checkpoint_id', 'N/A')[:8]}...")
    
    print("\n" + "=" * 60)
    print("✅ Checkpointing demo complete!")
    print(f"💾 Checkpoint database: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
