"""
Production-ready agent implementation using LangChain.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools import calculator_tool, web_search_tool, weather_tool

# Load environment variables
load_dotenv()


class ProductionAgent:
    """
    A production-ready single-loop agent with:
    - Tool calling
    - Memory management
    - Error handling
    - Execution tracking
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        temperature: float = 0,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize LLM with Ollama
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        # Define tools
        self.tools = [
            calculator_tool,
            web_search_tool,
            weather_tool
        ]

        # System prompt
        self.system_prompt = f"""You are a helpful assistant with access to tools.

Use tools when necessary to provide accurate, up-to-date information.
Think step-by-step about which tool to use.

Current date: {datetime.now().strftime("%Y-%m-%d")}

Remember:
- Use calculator for any mathematical operations
- Use web_search for current events or information you don't have
- Use get_weather for weather information"""

        # Create agent graph
        self.agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            debug=verbose
        )

        # Memory: Conversation history
        self.messages: List[Dict[str, str]] = []

        # Tracking: Execution metrics
        self.execution_log = []

    def run(self, user_input: str) -> dict:
        """
        Execute agent with user input.

        Returns comprehensive result including:
        - Final answer
        - Intermediate steps
        - Execution metadata
        """
        start_time = datetime.now()

        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})

            # Execute agent
            inputs = {"messages": self.messages}
            result = None
            step_count = 0

            for chunk in self.agent_graph.stream(inputs, stream_mode="updates"):
                if self.verbose:
                    print(chunk)
                step_count += 1
                result = chunk

            # Extract final answer from the last AI message
            final_answer = ""
            if result and 'model' in result:
                # The key is 'model' not 'agent' in the new LangChain API
                ai_messages = result['model'].get('messages', [])
                if ai_messages:
                    last_message = ai_messages[-1]
                    if hasattr(last_message, 'content'):
                        final_answer = last_message.content
                    elif isinstance(last_message, dict):
                        final_answer = last_message.get('content', '')

            # Update message history with agent response
            if final_answer:
                self.messages.append({"role": "assistant", "content": final_answer})

            # Track execution
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_record = {
                "timestamp": start_time.isoformat(),
                "input": user_input,
                "output": final_answer,
                "steps": step_count,
                "execution_time": execution_time,
                "success": True
            }
            self.execution_log.append(execution_record)

            return {
                "answer": final_answer,
                "intermediate_steps": [],  # Would need to parse from stream
                "execution_time": execution_time,
                "success": True
            }

        except Exception as e:
            # Log error
            execution_record = {
                "timestamp": start_time.isoformat(),
                "input": user_input,
                "error": str(e),
                "success": False
            }
            self.execution_log.append(execution_record)

            return {
                "answer": f"I encountered an error: {str(e)}",
                "error": str(e),
                "success": False
            }

    def get_execution_stats(self) -> dict:
        """Get execution statistics."""
        if not self.execution_log:
            return {"message": "No executions yet"}

        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e.get("success"))
        avg_time = sum(e.get("execution_time", 0) for e in self.execution_log) / total
        avg_steps = sum(e.get("steps", 0) for e in self.execution_log if "steps" in e) / total

        return {
            "total_executions": total,
            "successful": successful,
            "success_rate": successful / total,
            "avg_execution_time": avg_time,
            "avg_steps_per_execution": avg_steps
        }

    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = []
        print("Conversation history reset")


def main():
    """Test the production agent."""

    print("=" * 60)
    print("PRODUCTION AGENT - SINGLE-LOOP WITH TOOLS")
    print("=" * 60)
    print()

    # Initialize agent
    agent = ProductionAgent(
        model_name="qwen3:8b",
        temperature=0,
        verbose=True
    )

    # Test 1: Calculator
    print("\n" + "=" * 60)
    print("TEST 1: Calculator Tool")
    print("=" * 60)

    result1 = agent.run("What is 15% of 2500?")
    print(f"\nFinal Answer: {result1['answer']}")
    print(f"Execution Time: {result1['execution_time']:.2f}s")
    print(f"Steps Taken: {len(result1.get('intermediate_steps', []))}")

    # Test 2: Weather
    print("\n" + "=" * 60)
    print("TEST 2: Weather Tool")
    print("=" * 60)

    result2 = agent.run("What's the weather like in Tokyo?")
    print(f"\nFinal Answer: {result2['answer']}")
    print(f"Execution Time: {result2['execution_time']:.2f}s")

    # Test 3: Web Search
    print("\n" + "=" * 60)
    print("TEST 3: Web Search Tool")
    print("=" * 60)

    result3 = agent.run("Who won the 2024 Nobel Prize in Physics?")
    print(f"\nFinal Answer: {result3['answer']}")
    print(f"Execution Time: {result3['execution_time']:.2f}s")

    # Test 4: Multi-turn conversation
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Turn Conversation (Memory)")
    print("=" * 60)

    agent.run("My name is Alice and I live in Paris")
    result4 = agent.run("What's the weather where I live?")
    print(f"\nFinal Answer: {result4['answer']}")
    print("(Agent should remember Paris from previous message)")

    # Display statistics
    print("\n" + "=" * 60)
    print("EXECUTION STATISTICS")
    print("=" * 60)

    stats = agent.get_execution_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
