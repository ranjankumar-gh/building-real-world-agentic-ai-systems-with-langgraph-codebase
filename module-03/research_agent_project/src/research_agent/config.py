"""
Configuration settings for the Research Agent.

Centralized configuration for LLM settings, search parameters,
and agent behavior.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentConfig:
    """
    Configuration for the research agent.

    Attributes:
        llm_model: Model to use (qwen3:8b, llama3, etc.)
        llm_temperature: Temperature for LLM calls (0 = deterministic)
        ollama_base_url: Base URL for Ollama server (default: http://localhost:11434)
        max_retries: Maximum retry attempts for failed operations
        search_limit: Number of searches to execute (max)
        min_valid_results: Minimum valid results needed to proceed
    """

    # LLM Settings
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    # Agent Behavior
    max_retries: int = 2
    search_limit: int = 3
    min_valid_results: int = 2

    def __post_init__(self):
        """Load Ollama base URL from environment if set."""
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            self.ollama_base_url = ollama_url
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Create config from environment variables.

        Environment variables:
            OLLAMA_BASE_URL: Optional (default: http://localhost:11434)
            LLM_MODEL: Optional (default: qwen3:8b)
            LLM_TEMPERATURE: Optional (default: 0.0)
            MAX_RETRIES: Optional (default: 2)
            SEARCH_LIMIT: Optional (default: 3)

        Returns:
            AgentConfig instance
        """
        return cls(
            llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            search_limit=int(os.getenv("SEARCH_LIMIT", "3")),
            min_valid_results=int(os.getenv("MIN_VALID_RESULTS", "2"))
        )


# Default configuration instance
DEFAULT_CONFIG = AgentConfig(
    llm_model="qwen3:8b",
    llm_temperature=0.0,
    ollama_base_url="http://localhost:11434",
    max_retries=2,
    search_limit=3,
    min_valid_results=2
)
