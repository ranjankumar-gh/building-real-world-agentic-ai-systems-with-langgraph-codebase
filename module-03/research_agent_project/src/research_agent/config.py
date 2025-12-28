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
        llm_model: Model to use (gpt-4, claude-sonnet-4, etc.)
        llm_temperature: Temperature for LLM calls (0 = deterministic)
        max_retries: Maximum retry attempts for failed operations
        search_limit: Number of searches to execute (max)
        min_valid_results: Minimum valid results needed to proceed
        openai_api_key: OpenAI API key (from environment)
    """
    
    # LLM Settings
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    
    # Agent Behavior
    max_retries: int = 2
    search_limit: int = 3
    min_valid_results: int = 2
    
    # API Keys
    openai_api_key: str | None = None
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass to config."
            )
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Create config from environment variables.
        
        Environment variables:
            OPENAI_API_KEY: Required
            LLM_MODEL: Optional (default: gpt-4)
            LLM_TEMPERATURE: Optional (default: 0.0)
            MAX_RETRIES: Optional (default: 2)
            SEARCH_LIMIT: Optional (default: 3)
        
        Returns:
            AgentConfig instance
        """
        return cls(
            llm_model=os.getenv("LLM_MODEL", "gpt-4"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            search_limit=int(os.getenv("SEARCH_LIMIT", "3")),
            min_valid_results=int(os.getenv("MIN_VALID_RESULTS", "2")),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )


# Default configuration instance
DEFAULT_CONFIG = AgentConfig(
    llm_model="gpt-4",
    llm_temperature=0.0,
    max_retries=2,
    search_limit=3,
    min_valid_results=2
)
