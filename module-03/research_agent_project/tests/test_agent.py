"""
Unit tests for the Research Agent.

Run with: pytest tests/test_agent.py
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import (
    ResearchAgentState,
    create_initial_state,
    create_research_agent,
    AgentConfig
)


class TestStateCreation:
    """Test state initialization."""
    
    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state("Test query")
        
        assert state["research_query"] == "Test query"
        assert state["current_stage"] == "planning"
        assert state["retry_count"] == 0
        assert state["max_retries"] == 2
        assert state["messages"] == []
        assert state["search_queries"] == []
    
    def test_create_initial_state_with_retries(self):
        """Test creating state with custom retries."""
        state = create_initial_state("Test query", max_retries=5)
        
        assert state["max_retries"] == 5


class TestConfiguration:
    """Test agent configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig(openai_api_key="test-key")
        
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.0
        assert config.max_retries == 2
        assert config.search_limit == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.7,
            max_retries=5,
            openai_api_key="test-key"
        )
        
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.llm_temperature == 0.7
        assert config.max_retries == 5


class TestAgentCreation:
    """Test agent creation."""
    
    def test_create_agent(self):
        """Test creating agent."""
        # Set dummy API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        agent = create_research_agent()
        assert agent is not None
    
    def test_agent_has_nodes(self):
        """Test that agent has all required nodes."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        agent = create_research_agent()
        
        # Agent should be compiled and ready
        assert hasattr(agent, 'invoke')
        assert hasattr(agent, 'stream')
        assert hasattr(agent, 'get_state')


class TestRouting:
    """Test routing logic."""
    
    def test_validation_routing_success(self):
        """Test routing after successful validation."""
        from research_agent.graph import route_after_validation
        
        state = create_initial_state("test")
        state["current_stage"] = "processing"
        
        next_node = route_after_validation(state)
        assert next_node == "process"
    
    def test_validation_routing_error(self):
        """Test routing after failed validation."""
        from research_agent.graph import route_after_validation
        
        state = create_initial_state("test")
        state["current_stage"] = "error"
        
        next_node = route_after_validation(state)
        assert next_node == "handle_error"
    
    def test_error_routing_retry(self):
        """Test routing for retry."""
        from research_agent.graph import route_after_error
        
        state = create_initial_state("test")
        state["current_stage"] = "searching"
        
        next_node = route_after_error(state)
        assert next_node == "search"
    
    def test_error_routing_give_up(self):
        """Test routing when giving up."""
        from research_agent.graph import route_after_error
        
        state = create_initial_state("test")
        state["current_stage"] = "complete"
        
        next_node = route_after_error(state)
        assert next_node == "end"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
