"""
Unit tests for brain.py - Central Orchestrator
Tests the main cognitive pipeline and business logic
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from brain import CognitiveBrain, get_brain


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore"""
    with patch('brain.VectorStore') as mock:
        instance = mock.return_value
        instance.retrieve_relevant.return_value = [
            {
                'text': 'Sample text chunk',
                'url': 'https://example.com/doc',
                'relevance_score': 0.95
            }
        ]
        instance.get_domain_stats.return_value = {'chunks': 10, 'exists': True}
        instance.get_all_domains.return_value = ['example.com']
        instance.get_total_chunks.return_value = 10
        instance.health_check.return_value = True
        yield instance


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager"""
    with patch('brain.CacheManager') as mock:
        instance = mock.return_value
        instance.is_cached.return_value = True
        instance.get_cache_info.return_value = {
            'cached_at': '2024-01-01T00:00:00',
            'is_valid': True,
            'expires_in_hours': 20
        }
        instance.get_all_cached_domains.return_value = []
        instance.health_check.return_value = True
        yield instance


@pytest.fixture
def brain(mock_vector_store, mock_cache_manager):
    """Create Brain instance with mocked dependencies"""
    return CognitiveBrain()


class TestCognitiveBrain:
    """Test suite for CognitiveBrain class"""
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, brain):
        """Test successful query processing"""
        # Mock all agent functions
        with patch('brain.deconstruct_query', new_callable=AsyncMock) as mock_decon, \
             patch('brain.search_web', new_callable=AsyncMock) as mock_search, \
             patch('brain.verify_and_select_url', new_callable=AsyncMock) as mock_verify, \
             patch('brain.targeted_crawl', new_callable=AsyncMock) as mock_crawl, \
             patch('brain.synthesize_answer', new_callable=AsyncMock) as mock_synth:
            
            # Setup mocks
            from ai_agents.stage1_deconstructor import DeconstructedQuery
            from ai_agents.stage2_search_verify import VerifiedURL
            
            mock_decon.return_value = DeconstructedQuery(
                user_intent="reset password",
                identified_entity="GitHub",
                specific_details=["password", "reset"],
                inhibitor="forgot password"
            )
            
            mock_search.return_value = [Mock(url="https://github.com/help")]
            
            mock_verify.return_value = VerifiedURL(
                seed_url="https://github.com/help",
                reasoning="Official help page"
            )
            
            mock_crawl.return_value = []
            mock_synth.return_value = "Here's how to reset your password..."
            
            # Execute
            result = await brain.process_query("How do I reset my GitHub password?")
            
            # Verify
            assert "query" in result
            assert "answer" in result
            assert "metadata" in result
            assert result["answer"] == "Here's how to reset your password..."
    
    @pytest.mark.asyncio
    async def test_process_query_no_search_results(self, brain):
        """Test query processing with no search results"""
        with patch('brain.deconstruct_query', new_callable=AsyncMock) as mock_decon, \
             patch('brain.search_web', new_callable=AsyncMock) as mock_search:
            
            from ai_agents.stage1_deconstructor import DeconstructedQuery
            
            mock_decon.return_value = DeconstructedQuery(
                user_intent="test",
                identified_entity="Unknown",
                specific_details=[],
                inhibitor="none"
            )
            
            mock_search.return_value = []
            
            result = await brain.process_query("Invalid query")
            
            assert "error" in result
            assert result["error"] == "No search results found"
    
    def test_get_status(self, brain):
        """Test status retrieval"""
        status = brain.get_status()
        
        assert "status" in status
        assert status["status"] == "online"
        assert "version" in status
        assert "total_chunks" in status
        assert "domains" in status
    
    def test_clear_domain_cache(self, brain):
        """Test cache clearing"""
        result = brain.clear_domain_cache("example.com")
        
        assert "status" in result
        assert "domain" in result
        assert result["domain"] == "example.com"
    
    def test_health_check(self, brain):
        """Test health check"""
        health = brain.health_check()
        
        assert "healthy" in health
        assert health["healthy"] is True
        assert "vector_store" in health
        assert "cache_manager" in health
    
    def test_singleton_pattern(self):
        """Test that get_brain() returns singleton"""
        brain1 = get_brain()
        brain2 = get_brain()
        
        assert brain1 is brain2