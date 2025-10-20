"""
Unit tests for AI agents (Stage 1-4)
Tests each agent's functionality independently
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

# Stage 1 tests
class TestStage1Deconstructor:
    """Test query deconstruction agent"""
    
    @pytest.mark.asyncio
    async def test_deconstruct_query_success(self):
        """Test successful query deconstruction"""
        from ai_agents.stage1_deconstructor import deconstruct_query
        
        with patch('ai_agents.stage1_deconstructor.httpx.AsyncClient') as mock_client:
            # Mock Gemini API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"user_intent": "reset password", "identified_entity": "GitHub", "specific_details": ["password"], "inhibitor": "forgot password"}'
                        }]
                    }
                }]
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await deconstruct_query("How do I reset my GitHub password?")
            
            assert result.identified_entity == "GitHub"
            assert result.user_intent == "reset password"
            assert "password" in result.specific_details
    
    @pytest.mark.asyncio
    async def test_deconstruct_query_fallback(self):
        """Test fallback when API fails"""
        from ai_agents.stage1_deconstructor import deconstruct_query
        
        with patch('ai_agents.stage1_deconstructor.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=Exception("API Error"))
            
            result = await deconstruct_query("Test query about GitHub")
            
            # Should use fallback
            assert result.identified_entity is not None
            assert result.user_intent is not None


# Stage 2 tests
class TestStage2SearchVerify:
    """Test search and verification agent"""
    
    @pytest.mark.asyncio
    async def test_search_web_duckduckgo(self):
        """Test DuckDuckGo search"""
        from ai_agents.stage2_search_verify import search_web
        
        with patch('ai_agents.stage2_search_verify.DDGS') as mock_ddgs:
            mock_ddgs.return_value.text.return_value = [
                {
                    'title': 'GitHub Help',
                    'href': 'https://help.github.com',
                    'body': 'Official GitHub documentation'
                }
            ]
            
            results = await search_web("GitHub password reset")
            
            assert len(results) > 0
            assert results[0].url == 'https://help.github.com'
    
    @pytest.mark.asyncio
    async def test_verify_url_success(self):
        """Test URL verification"""
        from ai_agents.stage2_search_verify import verify_and_select_url, SearchResult
        from ai_agents.stage1_deconstructor import DeconstructedQuery
        
        deconstructed = DeconstructedQuery(
            user_intent="reset password",
            identified_entity="GitHub",
            specific_details=["password"],
            inhibitor="forgot"
        )
        
        search_results = [
            SearchResult(
                title="GitHub Help",
                url="https://help.github.com/password",
                snippet="Official password help"
            )
        ]
        
        with patch('ai_agents.stage2_search_verify.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"seed_url": "https://help.github.com/password", "reasoning": "Official source"}'
                        }]
                    }
                }]
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await verify_and_select_url(deconstructed, search_results)
            
            assert "github.com" in result.seed_url
            assert result.reasoning is not None


# Stage 3 tests
class TestStage3Crawler:
    """Test web crawler agent"""
    
    @pytest.mark.asyncio
    async def test_scrape_page_success(self):
        """Test successful page scraping"""
        from ai_agents.stage3_crawler import scrape_page
        
        mock_html = """
        <html>
            <body>
                <p>This is test content about password reset.</p>
                <a href="/help">Help link</a>
            </body>
        </html>
        """
        
        with patch('ai_agents.stage3_crawler.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_html
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            text, links = await scrape_page("https://example.com")
            
            assert "test content" in text.lower()
            assert len(links) >= 0
    
    def test_chunk_text(self):
        """Test text chunking"""
        from ai_agents.stage3_crawler import chunk_text
        
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
    
    def test_score_link_relevance(self):
        """Test link relevance scoring"""
        from ai_agents.stage3_crawler import score_link_relevance
        
        # Relevant link
        score1 = score_link_relevance(
            "https://help.github.com/password",
            "password reset",
            ["password", "reset"]
        )
        
        # Irrelevant link
        score2 = score_link_relevance(
            "https://example.com/login",
            "login",
            ["password", "reset"]
        )
        
        assert score1 > score2


# Stage 4 tests
class TestStage4Synthesis:
    """Test answer synthesis agent"""
    
    @pytest.mark.asyncio
    async def test_synthesize_answer_success(self):
        """Test successful answer synthesis"""
        from ai_agents.stage4_synthesis import synthesize_answer
        from ai_agents.stage1_deconstructor import DeconstructedQuery
        
        deconstructed = DeconstructedQuery(
            user_intent="reset password",
            identified_entity="GitHub",
            specific_details=["password"],
            inhibitor="forgot"
        )
        
        relevant_chunks = [
            {
                'text': 'To reset your password, click forgot password...',
                'url': 'https://help.github.com/password',
                'relevance_score': 0.95
            }
        ]
        
        with patch('ai_agents.stage4_synthesis.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "Here's how to reset your GitHub password: Click on forgot password..."
                        }]
                    }
                }]
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            answer = await synthesize_answer("How do I reset password?", deconstructed, relevant_chunks)
            
            assert len(answer) > 0
            assert "password" in answer.lower()
            assert "https://help.github.com/password" in answer
    
    @pytest.mark.asyncio
    async def test_synthesize_answer_no_chunks(self):
        """Test synthesis with no chunks"""
        from ai_agents.stage4_synthesis import synthesize_answer
        from ai_agents.stage1_deconstructor import DeconstructedQuery
        
        deconstructed = DeconstructedQuery(
            user_intent="test",
            identified_entity="Test",
            specific_details=[],
            inhibitor="none"
        )
        
        answer = await synthesize_answer("Test query", deconstructed, [])
        
        assert "couldn't find" in answer.lower() or "no" in answer.lower()