"""
Unit tests for app.py - API endpoints
Tests FastAPI application and endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app import app


# Test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "test-api-key"


@pytest.fixture
def mock_brain():
    """Mock brain instance"""
    with patch('app.get_brain') as mock:
        instance = mock.return_value
        
        # Mock process_query
        instance.process_query = AsyncMock(return_value={
            "query": "test query",
            "answer": "test answer",
            "metadata": {"processing_time_seconds": 1.5},
            "deconstructed": {},
            "verified_url": {}
        })
        
        # Mock get_status
        instance.get_status.return_value = {
            "status": "online",
            "version": "2.0.0",
            "total_chunks": 100,
            "domains": []
        }
        
        # Mock health_check
        instance.health_check.return_value = {
            "healthy": True,
            "vector_store": "connected",
            "cache_manager": "connected"
        }
        
        # Mock clear_domain_cache
        instance.clear_domain_cache.return_value = {
            "status": "cleared",
            "domain": "example.com"
        }
        
        # Mock ingest_domain
        instance.ingest_domain = AsyncMock(return_value={
            "status": "success",
            "chunks_ingested": 50
        })
        
        yield instance


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Cognitive Web Support Engine API"
        assert data["version"] == "2.0.0"
    
    def test_health_endpoint_success(self, mock_brain):
        """Test health check success"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data


class TestQueryEndpoint:
    """Test query processing endpoint"""
    
    def test_query_without_auth(self):
        """Test query without authentication"""
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code == 403  # Forbidden
    
    def test_query_with_invalid_auth(self):
        """Test query with invalid API key"""
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401  # Unauthorized
    
    @patch('config.API_KEY', TEST_API_KEY)
    def test_query_success(self, mock_brain):
        """Test successful query processing"""
        response = client.post(
            "/api/query",
            json={"query": "How do I reset password?"},
            headers={"X-API-Key": TEST_API_KEY}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "metadata" in data
    
    @patch('config.API_KEY', TEST_API_KEY)
    def test_query_validation_error(self):
        """Test query with invalid data"""
        response = client.post(
            "/api/query",
            json={"query": ""},  # Too short
            headers={"X-API-Key": TEST_API_KEY}
        )
        
        assert response.status_code == 422  # Validation error


class TestStatusEndpoint:
    """Test status endpoint"""
    
    @patch('config.API_KEY', TEST_API_KEY)
    def test_status_success(self, mock_brain):
        """Test status retrieval"""
        response = client.get(
            "/api/status",
            headers={"X-API-Key": TEST_API_KEY}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "total_chunks" in data


class TestCacheEndpoint:
    """Test cache management endpoint"""
    
    @patch('config.API_KEY', TEST_API_KEY)
    def test_clear_cache_success(self, mock_brain):
        """Test cache clearing"""
        response = client.delete(
            "/api/cache/example.com",
            headers={"X-API-Key": TEST_API_KEY}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"