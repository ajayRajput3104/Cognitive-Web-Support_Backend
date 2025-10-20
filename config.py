"""
Configuration Management
Loads and validates all environment variables and settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================================
# AI API KEYS
# ============================================================================

# Gemini AI (REQUIRED)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Google Search (Optional - only if USE_FREE_SEARCH=false)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# ============================================================================
# PERSISTENT STORAGE
# ============================================================================

# Pinecone Vector Database (REQUIRED for production)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'cognitive-support')

# Redis Cache (REQUIRED for production)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# ============================================================================
# SECURITY
# ============================================================================

# API Key for authentication
API_KEY = os.getenv('API_KEY', 'dev-api-key-change-in-production')

# CORS Configuration
ALLOWED_ORIGINS_STR = os.getenv('ALLOWED_ORIGINS', '*')
ALLOWED_ORIGINS: List[str] = (
    ['*'] if ALLOWED_ORIGINS_STR == '*' 
    else [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(',')]
)

# Rate Limiting
RATE_LIMIT = os.getenv('RATE_LIMIT', '10/minute')

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

# Server Configuration
PORT = int(os.getenv('PORT', 8000))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Cache Settings
CACHE_DURATION_HOURS = int(os.getenv('CACHE_DURATION_HOURS', 24))

# Crawling Settings
MAX_PAGES_TO_CRAWL = int(os.getenv('MAX_PAGES_TO_CRAWL', 10))

# Text Processing
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

# RAG Settings
TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', 5))

# Embedding Model
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Use free DuckDuckGo search instead of Google
USE_FREE_SEARCH = os.getenv('USE_FREE_SEARCH', 'true').lower() == 'true'

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validates that all required configuration is present
    Raises ValueError if required settings are missing
    """
    errors = []
    warnings = []
    
    # Check required AI keys
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required")
    
    # Check Pinecone configuration
    if not PINECONE_API_KEY:
        warnings.append("PINECONE_API_KEY not set - will use in-memory storage (not recommended for production)")
    
    if not PINECONE_ENVIRONMENT:
        warnings.append("PINECONE_ENVIRONMENT not set - using default 'us-east-1-aws'")
    
    # Check Redis configuration
    if REDIS_URL == 'redis://localhost:6379':
        warnings.append("Using default local Redis - ensure Redis is running or set REDIS_URL")
    
    # Check Google Search keys (only if not using free search)
    if not USE_FREE_SEARCH:
        if not GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY required when USE_FREE_SEARCH=false")
        if not GOOGLE_CSE_ID:
            errors.append("GOOGLE_CSE_ID required when USE_FREE_SEARCH=false")
    
    # Check security
    if API_KEY == 'dev-api-key-change-in-production':
        warnings.append("Using default API_KEY - CHANGE THIS IN PRODUCTION!")
    
    # Check CORS
    if '*' in ALLOWED_ORIGINS:
        warnings.append("CORS allows all origins (*) - restrict in production")
    
    # Print results
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"  ❌ {e}" for e in errors)
        raise ValueError(error_msg)
    
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    print("✅ Configuration validated successfully")
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"🔑 AI Provider: Gemini AI")
    print(f"🔍 Search: {'DuckDuckGo (Free)' if USE_FREE_SEARCH else 'Google Custom Search'}")
    print(f"💾 Vector DB: {'Pinecone' if PINECONE_API_KEY else 'In-Memory (not persistent)'}")
    print(f"⚡ Cache: Redis at {REDIS_URL}")
    print(f"🔒 Authentication: {'Enabled' if API_KEY != 'dev-api-key-change-in-production' else 'Development Mode'}")
    print(f"🌐 Port: {PORT}")
    print(f"📊 Log Level: {LOG_LEVEL}")
    print(f"⏰ Cache Duration: {CACHE_DURATION_HOURS} hours")
    print(f"📄 Max Pages to Crawl: {MAX_PAGES_TO_CRAWL}")
    print(f"🧩 Top K Chunks: {TOP_K_CHUNKS}")
    print("=" * 80 + "\n")


# ============================================================================
# INITIALIZATION
# ============================================================================

# Validate configuration on module import
try:
    validate_config()
except ValueError as e:
    print(f"\n❌ CONFIGURATION ERROR:\n{e}\n")
    print("Please set required environment variables in .env file")
    print("See .env.example for template\n")
    raise