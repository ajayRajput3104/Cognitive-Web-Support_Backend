"""
Configuration Management (UPDATED 2024)
Loads and validates all environment variables and settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file in ROOT directory
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================================
# AI API KEYS
# ============================================================================

# Gemini AI (REQUIRED)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Google Search (Optional - for hybrid search before DuckDuckGo fallback)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# ============================================================================
# PERSISTENT STORAGE (2024 UPDATES)
# ============================================================================

# Pinecone Vector Database (NEW API - no environment parameter needed)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'cognitive-support')

# Render Key-Value Store (Redis/Valkey compatible - NO CHANGES NEEDED)
# Valkey is a drop-in replacement for Redis since Feb 2025
# Uses same redis:// URL scheme
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
    
    # Check Pinecone configuration (NEW API - no environment needed)
    if not PINECONE_API_KEY:
        warnings.append("PINECONE_API_KEY not set - will use in-memory storage (not recommended for production)")
    
    # Check Redis/Valkey configuration
    if REDIS_URL == 'redis://localhost:6379':
        warnings.append("Using default local Redis/Valkey - ensure it's running or set REDIS_URL")
    
    # Check Google Search keys (optional for hybrid search)
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        warnings.append("Google Search configured - will use as primary with DuckDuckGo fallback")
    else:
        warnings.append("Google Search not configured - will use DuckDuckGo only (unlimited, free)")
    
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
    print("CONFIGURATION SUMMARY (2024 UPDATE)")
    print("=" * 80)
    print(f"🔑 AI Provider: Gemini AI")
    print(f"🔍 Search: Hybrid (Google → DuckDuckGo fallback)")
    print(f"💾 Vector DB: {'Pinecone (NEW API 2024)' if PINECONE_API_KEY else 'In-Memory (not persistent)'}")
    print(f"⚡ Cache: Render Key-Value (Valkey/Redis compatible)")
    print(f"   URL: {REDIS_URL}")
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