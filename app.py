"""
FastAPI Application - MEMORY OPTIMIZED for Render Free Tier (512MB)
Production-Ready API Layer with memory management
VERSION: Fixed rate limiting (no slowapi dependency)
"""

from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime
import gc

from brain import get_brain
from config import ALLOWED_ORIGINS, PORT, LOG_LEVEL
from middleware.auth import verify_api_key
from middleware.rate_limiter import check_rate_limit  # Simple rate limiter

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Cognitive Web Support Engine API",
    description="Memory-optimized multi-agent AI system for intelligent web support",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., min_length=5, max_length=500, description="User's question")
    force_refresh: Optional[bool] = Field(False, description="Force re-crawl even if cached")


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str
    answer: str
    metadata: dict
    deconstructed: dict
    verified_url: dict


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================================================================
# MEMORY MONITORING
# ============================================================================

def log_memory_usage(context: str = ""):
    """Log current memory usage"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 Memory [{context}]: {mem_mb:.1f} MB")
        
        # Warning if getting close to limit
        if mem_mb > 400:
            logger.warning(f"⚠️  High memory usage: {mem_mb:.1f} MB / 512 MB limit")
        
        return mem_mb
    except ImportError:
        return 0


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for structured error responses"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Cleanup memory on error
    gc.collect()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cognitive Web Support Engine API",
        "version": "2.0.1 (Memory Optimized)",
        "status": "online",
        "docs": "/docs",
        "optimization": "Render Free Tier (512MB)"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with memory monitoring"""
    try:
        brain = get_brain()
        system_status = brain.health_check()
        
        # Log memory usage
        mem_usage = log_memory_usage("health_check")
        
        return {
            "status": "healthy",
            "service": "Cognitive Web Support Engine API",
            "version": "2.0.1",
            "timestamp": datetime.now().isoformat(),
            "system": system_status,
            "memory_mb": round(mem_usage, 1),
            "memory_limit_mb": 512
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Main endpoint: Process user query through 4-stage AI pipeline
    
    **Authentication Required:** X-API-Key header
    **Rate Limited:** 10 requests per minute
    
    Args:
        query_request: Query details including user question
        
    Returns:
        Complete response with answer, sources, and metadata
        
    Raises:
        HTTPException: If processing fails
    """
    # Apply rate limiting
    check_rate_limit(request, max_requests=10, window_seconds=60)
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Processing query: {query_request.query[:100]}")
        log_memory_usage("query_start")
        
        brain = get_brain()
        result = await brain.process_query(
            query_request.query, 
            query_request.force_refresh
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
        
        # Memory cleanup after processing
        gc.collect()
        log_memory_usage("query_end")
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        gc.collect()  # Cleanup on error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/api/ingest")
async def ingest_domain(
    request: Request,
    url: str,
    api_key: str = Security(verify_api_key)
):
    """
    Manually ingest a domain for pre-caching
    
    **Authentication Required:** X-API-Key header
    **Rate Limited:** 3 requests per minute
    
    Args:
        url: Domain URL to ingest (query parameter)
        
    Returns:
        Ingestion results with chunks count
        
    Raises:
        HTTPException: If ingestion fails
    """
    # Apply rate limiting (stricter for ingestion)
    check_rate_limit(request, max_requests=3, window_seconds=60)
    
    try:
        logger.info(f"📥 Manual ingestion requested for: {url}")
        log_memory_usage("ingest_start")
        
        brain = get_brain()
        result = await brain.ingest_domain(url)
        
        logger.info(f"✅ Ingestion complete: {result.get('chunks_ingested', 0)} chunks")
        
        # Aggressive cleanup after ingestion
        gc.collect()
        log_memory_usage("ingest_end")
        
        return result
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        gc.collect()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest domain: {str(e)}"
        )


@app.get("/api/status")
async def get_status(api_key: str = Security(verify_api_key)):
    """
    Get system status and cached domains info
    
    **Authentication Required:** X-API-Key header
    
    Returns:
        System statistics, cached domains, and health metrics
    """
    try:
        brain = get_brain()
        status_data = brain.get_status()
        
        # Add memory info
        mem_usage = log_memory_usage("status")
        
        return {
            **status_data,
            "memory_mb": round(mem_usage, 1),
            "memory_limit_mb": 512,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve status: {str(e)}"
        )


@app.delete("/api/cache/{domain}")
async def clear_cache(
    request: Request,
    domain: str,
    api_key: str = Security(verify_api_key)
):
    """
    Clear cache for a specific domain
    
    **Authentication Required:** X-API-Key header
    **Rate Limited:** 20 requests per minute
    
    Args:
        domain: Domain name to clear (path parameter)
        
    Returns:
        Status of cache clearing operation
    """
    # Apply rate limiting
    check_rate_limit(request, max_requests=20, window_seconds=60)
    
    try:
        logger.info(f"🗑️  Cache clear requested for: {domain}")
        
        brain = get_brain()
        result = brain.clear_domain_cache(domain)
        
        logger.info(f"✅ Cache cleared: {result}")
        
        # Cleanup after clearing
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Starting Cognitive Web Support Engine v2.0.1 (Memory Optimized)")
    logger.info("=" * 80)
    
    try:
        # Initialize brain (model will load lazily on first use)
        brain = get_brain()
        logger.info("✅ Brain initialized successfully")
        logger.info(f"📊 Port: {PORT}")
        logger.info(f"🔒 Authentication: Enabled")
        logger.info(f"⚡ Rate Limiting: Enabled (Simple)")
        logger.info(f"💾 Memory Optimization: Active (Render Free Tier)")
        
        # Log initial memory
        log_memory_usage("startup")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Cognitive Web Support Engine")
    try:
        brain = get_brain()
        brain.cleanup()
        
        # Final memory cleanup
        gc.collect()
        
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)


# ============================================================================
# MAIN ENTRY POINT (for local development only)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # CRITICAL: Use PORT from environment (Render sets this dynamically)
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level=LOG_LEVEL.lower(),
        workers=1  # Single worker for memory efficiency
    )