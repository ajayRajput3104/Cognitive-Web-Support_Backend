"""
FastAPI Application - FIXED FOR NEW RESPONSE STRUCTURE
Compatible with enhanced brain.py response format
"""

from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from brain import get_brain
from config import ALLOWED_ORIGINS, PORT, LOG_LEVEL
from middleware.auth import verify_api_key
from middleware.rate_limiter import limiter

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title="Cognitive Web Support Engine API",
    description="Production-ready multi-agent AI system for intelligent web support",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter state
app.state.limiter = limiter

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# UPDATED REQUEST/RESPONSE MODELS (COMPATIBLE WITH NEW BRAIN.PY)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., min_length=5, max_length=500)
    force_refresh: Optional[bool] = Field(False)


class QueryResponse(BaseModel):
    """UPDATED Response model - matches new brain.py structure"""
    success: bool
    message: str
    query: str
    answer: Optional[str] = None
    metadata: Dict[str, Any]
    deconstructed: Dict[str, Any]
    verified_url: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    suggestion: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for structured error responses"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
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
        "version": "2.0.1",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        brain = get_brain()
        system_status = brain.health_check()
        
        return {
            "status": "healthy",
            "service": "Cognitive Web Support Engine API",
            "version": "2.0.1",
            "timestamp": datetime.now().isoformat(),
            "system": system_status
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


@app.post("/api/query")
@limiter.limit("10/minute")
async def process_query(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Main endpoint: Process user query through 4-stage AI pipeline
    
    UPDATED: Returns enhanced response structure with success flag
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing query: {query_request.query[:100]}")
        
        brain = get_brain()
        result = await brain.process_query(
            query_request.query, 
            query_request.force_refresh
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f}s - Success: {result.get('success', False)}")
        
        # Return result as-is (already has correct structure from brain.py)
        return result
        
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        
        # Return error in new structure format
        return {
            "success": False,
            "message": f"Query processing failed: {str(e)}",
            "query": query_request.query,
            "error": str(e),
            "suggestion": "Try again or use force_refresh=true",
            "metadata": {
                "status": "error",
                "timestamp": datetime.now().isoformat()
            },
            "deconstructed": {
                "user_intent": "unknown",
                "identified_entity": "unknown",
                "specific_details": [],
                "inhibitor": "none"
            }
        }


@app.post("/api/ingest")
@limiter.limit("5/minute")
async def ingest_domain(
    request: Request,
    url: str,
    api_key: str = Security(verify_api_key)
):
    """
    Manually ingest a domain for pre-caching
    """
    try:
        logger.info(f"Manual ingestion requested for: {url}")
        
        brain = get_brain()
        result = await brain.ingest_domain(url)
        
        logger.info(f"Ingestion complete: {result.get('status')}")
        return result
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return {
            "status": "error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/status")
async def get_status(api_key: str = Security(verify_api_key)):
    """
    Get system status and cached domains info
    """
    try:
        brain = get_brain()
        status_data = brain.get_status()
        
        return {
            **status_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status retrieval error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.delete("/api/cache/{domain}")
@limiter.limit("20/minute")
async def clear_cache(
    request: Request,
    domain: str,
    api_key: str = Security(verify_api_key)
):
    """
    Clear cache for a specific domain
    """
    try:
        logger.info(f"Cache clear requested for: {domain}")
        
        brain = get_brain()
        result = brain.clear_domain_cache(domain)
        
        logger.info(f"Cache cleared: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}", exc_info=True)
        return {
            "status": "error",
            "domain": domain,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Starting Cognitive Web Support Engine v2.0.1")
    logger.info("=" * 80)
    
    try:
        brain = get_brain()
        logger.info("✅ Brain initialized successfully")
        logger.info(f"📊 Port: {PORT}")
        logger.info(f"🔒 Authentication: Enabled")
        logger.info(f"⚡ Rate Limiting: Enabled")
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
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting server on port {PORT}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )