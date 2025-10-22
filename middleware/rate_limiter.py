"""
Simple Rate Limiting Middleware (Fixed for FastAPI compatibility)
Using Python dictionary instead of slowapi to avoid middleware conflicts
"""

from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleRateLimiter:
    """
    Simple in-memory rate limiter without external dependencies
    Tracks requests per IP address with sliding window
    """
    
    def __init__(self):
        # Store: {ip: [(timestamp, count), ...]}
        self.requests: Dict[str, list] = defaultdict(list)
        
    def check_rate_limit(
        self, 
        request: Request, 
        max_requests: int = 10, 
        window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Args:
            request: FastAPI request object
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, requests_remaining)
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Current time
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        # Clean old requests
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip] 
            if ts > cutoff
        ]
        
        # Check limit
        current_count = len(self.requests[client_ip])
        
        if current_count >= max_requests:
            remaining = 0
            return False, remaining
        
        # Add current request
        self.requests[client_ip].append(now)
        remaining = max_requests - current_count - 1
        
        return True, remaining
    
    def cleanup_old_entries(self, max_age_hours: int = 24):
        """Cleanup old entries to prevent memory bloat"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                ts for ts in self.requests[ip] 
                if ts > cutoff
            ]
            
            # Remove empty entries
            if not self.requests[ip]:
                del self.requests[ip]


# Global rate limiter instance
rate_limiter = SimpleRateLimiter()


def check_rate_limit(request: Request, max_requests: int = 10, window_seconds: int = 60):
    """
    Decorator-style rate limit checker
    
    Usage in endpoints:
        check_rate_limit(request, max_requests=10, window_seconds=60)
    
    Raises:
        HTTPException: If rate limit exceeded
    """
    allowed, remaining = rate_limiter.check_rate_limit(
        request, 
        max_requests=max_requests, 
        window_seconds=window_seconds
    )
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.",
            headers={
                "Retry-After": str(window_seconds),
                "X-RateLimit-Limit": str(max_requests),
                "X-RateLimit-Remaining": "0"
            }
        )
    
    # Return remaining count for logging
    return remaining


def get_rate_limiter() -> SimpleRateLimiter:
    """Get the rate limiter instance"""
    return rate_limiter