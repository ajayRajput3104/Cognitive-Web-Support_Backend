"""
Rate Limiting Middleware
Prevents API abuse by limiting requests per IP address
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],
    storage_uri="memory://",  # Use in-memory storage for rate limits
    headers_enabled=True
)


def get_rate_limiter() -> Limiter:
    """
    Get the rate limiter instance
    
    Returns:
        Limiter instance
    """
    return limiter