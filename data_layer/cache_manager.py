"""
Cache Manager - Persistent Redis-based caching
Handles domain-level caching with timestamp tracking
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
import redis
import json

from config import CACHE_DURATION_HOURS, REDIS_URL

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages domain-level caching with Redis for persistence
    Cache survives server restarts and is shared across instances
    """
    
    def __init__(self, cache_duration_hours: int = CACHE_DURATION_HOURS):
        """
        Initialize cache manager with Redis
        
        Args:
            cache_duration_hours: How long to cache domain data
        """
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logger.info(f"✓ Connected to Redis at {REDIS_URL}")
            
        except Exception as e:
            logger.warning(f"⚠ Redis connection failed: {e}")
            logger.warning("⚠ Falling back to in-memory cache")
            self.cache = {}  # Fallback to in-memory
            self.use_redis = False
    
    def is_cached(self, domain: str, force_refresh: bool = False) -> bool:
        """
        Check if domain is cached and cache is still valid
        
        Args:
            domain: Domain name
            force_refresh: Force refresh even if cached
            
        Returns:
            True if domain is cached and valid, False otherwise
        """
        if force_refresh:
            return False
        
        try:
            if self.use_redis:
                # Get from Redis
                cached_time_str = self.redis_client.get(f"cache:{domain}")
                if not cached_time_str:
                    return False
                
                cache_time = datetime.fromisoformat(cached_time_str)
                age = datetime.now() - cache_time
                return age < self.cache_duration
            else:
                # In-memory fallback
                if domain not in self.cache:
                    return False
                
                cache_time = self.cache[domain]
                age = datetime.now() - cache_time
                return age < self.cache_duration
                
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
            return False
    
    def mark_cached(self, domain: str):
        """
        Mark a domain as cached with current timestamp
        
        Args:
            domain: Domain name
        """
        try:
            current_time = datetime.now()
            
            if self.use_redis:
                # Store in Redis with expiration
                self.redis_client.set(
                    f"cache:{domain}",
                    current_time.isoformat(),
                    ex=int(self.cache_duration.total_seconds())
                )
                logger.info(f"✓ Marked {domain} as cached in Redis")
            else:
                # In-memory fallback
                self.cache[domain] = current_time
                logger.info(f"✓ Marked {domain} as cached in memory")
                
        except Exception as e:
            logger.error(f"Failed to mark domain as cached: {e}")
    
    def get_cache_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get cache information for a domain
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with cache info or None if not cached
        """
        try:
            if self.use_redis:
                cached_time_str = self.redis_client.get(f"cache:{domain}")
                if not cached_time_str:
                    return None
                
                cache_time = datetime.fromisoformat(cached_time_str)
            else:
                if domain not in self.cache:
                    return None
                cache_time = self.cache[domain]
            
            age = datetime.now() - cache_time
            is_valid = age < self.cache_duration
            
            return {
                'domain': domain,
                'cached_at': cache_time.isoformat(),
                'age_hours': age.total_seconds() / 3600,
                'is_valid': is_valid,
                'expires_in_hours': (self.cache_duration - age).total_seconds() / 3600 if is_valid else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return None
    
    def clear_domain(self, domain: str) -> bool:
        """
        Clear cache for a specific domain
        
        Args:
            domain: Domain name
            
        Returns:
            True if domain was cached, False otherwise
        """
        try:
            if self.use_redis:
                result = self.redis_client.delete(f"cache:{domain}")
                logger.info(f"✓ Cleared cache for {domain}")
                return result > 0
            else:
                if domain in self.cache:
                    del self.cache[domain]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_all_cached_domains(self) -> List[Dict[str, Any]]:
        """
        Get information about all cached domains
        
        Returns:
            List of dictionaries with cache info for each domain
        """
        try:
            domains = []
            
            if self.use_redis:
                # Get all cache keys
                keys = self.redis_client.keys("cache:*")
                for key in keys:
                    domain = key.replace("cache:", "")
                    info = self.get_cache_info(domain)
                    if info:
                        domains.append(info)
            else:
                for domain in self.cache.keys():
                    info = self.get_cache_info(domain)
                    if info:
                        domains.append(info)
            
            return domains
            
        except Exception as e:
            logger.error(f"Failed to get cached domains: {e}")
            return []
    
    def clear_expired(self) -> int:
        """
        Remove all expired cache entries
        
        Returns:
            Number of expired entries removed
        """
        try:
            if self.use_redis:
                # Redis handles expiration automatically
                return 0
            else:
                now = datetime.now()
                expired_domains = [
                    domain for domain, timestamp in self.cache.items()
                    if now - timestamp >= self.cache_duration
                ]
                
                for domain in expired_domains:
                    del self.cache[domain]
                
                return len(expired_domains)
                
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0
    
    def health_check(self) -> bool:
        """
        Check if cache is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if self.use_redis:
                self.redis_client.ping()
            return True
        except:
            return False
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            if self.use_redis:
                self.redis_client.close()
                logger.info("✓ Redis connection closed")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")