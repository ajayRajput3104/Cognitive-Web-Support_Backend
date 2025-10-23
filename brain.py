"""
The Brain - Central Orchestrator for Multi-Agent AI System
ENHANCED: Better response quality, smart re-crawling, improved answers
"""

from datetime import datetime
from urllib.parse import urlparse
from typing import Optional, Dict, Any
import logging

from ai_agents.stage1_deconstructor import deconstruct_query, DeconstructedQuery
from ai_agents.stage2_search_verify import search_web, verify_and_select_url, VerifiedURL
from ai_agents.stage3_crawler import targeted_crawl
from ai_agents.stage4_synthesis import synthesize_answer
from data_layer.vector_store import VectorStore
from data_layer.cache_manager import CacheManager
from config import TOP_K_CHUNKS

logger = logging.getLogger(__name__)


class CognitiveBrain:
    """
    Enhanced central orchestrator with smart re-crawling and better responses
    """
    
    def __init__(self):
        """Initialize the brain with persistent storage backends"""
        try:
            logger.info("🧠 Initializing Cognitive Brain...")
            
            self.vector_store = VectorStore()
            logger.info("✅ Vector store (Pinecone) connected")
            
            self.cache_manager = CacheManager()
            logger.info("✅ Cache manager (Redis) connected")
            
            logger.info("🧠 Cognitive Brain fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Brain initialization failed: {e}", exc_info=True)
            raise
    
    async def process_query(
        self, 
        query: str, 
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced pipeline with smart re-crawling when needed
        """
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info(f"🔍 Processing Query: {query}")
        logger.info("=" * 80)
        
        try:
            # =================================================================
            # STAGE 1: DECONSTRUCT QUERY
            # =================================================================
            logger.info("📋 Stage 1: Deconstructing query...")
            deconstructed = await deconstruct_query(query)
            logger.info(f"   ✓ Entity: {deconstructed.identified_entity}")
            logger.info(f"   ✓ Intent: {deconstructed.user_intent}")
            logger.info(f"   ✓ Details: {', '.join(deconstructed.specific_details[:3])}")
            
            # =================================================================
            # STAGE 2: SEARCH & VERIFY URL
            # =================================================================
            logger.info("\n🔎 Stage 2: Searching and verifying URL...")
            search_query = f"{deconstructed.identified_entity} {deconstructed.user_intent} {' '.join(deconstructed.specific_details[:3])}"
            search_results = await search_web(search_query)
            
            if not search_results:
                logger.warning("No search results found")
                return self._error_response(
                    query=query,
                    deconstructed=deconstructed,
                    error="No search results found",
                    suggestion="Try rephrasing your query or checking if the service name is correct"
                )
            
            verified_url = await verify_and_select_url(deconstructed, search_results)
            domain = urlparse(verified_url.seed_url).netloc
            logger.info(f"   ✓ Verified URL: {verified_url.seed_url}")
            logger.info(f"   ✓ Domain: {domain}")
            
            # =================================================================
            # STAGE 3: INTELLIGENT CRAWLING WITH AUTO-RETRY
            # =================================================================
            use_cached = self.cache_manager.is_cached(domain, force_refresh)
            chunks_crawled = 0
            
            # Check if we have relevant content cached
            if use_cached:
                logger.info(f"\n✨ Stage 3: Checking cached data for {domain}...")
                domain_stats = self.vector_store.get_domain_stats(domain)
                chunks_crawled = domain_stats.get('chunks', 0)
                logger.info(f"   ✓ Cache hit: {chunks_crawled} chunks available")
                
                # ENHANCED: More aggressive relevance check
                if chunks_crawled > 0:
                    test_chunks = self.vector_store.retrieve_relevant(
                        query, 
                        domain, 
                        top_k=3  # Check top 3 instead of 1
                    )
                    
                    # Calculate average relevance of top results
                    if test_chunks:
                        avg_relevance = sum(c['relevance_score'] for c in test_chunks) / len(test_chunks)
                        max_relevance = max(c['relevance_score'] for c in test_chunks)
                        
                        logger.info(f"   ✓ Cached relevance - Avg: {avg_relevance:.2f}, Max: {max_relevance:.2f}")
                        
                        # OPTIMAL THRESHOLD: Re-crawl if average < 0.55 OR max < 0.65
                        if avg_relevance < 0.55 or max_relevance < 0.65:
                            logger.info(f"   🔄 Relevance too low (avg: {avg_relevance:.2f}, max: {max_relevance:.2f})")
                            logger.info(f"   🕷️  Crawling specific page for better content...")
                            use_cached = False
            
            if not use_cached:
                logger.info(f"\n🕷️  Stage 3: Crawling {domain}...")
                chunks = await targeted_crawl(verified_url.seed_url, deconstructed)
                
                if chunks:
                    self.vector_store.ingest_chunks(chunks, domain)
                    self.cache_manager.mark_cached(domain)
                    chunks_crawled = len(chunks)
                    logger.info(f"   ✓ Crawled and indexed {chunks_crawled} chunks")
                else:
                    logger.warning("   ⚠️  No content extracted from crawl")
            
            # =================================================================
            # STAGE 4: RETRIEVE & SYNTHESIZE WITH QUALITY CHECK
            # =================================================================
            logger.info(f"\n💡 Stage 4: Synthesizing answer...")
            relevant_chunks = self.vector_store.retrieve_relevant(
                query, 
                domain, 
                top_k=TOP_K_CHUNKS
            )
            logger.info(f"   ✓ Retrieved {len(relevant_chunks)} relevant chunks")
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found in vector store")
                
                # ENHANCED: Always try crawling if nothing found (even on first attempt)
                logger.info("   🔄 No relevant content in cache, attempting targeted crawl...")
                chunks = await targeted_crawl(verified_url.seed_url, deconstructed, max_pages=3)
                
                if chunks:
                    logger.info(f"   📥 Ingesting {len(chunks)} newly crawled chunks...")
                    self.vector_store.ingest_chunks(chunks, domain)
                    self.cache_manager.mark_cached(domain)
                    
                    # Try retrieving again
                    relevant_chunks = self.vector_store.retrieve_relevant(query, domain, top_k=TOP_K_CHUNKS)
                    logger.info(f"   ✓ Found {len(relevant_chunks)} relevant chunks after crawl")
                
                if not relevant_chunks:
                    return self._error_response(
                        query=query,
                        deconstructed=deconstructed,
                        error="Could not find relevant information",
                        suggestion=f"The page at {domain} may not contain information about '{query}'. Try rephrasing your question or asking about a different topic."
                    )
            
            # Log relevance scores for debugging
            if relevant_chunks:
                avg_score = sum(c['relevance_score'] for c in relevant_chunks) / len(relevant_chunks)
                logger.info(f"   ✓ Average relevance: {avg_score:.2f}")
            
            answer = await synthesize_answer(query, deconstructed, relevant_chunks)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get actual domain stats
            domain_stats = self.vector_store.get_domain_stats(domain)
            
            logger.info(f"\n✅ Complete! Processed in {processing_time:.2f}s")
            logger.info("=" * 80 + "\n")
            
            return {
                "success":True,
                "message":"Query processed successfully",
                "query": query,
                "deconstructed": deconstructed.dict(),
                "verified_url": verified_url.dict(),
                "answer": answer,
                "metadata": {
                    "domain": domain,
                    "chunks_in_database": domain_stats.get('chunks', chunks_crawled),
                    "chunks_used": len(relevant_chunks),
                    "avg_relevance_score": round(sum(c['relevance_score'] for c in relevant_chunks) / len(relevant_chunks), 3) if relevant_chunks else 0,
                    "cached": use_cached,
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}", exc_info=True)
            return self._error_response(
                query=query,
                deconstructed=deconstructed if 'deconstructed' in locals() else DeconstructedQuery(
                    user_intent="unknown",
                    identified_entity="unknown",
                    specific_details=[],
                    inhibitor="none"
                ),
                error=str(e),
                suggestion="Unexpected error occurred.Try again or use force_refresh"
            )
    
    async def ingest_domain(self, url: str) -> Dict[str, Any]:
        """Manually ingest a domain for pre-caching"""
        domain = urlparse(url).netloc
        
        logger.info(f"\n📥 Manual ingestion of {domain}...")
        
        try:
            deconstructed = DeconstructedQuery(
                user_intent="general documentation",
                identified_entity=domain,
                specific_details=["help", "support", "documentation", "guide", "tutorial"],
                inhibitor="none"
            )
            
            chunks = await targeted_crawl(url, deconstructed, max_pages=15)
            
            if not chunks:
                logger.warning(f"No chunks extracted from {domain}")
                return {
                    "status": "failed",
                    "domain": domain,
                    "url": url,
                    "error": "No content could be extracted"
                }
            
            self.vector_store.ingest_chunks(chunks, domain)
            self.cache_manager.mark_cached(domain)
            
            pages_crawled = len(set(c['url'] for c in chunks))
            
            logger.info(f"✅ Ingestion complete: {len(chunks)} chunks from {pages_crawled} pages")
            
            return {
                "status": "success",
                "domain": domain,
                "url": url,
                "chunks_ingested": len(chunks),
                "pages_crawled": pages_crawled,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed for {domain}: {e}", exc_info=True)
            return {
                "status": "error",
                "domain": domain,
                "url": url,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics"""
        try:
            domains_info = []
            
            for domain in self.vector_store.get_all_domains():
                domain_stats = self.vector_store.get_domain_stats(domain)
                cache_info = self.cache_manager.get_cache_info(domain)
                
                domains_info.append({
                    "domain": domain,
                    "chunks": domain_stats['chunks'],
                    "cached_at": cache_info['cached_at'] if cache_info else None,
                    "cache_valid": cache_info['is_valid'] if cache_info else False,
                    "expires_in_hours": round(cache_info['expires_in_hours'], 2) if cache_info else 0
                })
            
            domains_info.sort(key=lambda x: x['chunks'], reverse=True)
            
            return {
                "status": "online",
                "version": "2.0.1",
                "cached_domains": len(self.cache_manager.get_all_cached_domains()),
                "total_chunks": self.vector_store.get_total_chunks(),
                "total_domains": len(self.vector_store.get_all_domains()),
                "domains": domains_info,
                "storage": {
                    "vector_db": "Pinecone (persistent)",
                    "cache": "Redis (persistent)"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Status retrieval failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_domain_cache(self, domain: str) -> Dict[str, Any]:
        """Clear cache for a specific domain"""
        try:
            logger.info(f"🗑️  Clearing cache for {domain}...")
            
            vector_cleared = self.vector_store.clear_domain(domain)
            cache_cleared = self.cache_manager.clear_domain(domain)
            
            status = "cleared" if (vector_cleared or cache_cleared) else "not_found"
            
            logger.info(f"✅ Cache clear complete: {status}")
            
            return {
                "status": status,
                "domain": domain,
                "vector_store_cleared": vector_cleared,
                "cache_cleared": cache_cleared,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Cache clear failed for {domain}: {e}", exc_info=True)
            return {
                "status": "error",
                "domain": domain,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        try:
            vector_health = self.vector_store.health_check()
            cache_health = self.cache_manager.health_check()
            
            all_healthy = vector_health and cache_health
            
            return {
                "healthy": all_healthy,
                "vector_store": "connected" if vector_health else "disconnected",
                "cache_manager": "connected" if cache_health else "disconnected",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}", exc_info=True)
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            logger.info("🧹 Cleaning up resources...")
            self.vector_store.cleanup()
            self.cache_manager.cleanup()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}", exc_info=True)
    
    def _error_response(
        self, 
        query: str, 
        deconstructed: DeconstructedQuery,
        error: str,
        suggestion: str
    ) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            "success":False,
            "message":f"Query processing failed:{error}",
            "query": query,
            "deconstructed": deconstructed.dict(),
            "error": error,
            "suggestion": suggestion,
            "metadata": {
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        }


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_brain_instance: Optional[CognitiveBrain] = None

def get_brain() -> CognitiveBrain:
    """Get or create the singleton Brain instance"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = CognitiveBrain()
    return _brain_instance