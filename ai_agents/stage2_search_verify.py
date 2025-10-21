"""
Stage 2: Search & Verification Agent (HYBRID SEARCH 2024)
Finds and verifies official documentation URLs
HYBRID APPROACH: Google first → DuckDuckGo fallback when quota exceeded
"""

import httpx
import json
import re
from typing import Optional, List
from pydantic import BaseModel
import logging

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, GEMINI_API_KEY
from ai_agents.stage1_deconstructor import DeconstructedQuery

logger = logging.getLogger(__name__)

# Free search alternative
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("DuckDuckGo search not available")


class SearchResult(BaseModel):
    """Single search result"""
    title: str
    url: str
    snippet: str


class VerifiedURL(BaseModel):
    """Verified URL with reasoning"""
    seed_url: str
    reasoning: str


# Track Google API quota status
GOOGLE_QUOTA_EXCEEDED = False


async def search_web(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    HYBRID SEARCH: Try Google first, fallback to DuckDuckGo if quota exceeded
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    global GOOGLE_QUOTA_EXCEEDED
    
    # Try Google first (if quota not exceeded and credentials available)
    if not GOOGLE_QUOTA_EXCEEDED and GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            logger.debug(f"Attempting Google Search: {query}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": GOOGLE_API_KEY,
                        "cx": GOOGLE_CSE_ID,
                        "q": query,
                        "num": num_results
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get("items", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", "")
                        ))
                    
                    if results:
                        logger.info(f"✓ Google Search: Found {len(results)} results")
                        return results
                
                elif response.status_code == 429:
                    # Quota exceeded - switch to DuckDuckGo
                    logger.warning("⚠ Google Search quota exceeded (100/day). Switching to DuckDuckGo fallback...")
                    GOOGLE_QUOTA_EXCEEDED = True
                    # Fall through to DuckDuckGo
                
                else:
                    logger.warning(f"Google Search API error: {response.status_code}")
                    # Fall through to DuckDuckGo
                    
        except Exception as e:
            logger.warning(f"Google Search failed: {e}. Falling back to DuckDuckGo...")
    
    # Fallback to DuckDuckGo (free, unlimited)
    if DDGS_AVAILABLE:
        try:
            logger.debug(f"Using DuckDuckGo Search: {query}")
            
            ddgs = DDGS()
            results = []
            
            for r in ddgs.text(query, max_results=num_results):
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', ''),
                    snippet=r.get('body', '')
                ))
            
            if results:
                logger.info(f"✓ DuckDuckGo Search: Found {len(results)} results")
                return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}", exc_info=True)
    else:
        logger.error("❌ No search engines available (DuckDuckGo not installed)")
    
    return []


async def verify_and_select_url(
    deconstructed: DeconstructedQuery,
    search_results: List[SearchResult]
) -> VerifiedURL:
    """
    Uses Gemini Flash to select the best official URL
    
    Args:
        deconstructed: Structured query information
        search_results: List of search results
        
    Returns:
        VerifiedURL with the best seed URL and reasoning
    """
    
    if not search_results:
        raise ValueError("No search results to verify")
    
    # Format results for prompt
    results_text = "\n\n".join([
        f"Result {i+1}:\nTitle: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
        for i, r in enumerate(search_results[:5])
    ])
    
    prompt = f"""You are a URL verification expert. Select the SINGLE BEST official support/documentation URL.

Entity: {deconstructed.identified_entity}
User Intent: {deconstructed.user_intent}
Details: {', '.join(deconstructed.specific_details[:5])}

Search Results:
{results_text}

Select the ONE best URL that is:
1. Official (help., docs., support., developer., api., etc. subdomains preferred)
2. Most relevant to the specific user problem
3. Comprehensive and authoritative

Respond with ONLY valid JSON (no markdown):
{{
  "seed_url": "the complete URL",
  "reasoning": "brief explanation why this is the best choice"
}}"""

    try:
        logger.debug("Verifying URLs with Gemini...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 300
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.debug(f"URL verified: {parsed['seed_url']}")
                return VerifiedURL(**parsed)
            else:
                logger.warning("No JSON in verification response, using first result")
                return VerifiedURL(
                    seed_url=search_results[0].url,
                    reasoning="Selected first result (parsing fallback)"
                )
                
    except Exception as e:
        logger.warning(f"URL verification failed, using first result: {e}")
        return VerifiedURL(
            seed_url=search_results[0].url,
            reasoning="Selected first result (error fallback)"
        )