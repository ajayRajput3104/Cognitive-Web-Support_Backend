"""
Stage 2: Search & Verification Agent
Finds and verifies official documentation URLs with better error handling
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
                    logger.warning("⚠  Google Search quota exceeded. Switching to DuckDuckGo...")
                    GOOGLE_QUOTA_EXCEEDED = True
                
                else:
                    logger.warning(f"Google Search API error: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Google Search failed: {e}. Falling back to DuckDuckGo...")
    
    # Fallback to DuckDuckGo
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
    
    return []


async def verify_and_select_url(
    deconstructed: DeconstructedQuery,
    search_results: List[SearchResult]
) -> VerifiedURL:
    """
    Uses Gemini Flash to select the best official URL - FIXED JSON PARSING
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

CRITICAL INSTRUCTIONS:
1. Select the ONE best URL that is official and authoritative
2. Prefer URLs with: docs., help., support., documentation, api., developer.
3. Avoid: reddit, quora, stackoverflow, forums (unless official), blogs
4. Return ONLY valid JSON - no markdown, no code blocks, no extra text

Respond with this EXACT format:
{{"seed_url": "complete_url_here", "reasoning": "brief_explanation_here"}}"""

    try:
        logger.debug("Verifying URLs with Gemini...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 300,
                        "responseMimeType": "application/json"  # Force JSON response
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            
            # FIXED: Better error handling for response structure
            try:
                # Try to get the text response
                if "candidates" not in result or not result["candidates"]:
                    logger.warning("No candidates in Gemini response")
                    raise KeyError("candidates")
                
                candidate = result["candidates"][0]
                
                if "content" not in candidate:
                    logger.warning("No content in candidate")
                    raise KeyError("content")
                
                content = candidate["content"]
                
                if "parts" not in content or not content["parts"]:
                    logger.warning("No parts in content")
                    raise KeyError("parts")
                
                text_response = content["parts"][0].get("text", "")
                
                if not text_response:
                    logger.warning("Empty text response")
                    raise ValueError("Empty response")
                
                # Try to parse JSON
                # Remove markdown code blocks if present
                text_response = re.sub(r'```json\s*|\s*```', '', text_response)
                text_response = text_response.strip()
                
                # Extract JSON if wrapped in other text
                json_match = re.search(r'\{[^{}]*"seed_url"[^{}]*\}', text_response)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = json.loads(text_response)
                
                logger.debug(f"URL verified: {parsed['seed_url']}")
                return VerifiedURL(**parsed)
                
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse Gemini response: {e}")
                logger.debug(f"Raw response: {result}")
                
                # Fallback: Use first official-looking result
                for search_result in search_results:
                    url_lower = search_result.url.lower()
                    # Prefer official documentation URLs
                    if any(indicator in url_lower for indicator in 
                           ['docs.', 'help.', 'support.', 'documentation', 
                            'api.', 'developer.', 'guide', '/docs/', '/help/']):
                        logger.info(f"Using official-looking URL: {search_result.url}")
                        return VerifiedURL(
                            seed_url=search_result.url,
                            reasoning="Selected official documentation URL (parsing fallback)"
                        )
                
                # Last resort: first result
                logger.warning("Using first search result as fallback")
                return VerifiedURL(
                    seed_url=search_results[0].url,
                    reasoning="Selected first result (parsing fallback)"
                )
                
    except Exception as e:
        logger.warning(f"URL verification failed: {e}")
        # Fallback to first result
        return VerifiedURL(
            seed_url=search_results[0].url,
            reasoning="Selected first result (error fallback)"
        )