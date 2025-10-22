"""
Stage 3: Targeted Crawling Agent - MEMORY OPTIMIZED
Smart, keyword-guided web crawling with streaming and size limits
"""

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Optional, List, Dict, Tuple
import logging
import gc

from ai_agents.stage1_deconstructor import DeconstructedQuery
from config import MAX_PAGES_TO_CRAWL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CONTENT_SIZE_MB

logger = logging.getLogger(__name__)

# Maximum content size in bytes
MAX_CONTENT_BYTES = MAX_CONTENT_SIZE_MB * 1024 * 1024


async def scrape_page(url: str) -> Tuple[str, List[str]]:
    """
    Scrapes a single page with MEMORY-EFFICIENT streaming
    Prevents OOM by limiting content size and using streaming
    
    Args:
        url: URL to scrape
        
    Returns:
        Tuple of (text_content, list_of_links)
    """
    try:
        async with httpx.AsyncClient(
            timeout=15.0, 
            follow_redirects=True,
            limits=httpx.Limits(max_connections=5) 
        ) as client:
            
            # Use streaming to avoid loading entire page into memory
            content = b""
            try:
                async with client.stream(
                    'GET',
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                ) as response:
                    
                    if response.status_code != 200:
                        logger.warning(f"Failed to scrape {url}: {response.status_code}")
                        return "", []
                    
                    # Read in chunks with size limit
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        content += chunk
                        
                        # CRITICAL: Stop if content too large (prevents OOM)
                        if len(content) > MAX_CONTENT_BYTES:
                            logger.warning(f"Content too large for {url}, truncating at {MAX_CONTENT_SIZE_MB}MB")
                            break
            
            except httpx.ReadTimeout:
                logger.warning(f"Timeout reading {url}")
                if not content:  # If we got nothing, return empty
                    return "", []
                # Otherwise continue with partial content
            
            # Parse HTML
            try:
                html_text = content.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to decode content from {url}: {e}")
                return "", []
            
            # Cleanup content from memory immediately
            del content
            gc.collect()
            
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # Remove noise elements aggressively
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                                'aside', 'iframe', 'noscript', 'meta', 'link']):
                element.decompose()
            
            # Extract clean text
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            # Limit text size (additional safety)
            if len(text) > 50000:  # ~50KB text max
                text = text[:50000]
                logger.debug(f"Truncated text for {url} to 50KB")
            
            # Extract links from same domain
            links = []
            base_domain = urlparse(url).netloc
            
            for a in soup.find_all('a', href=True, limit=100):  # Limit links processed
                try:
                    full_url = urljoin(url, a['href'])
                    parsed = urlparse(full_url)
                    
                    # Only same domain, HTTP/HTTPS
                    if parsed.netloc == base_domain and parsed.scheme in ['http', 'https']:
                        # Remove fragments and query params for deduplication
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        links.append(clean_url)
                except Exception:
                    continue  # Skip malformed URLs
            
            # Cleanup
            del soup
            del html_text
            gc.collect()
            
            return text, list(set(links))[:50]  # Limit to 50 unique links
            
    except httpx.ConnectError:
        logger.warning(f"Connection error for {url}")
        return "", []
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to {url}")
        return "", []
    except Exception as e:
        logger.warning(f"Scraping error for {url}: {e}")
        return "", []


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks with memory efficiency
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) < 50:
        return []
    
    words = text.split()
    chunks = []
    
    # Limit total chunks to prevent memory bloat
    max_chunks = 100
    chunk_count = 0
    
    for i in range(0, len(words), chunk_size - overlap):
        if chunk_count >= max_chunks:
            logger.debug(f"Reached max chunks ({max_chunks}), stopping")
            break
        
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Only keep substantial chunks
            chunks.append(chunk)
            chunk_count += 1
    
    return chunks


def score_link_relevance(url: str, anchor_text: str, keywords: List[str]) -> float:
    """
    Scores link relevance based on keywords
    
    Args:
        url: Link URL
        anchor_text: Anchor text of the link
        keywords: Keywords to match against
        
    Returns:
        Relevance score (higher is better)
    """
    text = (url + " " + anchor_text).lower()
    score = 0.0
    
    # Keyword matching
    for kw in keywords[:10]:  # Limit keywords processed
        if kw.lower() in text:
            score += 2.0
    
    # Boost for support-related terms
    support_terms = ['help', 'support', 'docs', 'documentation', 'guide', 'tutorial', 
                     'faq', 'how-to', 'troubleshoot', 'api', 'reference']
    for term in support_terms:
        if term in text:
            score += 1.0
    
    # Penalize for non-content pages
    noise_terms = ['login', 'signup', 'register', 'cart', 'checkout', 'account', 
                   'privacy', 'terms', 'cookie', 'download', 'pdf', 'print']
    for term in noise_terms:
        if term in text:
            score -= 0.5
    
    return max(0, score)


async def targeted_crawl(
    seed_url: str,
    deconstructed: DeconstructedQuery,
    max_pages: Optional[int] = None
) -> List[Dict]:
    """
    Performs intelligent, keyword-guided crawling with MEMORY LIMITS
    
    Args:
        seed_url: Starting URL
        deconstructed: Structured query information for guidance
        max_pages: Maximum pages to crawl
        
    Returns:
        List of chunks with metadata
    """
    
    if max_pages is None:
        max_pages = MAX_PAGES_TO_CRAWL
    
    # Further limit on free tier
    max_pages = min(max_pages, 5)  # Hard cap at 5 pages
    
    domain = urlparse(seed_url).netloc
    keywords = deconstructed.specific_details[:5] + [deconstructed.user_intent]  # Limit keywords
    
    visited = set()
    to_visit = [(seed_url, 0, 10.0)]  # (url, depth, priority)
    all_chunks = []
    
    logger.info(f"🕷️  Starting crawl of {domain} (max {max_pages} pages)")
    logger.debug(f"Keywords: {keywords[:3]}")
    
    while to_visit and len(visited) < max_pages:
        # Sort by priority (highest first) but limit queue size
        to_visit = sorted(to_visit, key=lambda x: x[2], reverse=True)[:20]
        
        current_url, depth, priority = to_visit.pop(0)
        
        # Skip if already visited or too deep
        if current_url in visited or depth > 2:
            continue
        
        visited.add(current_url)
        logger.debug(f"Crawling [{len(visited)}/{max_pages}]: {current_url[:80]}")
        
        # Scrape page
        text, links = await scrape_page(current_url)
        
        if text:
            # Chunk the text
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'url': current_url,
                    'domain': domain
                })
            
            # Memory cleanup after processing each page
            del text
            gc.collect()
        
        # Add relevant links to queue (only if not too deep)
        if depth < 2 and links:
            link_scores = []
            for link in links[:50]:  # Limit links evaluated
                score = score_link_relevance(link, "", keywords)
                if score > 0:
                    link_scores.append((link, score))
            
            # Add top scoring links
            link_scores.sort(key=lambda x: x[1], reverse=True)
            for link, score in link_scores[:5]:  # Only top 5 links
                if link not in visited:
                    to_visit.append((link, depth + 1, score))
            
            # Cleanup
            del link_scores
            gc.collect()
    
    pages_crawled = len(visited)
    logger.info(f"✅ Crawl complete: {pages_crawled} pages, {len(all_chunks)} chunks")
    
    # Final memory cleanup
    del visited
    del to_visit
    gc.collect()
    
    return all_chunks