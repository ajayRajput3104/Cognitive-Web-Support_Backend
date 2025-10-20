"""
Stage 3: Targeted Crawling Agent
Smart, keyword-guided web crawling and content extraction
"""

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Optional, List, Dict, Tuple
import logging

from ai_agents.stage1_deconstructor import DeconstructedQuery
from config import MAX_PAGES_TO_CRAWL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


async def scrape_page(url: str) -> Tuple[str, List[str]]:
    """
    Scrapes a single page and extracts text and links
    
    Args:
        url: URL to scrape
        
    Returns:
        Tuple of (text_content, list_of_links)
    """
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            if response.status_code != 200:
                logger.warning(f"Failed to scrape {url}: {response.status_code}")
                return "", []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove noise elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()
            
            # Extract clean text
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            # Extract links from same domain
            links = []
            base_domain = urlparse(url).netloc
            
            for a in soup.find_all('a', href=True):
                full_url = urljoin(url, a['href'])
                parsed = urlparse(full_url)
                
                # Only same domain, HTTP/HTTPS
                if parsed.netloc == base_domain and parsed.scheme in ['http', 'https']:
                    # Remove fragments and query params for deduplication
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    links.append(clean_url)
            
            return text, list(set(links))
            
    except Exception as e:
        logger.warning(f"Scraping error for {url}: {e}")
        return "", []


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks for better context preservation
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Only keep substantial chunks
            chunks.append(chunk)
    
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
    for kw in keywords:
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
                   'privacy', 'terms', 'cookie', 'download', 'pdf']
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
    Performs intelligent, keyword-guided crawling
    
    Args:
        seed_url: Starting URL
        deconstructed: Structured query information for guidance
        max_pages: Maximum pages to crawl
        
    Returns:
        List of chunks with metadata
    """
    
    if max_pages is None:
        max_pages = MAX_PAGES_TO_CRAWL
    
    domain = urlparse(seed_url).netloc
    keywords = deconstructed.specific_details + [deconstructed.user_intent]
    
    visited = set()
    to_visit = [(seed_url, 0, 10.0)]  # (url, depth, priority)
    all_chunks = []
    
    logger.info(f"Starting crawl of {domain} (max {max_pages} pages)")
    logger.debug(f"Keywords: {keywords[:5]}")
    
    while to_visit and len(visited) < max_pages:
        # Sort by priority (highest first)
        to_visit.sort(key=lambda x: x[2], reverse=True)
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
        
        # Add relevant links to queue
        if depth < 2 and links:
            link_scores = []
            for link in links[:100]:  # Limit to avoid memory issues
                score = score_link_relevance(link, "", keywords)
                if score > 0:
                    link_scores.append((link, score))
            
            # Add top scoring links
            link_scores.sort(key=lambda x: x[1], reverse=True)
            for link, score in link_scores[:10]:
                if link not in visited:
                    to_visit.append((link, depth + 1, score))
    
    logger.info(f"Crawl complete: {len(visited)} pages, {len(all_chunks)} chunks")
    return all_chunks