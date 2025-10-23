"""
Stage 3: Targeted Crawling Agent - ENHANCED
Smart crawling with anti-bot bypass and better content extraction
"""

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Optional, List, Dict, Tuple
import logging
import gc
import asyncio

from ai_agents.stage1_deconstructor import DeconstructedQuery
from config import MAX_PAGES_TO_CRAWL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CONTENT_SIZE_MB

logger = logging.getLogger(__name__)

# Maximum content size in bytes
MAX_CONTENT_BYTES = MAX_CONTENT_SIZE_MB * 1024 * 1024

# Enhanced user agent to avoid bot detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


async def scrape_page(url: str, retry_count: int = 0) -> Tuple[str, List[str]]:
    """
    Scrapes a single page with anti-bot measures and retry logic
    """
    try:
        # Rotate user agents
        user_agent = USER_AGENTS[retry_count % len(USER_AGENTS)]
        
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=5)
        ) as client:
            
            # Add delay to avoid rate limiting
            if retry_count > 0:
                await asyncio.sleep(2)  # 2 second delay on retries
            
            try:
                async with client.stream(
                    'GET',
                    url,
                    headers={
                        'User-Agent': user_agent,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Cache-Control': 'max-age=0'
                    }
                ) as response:
                    
                    if response.status_code == 403 and retry_count < 2:
                        # Retry with different user agent
                        logger.info(f"403 Forbidden, retrying {url} (attempt {retry_count + 1}/3)")
                        return await scrape_page(url, retry_count + 1)
                    
                    if response.status_code != 200:
                        logger.warning(f"Failed to scrape {url}: {response.status_code}")
                        return "", []
                    
                    # Read in chunks with size limit
                    content = b""
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        content += chunk
                        
                        # Stop if content too large
                        if len(content) > MAX_CONTENT_BYTES:
                            logger.warning(f"Content too large for {url}, truncating at {MAX_CONTENT_SIZE_MB}MB")
                            break
            
            except httpx.ReadTimeout:
                logger.warning(f"Timeout reading {url}")
                if not content:
                    return "", []
            
            # Decode content
            try:
                html_text = content.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to decode content from {url}: {e}")
                return "", []
            
            # Cleanup content from memory
            del content
            gc.collect()
            
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # ENHANCED: Better content extraction for forums and documentation
            text = extract_main_content(soup, url)
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {url} ({len(text)} chars)")
                return "", []
            
            # Extract links from same domain
            links = []
            base_domain = urlparse(url).netloc
            
            for a in soup.find_all('a', href=True, limit=100):
                try:
                    full_url = urljoin(url, a['href'])
                    parsed = urlparse(full_url)
                    
                    # Only same domain, HTTP/HTTPS
                    if parsed.netloc == base_domain and parsed.scheme in ['http', 'https']:
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        links.append(clean_url)
                except Exception:
                    continue
            
            # Cleanup
            del soup
            del html_text
            gc.collect()
            
            logger.debug(f"Extracted {len(text)} chars and {len(set(links))} links from {url}")
            return text, list(set(links))[:50]
            
    except httpx.ConnectError:
        logger.warning(f"Connection error for {url}")
        return "", []
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to {url}")
        return "", []
    except Exception as e:
        logger.warning(f"Scraping error for {url}: {e}")
        return "", []


def extract_main_content(soup: BeautifulSoup, url: str) -> str:
    """
    Enhanced content extraction that works better with forums and documentation
    """
    
    # Remove noise elements aggressively
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                        'aside', 'iframe', 'noscript', 'meta', 'link',
                        'svg', 'button', 'input', 'form']):
        element.decompose()
    
    # Try to find main content area (works for most documentation sites)
    main_content = None
    
    # Strategy 1: Look for semantic HTML5 tags
    for tag in ['main', 'article', '[role="main"]', '.content', '.documentation', 
                '.post-content', '.article-content', '#content', '#main-content']:
        main_content = soup.select_one(tag)
        if main_content:
            logger.debug(f"Found main content using selector: {tag}")
            break
    
    # Strategy 2: Look for Discourse forum posts (common forum software)
    if not main_content:
        discourse_posts = soup.select('.topic-post')
        if discourse_posts:
            logger.debug("Found Discourse forum posts")
            main_content = soup.new_tag('div')
            for post in discourse_posts[:5]:  # Get first 5 posts
                main_content.append(post)
    
    # Strategy 3: Look for common documentation containers
    if not main_content:
        for selector in ['.markdown-body', '.rst-content', '.document', 
                        '.post-body', '.topic-body', '.cooked']:
            elements = soup.select(selector)
            if elements:
                logger.debug(f"Found content using selector: {selector}")
                main_content = soup.new_tag('div')
                for elem in elements:
                    main_content.append(elem)
                break
    
    # Strategy 4: Find largest text block
    if not main_content:
        # Find all divs and get the one with most text
        divs = soup.find_all(['div', 'section'], recursive=True)
        if divs:
            main_content = max(divs, key=lambda x: len(x.get_text(strip=True)))
            logger.debug("Using largest text block as main content")
    
    # Fallback: use entire body
    if not main_content:
        main_content = soup.find('body') or soup
        logger.debug("Fallback: using entire body")
    
    # Extract and clean text
    text = main_content.get_text(separator=' ', strip=True)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove common noise patterns
    text = re.sub(r'Cookie Policy.*?Accept', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sign up.*?Log in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subscribe to newsletter', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks with better sentence preservation
    """
    if not text or len(text.strip()) < 50:
        return []
    
    # Split into sentences first (better than word splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunks = 100
    
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        
        # If single sentence is too long, split it
        if words_in_sentence > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence into words
            words = sentence.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk.strip()) > 100:
                    chunks.append(chunk)
                    if len(chunks) >= max_chunks:
                        return chunks
            continue
        
        # Add sentence to current chunk
        if current_length + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_length += words_in_sentence
        else:
            # Save current chunk and start new one with overlap
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                if len(chunks) >= max_chunks:
                    return chunks
                
                # Keep last few sentences for overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 1 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.strip()) > 100:
            chunks.append(chunk_text)
    
    logger.debug(f"Created {len(chunks)} chunks from {len(text)} chars")
    return chunks


def score_link_relevance(url: str, anchor_text: str, keywords: List[str]) -> float:
    """Scores link relevance based on keywords"""
    text = (url + " " + anchor_text).lower()
    score = 0.0
    
    # Keyword matching
    for kw in keywords[:10]:
        if kw.lower() in text:
            score += 2.0
    
    # Boost for support-related terms
    support_terms = ['help', 'support', 'docs', 'documentation', 'guide', 'tutorial', 
                     'faq', 'how-to', 'troubleshoot', 'api', 'reference', 'getting-started']
    for term in support_terms:
        if term in text:
            score += 1.0
    
    # Penalize for non-content pages
    noise_terms = ['login', 'signup', 'register', 'cart', 'checkout', 'account', 
                   'privacy', 'terms', 'cookie', 'download', 'pdf', 'print', 'share']
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
    Intelligent crawling with anti-bot measures and better extraction
    """
    
    if max_pages is None:
        max_pages = MAX_PAGES_TO_CRAWL
    
    # Further limit on free tier
    max_pages = min(max_pages, 5)
    
    domain = urlparse(seed_url).netloc
    keywords = deconstructed.specific_details[:5] + [deconstructed.user_intent]
    
    visited = set()
    to_visit = [(seed_url, 0, 10.0)]  # (url, depth, priority)
    all_chunks = []
    
    logger.info(f"🕷️  Starting crawl of {domain} (max {max_pages} pages)")
    logger.debug(f"Keywords: {keywords[:3]}")
    
    while to_visit and len(visited) < max_pages:
        # Sort by priority
        to_visit = sorted(to_visit, key=lambda x: x[2], reverse=True)[:20]
        
        current_url, depth, priority = to_visit.pop(0)
        
        # Skip if already visited or too deep
        if current_url in visited or depth > 2:
            continue
        
        visited.add(current_url)
        logger.debug(f"Crawling [{len(visited)}/{max_pages}]: {current_url[:80]}")
        
        # Scrape page with anti-bot measures
        text, links = await scrape_page(current_url)
        
        if text:
            # Chunk the text with sentence preservation
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'url': current_url,
                    'domain': domain
                })
            
            logger.debug(f"Extracted {len(chunks)} chunks from {current_url}")
            
            # Memory cleanup
            del text
            gc.collect()
        else:
            logger.warning(f"No text extracted from {current_url}")
        
        # Add relevant links to queue (only if not too deep)
        if depth < 2 and links:
            link_scores = []
            for link in links[:50]:
                score = score_link_relevance(link, "", keywords)
                if score > 0:
                    link_scores.append((link, score))
            
            # Add top scoring links
            link_scores.sort(key=lambda x: x[1], reverse=True)
            for link, score in link_scores[:5]:
                if link not in visited:
                    to_visit.append((link, depth + 1, score))
            
            del link_scores
            gc.collect()
    
    pages_crawled = len(visited)
    logger.info(f"✅ Crawl complete: {pages_crawled} pages, {len(all_chunks)} chunks")
    
    # Final memory cleanup
    del visited
    del to_visit
    gc.collect()
    
    return all_chunks