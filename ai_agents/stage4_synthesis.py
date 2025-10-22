"""
Stage 4: Answer Synthesis Agent
RAG-based answer generation using Gemini Pro and retrieved context
"""

import httpx
import logging
from typing import List, Dict, Any

from ai_agents.stage1_deconstructor import DeconstructedQuery
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


async def synthesize_answer(
    user_query: str,
    deconstructed: DeconstructedQuery,
    relevant_chunks: List[Dict[str, Any]]
) -> str:
    """
    Synthesizes final answer using Gemini Pro and retrieved context
    
    Args:
        user_query: Original user question
        deconstructed: Structured query information
        relevant_chunks: Retrieved relevant text chunks from vector DB
        
    Returns:
        Synthesized answer with source citations
    """
    
    if not relevant_chunks:
        return (f"I couldn't find relevant information about {deconstructed.identified_entity} "
                f"for your query. The domain may need to be crawled first, or try rephrasing your question.")
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source {i+1} - Relevance: {chunk['relevance_score']:.2f}]\n"
        f"URL: {chunk['url']}\n"
        f"Content: {chunk['text'][:800]}"  # Limit to avoid token limits
        for i, chunk in enumerate(relevant_chunks)
    ])
    
    prompt = f"""You are an expert support agent for {deconstructed.identified_entity}.

A user asked: "{user_query}"

Analysis:
- Intent: {deconstructed.user_intent}
- Specific Details: {', '.join(deconstructed.specific_details[:5])}
- Problem: {deconstructed.inhibitor}

Using ONLY the following verified information from official {deconstructed.identified_entity} documentation, provide a clear, helpful answer:

{context}

Instructions:
1. Provide step-by-step guidance when applicable
2. Reference specific URLs from the sources (mention "Source 1", "Source 2", etc.)
3. If the context doesn't fully answer the question, state what information is available and what is missing
4. Be concise but thorough (aim for 200-400 words)
5. Use a helpful, professional tone
6. Format with clear paragraphs and bullet points where appropriate

Answer:"""

    try:
        logger.debug(f"Synthesizing answer for: {user_query[:100]}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2000
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Add source URLs at the end
            sources = "\n\n📚 **Sources:**\n" + "\n".join([
                f"{i+1}. {chunk['url']}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            logger.debug("Answer synthesis successful")
            return answer + sources
            
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}", exc_info=True)
        return f"Error generating answer: {str(e)}"