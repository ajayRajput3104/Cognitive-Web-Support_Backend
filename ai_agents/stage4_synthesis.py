"""
Stage 4: Answer Synthesis Agent - ENHANCED
Better formatting, clearer instructions, more structured outputs
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
    Synthesizes final answer with enhanced formatting and structure
    """
    
    if not relevant_chunks:
        return (f"I couldn't find relevant information about {deconstructed.identified_entity} "
                f"for your query. The domain may need to be crawled first, or try rephrasing your question.")
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source {i+1} - Relevance: {chunk['relevance_score']:.2f}]\n"
        f"URL: {chunk['url']}\n"
        f"Content: {chunk['text'][:1000]}"  # Increased from 800 for better context
        for i, chunk in enumerate(relevant_chunks)
    ])
    
    # Enhanced prompt for better, more detailed answers
    prompt = f"""You are an expert technical support specialist for {deconstructed.identified_entity}.

**User's Question:** "{user_query}"

**What the user wants to accomplish:** {deconstructed.user_intent}
**Specific topics they're asking about:** {', '.join(deconstructed.specific_details[:5])}
**Their current challenge:** {deconstructed.inhibitor}

**Verified Documentation Sources:**

{context}

**CRITICAL FORMATTING RULES:**

1. **NO decorative separators** - Do not use lines like ──────, ====, ____, etc.
2. **Clean markdown only** - Use proper markdown syntax:
   - Use ## for section headers
   - Use **bold** for emphasis (not ***text***)
   - Use backticks for code: `code here`
   - Use triple backticks for code blocks:
   ```language
   code block
   ```
3. **Structure:**
   - Start with a brief direct answer (1-2 sentences)
   - Then provide COMPLETE, DETAILED step-by-step instructions
   - Use numbered lists (1., 2., 3.) for sequential steps
   - Use bullet points (-, not *) for non-sequential items
4. **Code blocks:**
   - Always specify language: ```javascript or ```bash or ```python
   - Keep code examples concise but complete
   - Add brief comments in code where helpful
5. **Source citations:**
   - Cite sources naturally: "According to the documentation..."
   - Do NOT repeat citations unnecessarily
   - Do NOT add source list at the end (I will add that)
6. **Completeness:**
   - Include ALL relevant details
   - Explain each step thoroughly
   - Mention prerequisites, options, and alternatives
   - Add warnings for common mistakes
7. **Tone:**
   - Professional but friendly
   - Clear and easy to follow
   - Avoid unnecessary technical jargon

**IMPORTANT:**
- DO NOT add any separator lines or decorative borders
- DO NOT add a sources section (I will add that automatically)
- DO NOT use excessive asterisks or special characters
- Keep formatting clean and readable

Now provide your detailed, well-formatted answer:"""

    try:
        logger.debug(f"Synthesizing answer for: {user_query[:100]}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,  # Lower for more focused, detailed answers
                        "maxOutputTokens": 3000,  # Increased to allow longer, detailed responses
                        "topP": 0.9,
                        "topK": 30
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                
                # Enhanced fallback with better formatting
                best_chunk = relevant_chunks[0]
                return f"""Based on the {deconstructed.identified_entity} documentation, here's what I found:

{best_chunk['text'][:600]}...

**Note:** I encountered an API error (HTTP {response.status_code}) while generating a formatted response. The content above is directly from the source documentation.

**For complete information, visit:**
{best_chunk['url']}

**Alternative:** Try rephrasing your question or contact {deconstructed.identified_entity} support directly.

📚 **Sources:**
1. {best_chunk['url']}"""
            
            result = response.json()
            
            # Check if we got a valid response
            if "candidates" not in result or not result["candidates"]:
                logger.error("No candidates in Gemini response")
                return self._fallback_answer(user_query, deconstructed, relevant_chunks)
            
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean up the answer
            import re
            
            # Remove excessive newlines
            answer = re.sub(r'\n{3,}', '\n\n', answer)
            
            # Remove duplicate consecutive source citations
            answer = re.sub(r'\(Source (\d+)\)\s*\(Source \1\)', r'(Source \1)', answer)
            
            # Remove stray asterisks used for emphasis (markdown artifacts)
            # But preserve **bold** and *italic* that are properly formatted
            answer = re.sub(r'\*{3,}', '**', answer)  # *** → **
            answer = re.sub(r'(?<!\*)\*(?!\*)(?!\w)', '', answer)  # Remove single * not used for italic
            
            # Remove the long separator line that overflows
            answer = re.sub(r'─{10,}', '', answer)
            answer = re.sub(r'-{10,}', '', answer)
            answer = re.sub(r'_{10,}', '', answer)
            answer = re.sub(r'={10,}', '', answer)
            
            # Clean up source section formatting
            answer = re.sub(r'\*+Answers are based on.*?\*+', 'Answers are based on official documentation and verified sources only.', answer, flags=re.IGNORECASE)
            
            # Add clean, contained source section
            sources = "\n\n**📚 Official Documentation Sources:**\n"
            for i, chunk in enumerate(relevant_chunks):
                relevance_indicator = "⭐" if chunk['relevance_score'] >= 0.7 else "✓"
                sources += f"\n{relevance_indicator} **Source {i+1}:** {chunk['url']}"
            
            sources += "\n\n*Answers are based on official documentation and verified sources only.*"
            
            logger.debug("Answer synthesis successful")
            return answer + sources
            
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}", exc_info=True)
        return _fallback_answer(user_query, deconstructed, relevant_chunks)


def _fallback_answer(
    user_query: str,
    deconstructed: DeconstructedQuery,
    relevant_chunks: List[Dict[str, Any]]
) -> str:
    """
    Generate a fallback answer when Gemini API fails
    """
    best_chunk = relevant_chunks[0]
    
    return f"""I found relevant information from {deconstructed.identified_entity} documentation:

**Your Question:** {user_query}

**From the Documentation:**

{best_chunk['text'][:500]}...

**Direct Link:** {best_chunk['url']}

**Note:** I'm currently unable to generate a formatted answer, but the information above should help. For the complete guide, please visit the source link.

📚 **Additional Sources:**
""" + "\n".join([f"{i+1}. {chunk['url']}" for i, chunk in enumerate(relevant_chunks)])