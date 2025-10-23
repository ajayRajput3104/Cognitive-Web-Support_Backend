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

**INSTRUCTIONS - FOLLOW CAREFULLY:**

1. **Structure Your Response:**
   - Start with a brief direct answer (1-2 sentences)
   - Then provide COMPLETE, DETAILED step-by-step instructions
   - Include ALL relevant details, options, and settings
   - End with additional tips or warnings if applicable

2. **For Step-by-Step Instructions:**
   - Use numbered lists (1., 2., 3.)
   - Include EVERY step - don't skip "obvious" ones
   - For each step, explain WHAT to do AND WHERE to find it
   - If there are options/choices in a step, explain them
   - Example: "3. **Name your repository** - Enter a descriptive name (required). Choose between Public (visible to everyone) or Private (only you and collaborators can see it)."

3. **Source Citations:**
   - Cite sources inline ONLY when introducing NEW information
   - Use format: "According to the documentation (Source 1), ..."
   - DON'T repeat source citations for the same information
   - Use sources naturally, not after every sentence

4. **Completeness:**
   - Cover ALL aspects mentioned in the documentation
   - Include prerequisites, requirements, or permissions needed
   - Mention alternative methods if the docs show them
   - Add warnings about common mistakes or limitations

5. **Formatting:**
   - Use **bold** for important terms, button names, or settings
   - Use line breaks between major sections
   - Keep it scannable but thorough (300-500 words is fine if needed)
   - Use bullet points for non-sequential options or tips

6. **Tone:**
   - Be helpful and clear, not robotic
   - Explain WHY when it helps understanding
   - Assume user is following along step-by-step

**Example of GOOD detail level:**
"3. **Configure repository settings:**
   - Enter a repository name (required) - use lowercase, hyphens instead of spaces
   - Add a description (optional but recommended for public repos)
   - Choose visibility: **Public** (anyone can see) or **Private** (invitation only)
   - Initialize with README: Check this to create a starting README.md file
   - Add .gitignore: Select a template matching your project type to ignore common files
   - Choose a license: Important for open source projects (Source 2 has details)"

**Example of BAD detail level:**
"3. Configure settings and create repository."

Now provide your detailed, comprehensive answer:"""

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
            answer = re.sub(r'\n{3,}', '\n\n', answer)  # Remove excessive newlines
            # Remove duplicate consecutive source citations (e.g., "Source 1) (Source 1)" → "Source 1)")
            answer = re.sub(r'\(Source (\d+)\)\s*\(Source \1\)', r'(Source \1)', answer)
            
            # Add clean, professional source section
            sources = "\n\n" + "─" * 70 + "\n\n**📚 Official Documentation Sources:**\n"
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