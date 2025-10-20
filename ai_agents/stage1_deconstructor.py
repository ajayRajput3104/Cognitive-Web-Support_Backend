"""
Stage 1: Query Deconstruction Agent
Analyzes user queries and extracts structured information using Gemini Flash
"""

import httpx
import json
import re
from typing import Optional
from pydantic import BaseModel
import logging

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class DeconstructedQuery(BaseModel):
    """Structured representation of a deconstructed user query"""
    user_intent: str
    identified_entity: str
    specific_details: list[str]
    inhibitor: str


async def deconstruct_query(user_query: str) -> DeconstructedQuery:
    """
    Uses Gemini Flash to analyze and structure user queries
    
    Args:
        user_query: Raw user question
        
    Returns:
        DeconstructedQuery with structured information
    """
    
    prompt = f"""You are a query deconstruction expert. Analyze this support query and extract structured information.

User Query: "{user_query}"

Extract:
1. user_intent: What is the user trying to accomplish? (be specific)
2. identified_entity: Which company/platform/service? (e.g., "GitHub", "Shopify", "Netflix")
3. specific_details: List of specific details (features, error codes, actions, etc.)
4. inhibitor: What is blocking the user or causing the problem?

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "user_intent": "string describing what user wants to do",
  "identified_entity": "company/platform name",
  "specific_details": ["detail1", "detail2", "detail3"],
  "inhibitor": "what's preventing the user"
}}"""

    try:
        logger.debug(f"Deconstructing query: {user_query[:100]}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 500
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.debug(f"Deconstruction successful: {parsed['identified_entity']}")
                return DeconstructedQuery(**parsed)
            else:
                raise ValueError("No JSON found in Gemini response")
                
    except Exception as e:
        logger.warning(f"Deconstruction failed, using fallback: {e}")
        
        # Fallback: simple parsing
        entity = "Unknown"
        words = user_query.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 3:
                entity = word
                break
        
        return DeconstructedQuery(
            user_intent="Get support or help",
            identified_entity=entity,
            specific_details=[user_query[:100]],
            inhibitor="Unclear issue"
        )