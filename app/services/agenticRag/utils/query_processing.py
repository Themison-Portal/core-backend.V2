"""
Query processing utilities for retrieval
"""

from typing import Any, Dict, List


def summarize_long_query(query: str, max_length: int = 200) -> str:
    """
    Summarize a very long query to its key points
    This is useful for extremely long queries that might lose focus
    """
    if len(query) <= max_length:
        return query
    
    # Simple summarization: take first and last sentences
    sentences = query.split('. ')
    if len(sentences) <= 2:
        return query[:max_length] + "..."
    
    # Take first sentence and last sentence
    summary = sentences[0] + ". " + sentences[-1]
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def extract_key_phrases(query: str) -> List[str]:
    """
    Extract key phrases from a query for targeted search
    """
    # Simple key phrase extraction (you could use NLP libraries for better results)
    # Remove common stop words and extract meaningful phrases
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = query.lower().split()
    key_phrases = []
    current_phrase = []
    
    for word in words:
        if word not in stop_words and len(word) > 2:
            current_phrase.append(word)
        elif current_phrase:
            if len(current_phrase) >= 2:
                key_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    # Add the last phrase if it exists
    if current_phrase and len(current_phrase) >= 2:
        key_phrases.append(' '.join(current_phrase))
    
    return key_phrases


def create_query_variations(query: str) -> List[str]:
    """
    Create different variations of a query for better retrieval
    """
    variations = [query]
    
    # Add key phrase variations
    key_phrases = extract_key_phrases(query)
    for phrase in key_phrases[:3]:  # Limit to top 3 phrases
        variations.append(phrase)
    
    # Add summarized version for very long queries
    if len(query) > 500:
        summary = summarize_long_query(query)
        variations.append(summary)
    
    return variations


async def multi_query_search(
    query: str,
    search_function,
    max_variations: int = 3
) -> List[Dict[Any, Any]]:
    """
    Search using multiple query variations and aggregate results
    """
    variations = create_query_variations(query)[:max_variations]
    
    all_results = []
    for variation in variations:
        results = await search_function(variation)
        all_results.extend(results)
    
    # Deduplicate and rank results
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        result_id = result.get('id') or result.get('content', '')[:100]
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            unique_results.append(result)
    
    return unique_results[:10]  # Return top 10 unique results 