#!/usr/bin/env python3
"""
Hybrid Search Module for SYSK RAG Chatbot
Combines semantic (vector) search with keyword (text) search
"""

import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class HybridSearcher:
    """
    Combines semantic search (ChromaDB) with keyword search
    """
    
    def __init__(self, collection):
        """
        Args:
            collection: ChromaDB collection with transcript chunks
        """
        self.collection = collection
        
    def keyword_search(self, query: str, n_results: int = 10, title_weight: float = 5.0) -> Dict[str, Any]:
        """
        Perform keyword/exact text search on transcript chunks AND episode titles
        Title matches get configurable weighting
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            title_weight: Multiplier for title matches (default 5.0 = heavy boost, 1.0 = equal to content)
            
        Returns:
            Dictionary with matched chunks, formatted like ChromaDB results
        """
        # Normalize query for matching
        query_lower = query.lower()
        query_terms = re.findall(r'\b\w+\b', query_lower)
        
        # Filter out common stop words that don't add semantic value
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'can', 'could', 'may', 'might', 'must', 'shall',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their',
                     'this', 'that', 'these', 'those', 'of', 'to', 'for', 'in', 'on', 'at',
                     'by', 'with', 'from', 'about', 'as', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'between', 'under', 'over',
                     'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whom',
                     'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                     'work', 'works', 'mean', 'means'}  # Generic verbs that appear in many titles
        
        # Keep original terms for exact phrase matching
        query_terms_original = query_terms
        # Filter for scoring (keep important terms)
        query_terms_filtered = [term for term in query_terms if term not in stop_words]
        
        # If filtering removed all terms, use original
        if not query_terms_filtered:
            query_terms_filtered = query_terms_original
        
        # Get all documents from ChromaDB
        all_docs = self.collection.get()
        
        # Score each document
        matches = []
        for idx, (doc_id, document, metadata) in enumerate(zip(
            all_docs['ids'],
            all_docs['documents'],
            all_docs['metadatas']
        )):
            doc_lower = document.lower()
            
            # Calculate content match score (using filtered terms)
            content_score = self._calculate_keyword_score(doc_lower, query_lower, query_terms_filtered)
            
            # NEW: Calculate title match score with configurable weighting
            title = metadata.get('title', metadata.get('episode_title', ''))
            title_lower = title.lower()
            title_score = self._calculate_keyword_score(title_lower, query_lower, query_terms_filtered)
            
            # Apply configurable multiplier to title matches
            title_score_weighted = title_score * title_weight
            
            # Combine scores (title matches dominate)
            total_score = content_score + title_score_weighted
            
            # Determine match type
            if title_score > 0 and content_score > 0:
                match_type = "title_and_content"
            elif title_score > 0:
                match_type = "title_match"
            elif content_score > 0:
                match_type = self._get_match_type(doc_lower, query_lower, query_terms_filtered)
            else:
                match_type = "no_match"
            
            if total_score > 0:
                matches.append({
                    'id': doc_id,
                    'document': document,
                    'metadata': metadata,
                    'score': total_score,
                    'match_type': match_type,
                    'title_score': title_score,
                    'content_score': content_score
                })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N results in ChromaDB-like format
        top_matches = matches[:n_results]
        
        return {
            'ids': [[m['id'] for m in top_matches]],
            'documents': [[m['document'] for m in top_matches]],
            'metadatas': [[m['metadata'] for m in top_matches]],
            'distances': [[1.0 - min(m['score'] / 20.0, 1.0) for m in top_matches]],  # Normalize score to distance
            'match_types': [m['match_type'] for m in top_matches]
        }
    
    def _calculate_keyword_score(self, doc_lower: str, query_lower: str, query_terms: List[str]) -> float:
        """
        Calculate relevance score for keyword matching
        
        Scoring:
        - Exact phrase match: 10.0
        - All terms present: 5.0 + term frequency bonus
        - Partial match: proportional score (no minimum threshold)
        """
        # Check for exact phrase match
        if query_lower in doc_lower:
            # Count occurrences for bonus
            occurrences = doc_lower.count(query_lower)
            return 10.0 + (occurrences - 1) * 2.0
        
        # Check for individual term matches
        term_matches = 0
        total_frequency = 0
        
        for term in query_terms:
            if term in doc_lower:
                term_matches += 1
                # Count frequency (but cap to avoid over-weighting)
                freq = min(doc_lower.count(term), 5)
                total_frequency += freq
        
        if term_matches == 0:
            return 0.0
        
        # Base score from percentage of terms matched
        # Changed: Now any partial match gets scored (no 85% threshold)
        base_score = (term_matches / len(query_terms)) * 5.0
        
        # Bonus for term frequency
        frequency_bonus = (total_frequency / len(query_terms)) * 0.5
        
        # Bonus for matching important terms (longer words are more important)
        important_terms_matched = sum(1 for term in query_terms if term in doc_lower and len(term) > 4)
        importance_bonus = important_terms_matched * 0.5
        
        return base_score + frequency_bonus + importance_bonus
    
    def _get_match_type(self, doc_lower: str, query_lower: str, query_terms: List[str]) -> str:
        """Determine what type of match this is"""
        if query_lower in doc_lower:
            return "exact_phrase"
        
        term_matches = sum(1 for term in query_terms if term in doc_lower)
        
        if term_matches == len(query_terms):
            return "all_terms"
        elif term_matches > 0:
            return f"partial_{term_matches}/{len(query_terms)}_terms"
        else:
            return "no_match"
    
    def semantic_search(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Perform semantic search using ChromaDB embeddings
        
        Args:
            query: Search query
            n_results: Maximum number of results
            
        Returns:
            ChromaDB query results
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def hybrid_search(self, query: str, n_results: int = 10, 
                     semantic_weight: float = 0.5,
                     keyword_weight: float = 0.5,
                     title_boost: float = 2.0,
                     title_weight: float = 2.0) -> Dict[str, Any]:
        """
        Perform hybrid search combining semantic and keyword approaches
        NOW WITH TITLE MATCHING: Title matches get extra boost
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            semantic_weight: Weight for semantic search (0.0 to 1.0)
            keyword_weight: Weight for keyword search (0.0 to 1.0)
            title_boost: Additional multiplier for title matches (default 2.0)
            title_weight: Multiplier for title matches in keyword search
            
        Returns:
            Merged and ranked results with source indicators
        """
        # Run both searches
        semantic_results = self.semantic_search(query, n_results=n_results)
        keyword_results = self.keyword_search(query, n_results=n_results, title_weight=title_weight)
        
        # Merge results
        merged = self._merge_results(
            semantic_results, 
            keyword_results,
            semantic_weight,
            keyword_weight,
            n_results,
            title_boost
        )
        
        return merged
    
    def _merge_results(self, semantic_results: Dict, keyword_results: Dict,
                      semantic_weight: float, keyword_weight: float,
                      n_results: int, title_boost: float = 2.0) -> Dict[str, Any]:
        """
        Merge and rank results from both search methods
        Apply title_boost to results with title matches
        """
        # Build a map of document_id -> combined score
        scores = defaultdict(lambda: {'semantic': 0, 'keyword': 0, 'data': None, 'title_match': False})
        
        # Process semantic results (distances are 0=perfect, higher=worse)
        if semantic_results['ids'] and semantic_results['ids'][0]:
            for idx, doc_id in enumerate(semantic_results['ids'][0]):
                # Convert distance to similarity score (invert and normalize)
                distance = semantic_results['distances'][0][idx]
                similarity = max(0, 1.0 - distance)  # 1=perfect, 0=poor
                
                scores[doc_id]['semantic'] = similarity
                scores[doc_id]['data'] = {
                    'id': doc_id,
                    'document': semantic_results['documents'][0][idx],
                    'metadata': semantic_results['metadatas'][0][idx],
                }
        
        # Process keyword results (distances are already converted from scores)
        if keyword_results['ids'] and keyword_results['ids'][0]:
            for idx, doc_id in enumerate(keyword_results['ids'][0]):
                # Convert distance back to score
                distance = keyword_results['distances'][0][idx]
                score = max(0, 1.0 - distance)
                
                scores[doc_id]['keyword'] = score
                scores[doc_id]['match_type'] = keyword_results['match_types'][idx]
                
                # Check if this is a title match
                if 'title' in keyword_results['match_types'][idx]:
                    scores[doc_id]['title_match'] = True
                
                # If not in semantic results, add the data
                if scores[doc_id]['data'] is None:
                    scores[doc_id]['data'] = {
                        'id': doc_id,
                        'document': keyword_results['documents'][0][idx],
                        'metadata': keyword_results['metadatas'][0][idx],
                    }
        
        # Calculate combined scores with title boost
        ranked_results = []
        for doc_id, info in scores.items():
            combined_score = (
                info['semantic'] * semantic_weight +
                info['keyword'] * keyword_weight
            )
            
            # Apply title boost if this result has a title match
            if info['title_match']:
                combined_score *= title_boost
            
            # Determine primary match source
            if info['semantic'] > info['keyword']:
                primary_source = 'semantic'
            elif info['keyword'] > info['semantic']:
                primary_source = 'keyword'
            else:
                primary_source = 'both'
            
            ranked_results.append({
                'id': doc_id,
                'data': info['data'],
                'combined_score': combined_score,
                'semantic_score': info['semantic'],
                'keyword_score': info['keyword'],
                'primary_source': primary_source,
                'match_type': info.get('match_type', 'semantic_only'),
                'title_match': info['title_match']
            })
        
        # Sort by combined score
        ranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top N
        top_results = ranked_results[:n_results]
        
        # Format as ChromaDB-like output with additional metadata
        return {
            'ids': [[r['id'] for r in top_results]],
            'documents': [[r['data']['document'] for r in top_results]],
            'metadatas': [[r['data']['metadata'] for r in top_results]],
            'distances': [[1.0 - min(r['combined_score'], 1.0) for r in top_results]],
            'scores': {
                'combined': [r['combined_score'] for r in top_results],
                'semantic': [r['semantic_score'] for r in top_results],
                'keyword': [r['keyword_score'] for r in top_results],
            },
            'sources': [r['primary_source'] for r in top_results],
            'match_types': [r['match_type'] for r in top_results]
        }
    
    def search_with_fallback(self, query: str, n_results: int = 10,
                            semantic_threshold: float = 0.5,
                            title_weight: float = 5.0) -> Tuple[Dict[str, Any], str]:
        """
        Smart search with intelligent multi-strategy approach
        
        NEW STRATEGY:
        1. Always run keyword search (fast, checks titles)
        2. If keyword finds good title matches, use those
        3. Otherwise run semantic search
        4. If both are mediocre, use hybrid
        
        This ensures title matches are never missed!
        
        Args:
            query: Search query
            n_results: Number of results
            semantic_threshold: Minimum quality threshold for semantic results
            title_weight: Multiplier for title matches in keyword search
            
        Returns:
            (results, search_method_used)
        """
        # ALWAYS run keyword search first (it's fast and checks titles)
        keyword_results = self.keyword_search(query, n_results, title_weight=title_weight)
        
        # Check if keyword search found good title matches
        keyword_has_title_matches = False
        if keyword_results['ids'] and keyword_results['ids'][0]:
            for match_type in keyword_results.get('match_types', []):
                if 'title' in match_type:
                    keyword_has_title_matches = True
                    break
        
        # If keyword found title matches, use those (high confidence)
        if keyword_has_title_matches:
            return keyword_results, "keyword"
        
        # Otherwise, try semantic search
        semantic_results = self.semantic_search(query, n_results)
        
        # Check quality of semantic results
        if semantic_results['ids'] and semantic_results['ids'][0]:
            best_distance = semantic_results['distances'][0][0]
            best_similarity = 1.0 - best_distance
            
            # If semantic results are good, use those
            if best_similarity >= semantic_threshold:
                # But check if keyword has ANY matches (even content-only)
                if keyword_results['ids'] and keyword_results['ids'][0]:
                    # Both have results - use hybrid to combine them
                    hybrid_results = self.hybrid_search(query, n_results, 
                                                       semantic_weight=0.6, 
                                                       keyword_weight=0.4)
                    return hybrid_results, "hybrid"
                else:
                    # Only semantic has results
                    return semantic_results, "semantic"
        
        # If semantic was poor, check keyword match quality before using it
        if keyword_results['ids'] and keyword_results['ids'][0]:
            # Calculate keyword result quality
            best_keyword_distance = keyword_results['distances'][0][0]
            keyword_score = 1.0 - best_keyword_distance
            
            # Only use keyword if it has STRONG matches (> 0.5 score)
            # This prevents weak keyword content matches from overriding semantic
            if keyword_score > 0.5:
                return keyword_results, "keyword"
        
        # Semantic is decent OR keyword is weak - use hybrid to combine signals
        # Favor semantic for conceptual queries
        hybrid_results = self.hybrid_search(query, n_results,
                                           semantic_weight=0.7,
                                           keyword_weight=0.3)
        return hybrid_results, "hybrid"


def format_search_results_for_display(results: Dict[str, Any], 
                                      search_method: str,
                                      show_scores: bool = True) -> str:
    """
    Format search results for display in Streamlit
    
    Args:
        results: Search results dictionary
        search_method: Which search method was used
        show_scores: Whether to show relevance scores
        
    Returns:
        Formatted string for display
    """
    if not results['ids'] or not results['ids'][0]:
        return "No results found."
    
    output = []
    output.append(f"**Search Method:** {search_method.upper()}\n")
    output.append(f"**Found {len(results['ids'][0])} results**\n")
    
    for idx in range(len(results['ids'][0])):
        output.append(f"\n---\n**Result {idx + 1}**")
        
        # Show metadata
        metadata = results['metadatas'][0][idx]
        output.append(f"**Episode:** {metadata.get('episode_title', 'Unknown')}")
        output.append(f"**Date:** {metadata.get('episode_date', 'Unknown')}")
        
        if 'timestamp_start' in metadata:
            output.append(f"**Time:** {metadata['timestamp_start']}")
        
        # Show scores if available and requested
        if show_scores:
            if 'scores' in results:
                output.append(f"**Relevance:** {results['scores']['combined'][idx]:.2%}")
                output.append(f"  - Semantic: {results['scores']['semantic'][idx]:.2%}")
                output.append(f"  - Keyword: {results['scores']['keyword'][idx]:.2%}")
            else:
                distance = results['distances'][0][idx]
                similarity = max(0, 1.0 - distance)
                output.append(f"**Relevance:** {similarity:.2%}")
        
        # Show source/match type
        if 'sources' in results:
            source = results['sources'][idx]
            match_type = results['match_types'][idx]
            output.append(f"**Match:** {source} ({match_type})")
        
        # Show excerpt
        document = results['documents'][0][idx]
        preview = document[:300] + "..." if len(document) > 300 else document
        output.append(f"\n{preview}")
    
    return "\n".join(output)
