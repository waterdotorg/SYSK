#!/usr/bin/env python3
"""
Test the updated hybrid search with title matching
"""

import chromadb
from hybrid_search import HybridSearcher

def test_title_search():
    """Test that searching for 'hypermiling' now finds the episode via title"""
    
    print("="*80)
    print("TESTING TITLE-ENHANCED SEARCH")
    print("="*80)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("sysk_transcripts")
    
    # Initialize hybrid searcher
    searcher = HybridSearcher(collection)
    
    # Test query
    query = "hypermiling"
    
    print(f"\nQuery: '{query}'")
    print("="*80)
    
    # Test 1: Keyword-only search (should now find it via title)
    print("\n1️⃣ KEYWORD SEARCH (with title matching)")
    results = searcher.keyword_search(query, n_results=5)
    
    if results['documents'] and results['documents'][0]:
        print(f"✓ Found {len(results['documents'][0])} results\n")
        for i, (doc, meta, match_type) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['match_types']
        ), 1):
            title = meta.get('title', 'Unknown')
            print(f"  Result {i}: {title}")
            print(f"  Match type: {match_type}")
            print(f"  Content preview: {doc[:100]}...")
            print()
    else:
        print("❌ No results found")
    
    # Test 2: Smart search
    print("\n2️⃣ SMART SEARCH (auto-fallback)")
    results, method = searcher.search_with_fallback(query, n_results=5)
    
    print(f"Method used: {method}")
    if results['documents'] and results['documents'][0]:
        print(f"✓ Found {len(results['documents'][0])} results\n")
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            title = meta.get('title', 'Unknown')
            print(f"  Result {i}: {title}")
            if 'hypermiling' in title.lower():
                print("  ✓✓✓ CORRECT EPISODE FOUND!")
            print()
    else:
        print("❌ No results found")
    
    # Test 3: Hybrid search
    print("\n3️⃣ HYBRID SEARCH (semantic + keyword)")
    results = searcher.hybrid_search(
        query, 
        n_results=5,
        semantic_weight=0.3,
        keyword_weight=0.7  # Heavy keyword weight to favor title matches
    )
    
    if results['documents'] and results['documents'][0]:
        print(f"✓ Found {len(results['documents'][0])} results\n")
        for i, (doc, meta, match_type) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['match_types']
        ), 1):
            title = meta.get('title', 'Unknown')
            print(f"  Result {i}: {title}")
            print(f"  Match type: {match_type}")
            if 'hypermiling' in title.lower():
                print("  ✓✓✓ CORRECT EPISODE FOUND!")
            print()
    else:
        print("❌ No results found")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The updated hybrid search now:
1. Searches episode TITLES as well as content
2. Gives title matches 5x weight boost
3. Should find 'hypermiling' episode even though the word rarely appears in content

If you see "How can hypermiling save you gas?" in the results above, it's working!
    """)


if __name__ == "__main__":
    test_title_search()
