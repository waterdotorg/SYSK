#!/usr/bin/env python3
"""
Debug script to troubleshoot search issues
Run this to see what's in your ChromaDB and test different searches
"""

import chromadb
from chromadb.utils import embedding_functions
import re

def inspect_database():
    """Inspect what's actually in the ChromaDB"""
    
    print("="*80)
    print("CHROMADB INSPECTION")
    print("="*80)
    
    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("sysk_transcripts")
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return None
    
    # Get basic stats
    count = collection.count()
    print(f"\n✓ Connected to ChromaDB")
    print(f"✓ Collection: sysk_transcripts")
    print(f"✓ Total chunks: {count}")
    
    # Get a sample of documents
    print("\n" + "="*80)
    print("SAMPLE DOCUMENTS (first 5)")
    print("="*80)
    
    sample = collection.get(limit=5)
    
    for i, (doc_id, document, metadata) in enumerate(zip(
        sample['ids'],
        sample['documents'],
        sample['metadatas']
    ), 1):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc_id}")
        print(f"Metadata keys: {list(metadata.keys())}")
        print(f"Episode: {metadata.get('title', metadata.get('episode_title', 'N/A'))}")
        print(f"Date: {metadata.get('date', metadata.get('episode_date', 'N/A'))}")
        print(f"Episode type: {metadata.get('episode_type', 'N/A')}")
        print(f"Content preview: {document[:200]}...")
    
    return collection


def search_for_hypermiling(collection):
    """Test different search methods for 'hypermiling'"""
    
    print("\n" + "="*80)
    print("SEARCHING FOR 'HYPERMILING'")
    print("="*80)
    
    query = "hypermiling"
    
    # Test 1: Semantic search
    print("\n1️⃣ SEMANTIC SEARCH (ChromaDB query)")
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"✓ Found {len(results['documents'][0])} results")
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"\n  Result {i}:")
                print(f"  Episode: {meta.get('title', meta.get('episode_title', 'N/A'))}")
                print(f"  Distance: {dist:.4f}")
                print(f"  Content: {doc[:150]}...")
        else:
            print("❌ No results found")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Keyword search - manual
    print("\n2️⃣ KEYWORD SEARCH (manual text search)")
    try:
        # Get all documents
        all_docs = collection.get()
        
        matches = []
        for doc_id, document, metadata in zip(
            all_docs['ids'],
            all_docs['documents'],
            all_docs['metadatas']
        ):
            if query.lower() in document.lower():
                matches.append({
                    'id': doc_id,
                    'episode': metadata.get('title', metadata.get('episode_title', 'Unknown')),
                    'preview': document[:200]
                })
        
        if matches:
            print(f"✓ Found {len(matches)} documents containing '{query}'")
            for i, match in enumerate(matches[:5], 1):
                print(f"\n  Match {i}:")
                print(f"  Episode: {match['episode']}")
                print(f"  Preview: {match['preview']}...")
        else:
            print(f"❌ No documents contain the exact text '{query}'")
    except Exception as e:
        print(f"❌ Error: {e}")


def search_episode_titles(collection):
    """Search for episodes with 'hypermiling' in the title"""
    
    print("\n" + "="*80)
    print("SEARCHING EPISODE TITLES")
    print("="*80)
    
    try:
        all_docs = collection.get()
        
        # Find unique episodes
        episodes = {}
        for metadata in all_docs['metadatas']:
            title = metadata.get('title', metadata.get('episode_title', ''))
            if title and title not in episodes:
                episodes[title] = metadata
        
        print(f"\n✓ Found {len(episodes)} unique episodes")
        
        # Search for hypermiling
        print("\nSearching for 'hypermiling' in titles...")
        hypermiling_episodes = [title for title in episodes.keys() if 'hypermiling' in title.lower()]
        
        if hypermiling_episodes:
            print(f"✓ Found {len(hypermiling_episodes)} episodes:")
            for title in hypermiling_episodes:
                print(f"  - {title}")
        else:
            print("❌ No episodes with 'hypermiling' in title")
        
        # Search for related terms
        print("\nSearching for related terms...")
        related_terms = ['fuel', 'efficiency', 'mileage', 'gas', 'mpg']
        for term in related_terms:
            matches = [title for title in episodes.keys() if term in title.lower()]
            if matches:
                print(f"\n'{term}' found in {len(matches)} episodes:")
                for title in matches[:3]:
                    print(f"  - {title}")
        
        # Show some sample titles
        print("\n" + "-"*80)
        print("SAMPLE EPISODE TITLES (first 20):")
        for i, title in enumerate(list(episodes.keys())[:20], 1):
            print(f"{i}. {title}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_specific_queries(collection):
    """Test various query patterns"""
    
    print("\n" + "="*80)
    print("TESTING VARIOUS QUERIES")
    print("="*80)
    
    test_queries = [
        "hypermiling",
        "how hypermiling works",
        "fuel efficiency",
        "saving gas",
        "miles per gallon"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                print(f"  ✓ Found {len(results['documents'][0])} results")
                for i, (meta, dist) in enumerate(zip(
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    episode = meta.get('title', meta.get('episode_title', 'Unknown'))
                    print(f"    {i}. {episode} (distance: {dist:.4f})")
            else:
                print("  ❌ No results")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def check_metadata_structure(collection):
    """Check what metadata fields are available"""
    
    print("\n" + "="*80)
    print("METADATA STRUCTURE ANALYSIS")
    print("="*80)
    
    sample = collection.get(limit=100)
    
    # Collect all unique metadata keys
    all_keys = set()
    for metadata in sample['metadatas']:
        all_keys.update(metadata.keys())
    
    print(f"\nFound {len(all_keys)} unique metadata fields:")
    for key in sorted(all_keys):
        print(f"  - {key}")
    
    # Check for title field variations
    print("\nChecking title field names:")
    title_fields = ['title', 'episode_title', 'episode', 'name']
    for field in title_fields:
        count = sum(1 for m in sample['metadatas'] if field in m)
        if count > 0:
            print(f"  ✓ '{field}': found in {count}/{len(sample['metadatas'])} documents")
            # Show example
            example = next((m[field] for m in sample['metadatas'] if field in m), None)
            if example:
                print(f"    Example: {example}")


def main():
    """Run all debug checks"""
    
    print("\n" + "🔍 SYSK SEARCH DEBUGGER" + "\n")
    
    # Inspect database
    collection = inspect_database()
    
    if collection is None:
        print("\n❌ Cannot proceed without database connection")
        return
    
    # Check metadata structure
    check_metadata_structure(collection)
    
    # Search for hypermiling
    search_for_hypermiling(collection)
    
    # Search episode titles
    search_episode_titles(collection)
    
    # Test various queries
    test_specific_queries(collection)
    
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the results above:

1. Check if 'hypermiling' episode is actually indexed
   - Look at episode titles list
   - If not found, you may need to re-scrape or re-index

2. Check metadata field names
   - Verify if it's 'title' or 'episode_title'
   - Update code if field names don't match

3. Check semantic search results
   - If semantic search returns unrelated results, embeddings may need improvement
   - Keyword search should catch exact matches

4. If episode exists but isn't found
   - Check the actual content in the transcripts folder
   - Verify the episode was scraped correctly
   - Try re-indexing that specific episode

Next steps:
- Run: ls transcripts/ | grep -i hypermiling
- Check if the transcript file exists and has content
- Re-index if needed
    """)


if __name__ == "__main__":
    main()
