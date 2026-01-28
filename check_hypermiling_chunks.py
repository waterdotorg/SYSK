#!/usr/bin/env python3
"""
Check the hypermiling episode chunks in ChromaDB
"""

import chromadb

def check_hypermiling_chunks():
    """Check what chunks exist for the hypermiling episode"""
    
    print("="*80)
    print("HYPERMILING EPISODE CHUNKS INSPECTION")
    print("="*80)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("sysk_transcripts")
    
    # Get ALL documents
    print("\nFetching all documents (this may take a moment)...")
    all_docs = collection.get()
    
    # Find hypermiling chunks
    hypermiling_chunks = []
    for doc_id, document, metadata in zip(
        all_docs['ids'],
        all_docs['documents'],
        all_docs['metadatas']
    ):
        title = metadata.get('title', '')
        if 'hypermiling' in title.lower():
            hypermiling_chunks.append({
                'id': doc_id,
                'document': document,
                'metadata': metadata
            })
    
    if not hypermiling_chunks:
        print("❌ No chunks found for hypermiling episode!")
        print("   This means the episode exists in metadata but has no content chunks.")
        print("   The episode may not have been indexed properly.")
        return
    
    print(f"\n✓ Found {len(hypermiling_chunks)} chunks for hypermiling episode")
    print(f"   Title: {hypermiling_chunks[0]['metadata']['title']}")
    print(f"   Date: {hypermiling_chunks[0]['metadata']['date']}")
    
    # Check each chunk
    print("\n" + "="*80)
    print("CHUNK CONTENTS")
    print("="*80)
    
    for i, chunk in enumerate(hypermiling_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk['id']}")
        print(f"Start time: {chunk['metadata'].get('start_time', 'N/A')}")
        print(f"End time: {chunk['metadata'].get('end_time', 'N/A')}")
        print(f"Chunk ID: {chunk['metadata'].get('chunk_id', 'N/A')}")
        print(f"Length: {len(chunk['document'])} characters")
        
        # Check if "hypermiling" appears in the content
        count = chunk['document'].lower().count('hypermiling')
        print(f"'hypermiling' appears: {count} times")
        
        # Show content
        print(f"\nContent preview (first 500 chars):")
        print(chunk['document'][:500])
        print("...")
        
        if i >= 5:  # Show first 5 chunks in detail
            print(f"\n... and {len(hypermiling_chunks) - 5} more chunks")
            break
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    total_hypermiling_mentions = sum(
        chunk['document'].lower().count('hypermiling')
        for chunk in hypermiling_chunks
    )
    
    print(f"\nTotal chunks: {len(hypermiling_chunks)}")
    print(f"Total 'hypermiling' mentions in content: {total_hypermiling_mentions}")
    
    if total_hypermiling_mentions == 0:
        print("\n⚠️  PROBLEM FOUND:")
        print("   The word 'hypermiling' does NOT appear in any chunk content!")
        print("   It only appears in the metadata (title).")
        print("\n   This is why keyword search can't find it.")
        print("   The transcript file may have minimal use of the actual word,")
        print("   or the chunking may have separated the word from the main content.")
        print("\n   SOLUTION:")
        print("   1. Search for 'miles per gallon' instead (this works!)")
        print("   2. Search for 'fuel efficiency' or 'gas saving'")
        print("   3. Update the search to also search episode titles/metadata")
    else:
        print(f"\n✓ Word appears {total_hypermiling_mentions} times in content")
        print("   Keyword search should work - there may be a bug in the search code")
    
    # Show which terms DO appear frequently
    print("\n" + "-"*80)
    print("Common terms in hypermiling episode:")
    
    all_content = ' '.join(chunk['document'].lower() for chunk in hypermiling_chunks)
    
    test_terms = ['fuel', 'gas', 'efficiency', 'mileage', 'mpg', 'driving', 
                  'car', 'speed', 'miles', 'gallon']
    
    for term in test_terms:
        count = all_content.count(term)
        if count > 0:
            print(f"  '{term}': {count} times")

if __name__ == "__main__":
    check_hypermiling_chunks()
