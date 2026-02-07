#!/usr/bin/env python3
"""
Sync indexing_progress.json with actual Pinecone database state
Use this to fix mismatches between local and cloud indexing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from pinecone import Pinecone
except ImportError:
    print("ERROR: pinecone-client not installed. Run: pip install pinecone-client")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration from environment variables"""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sysk-transcripts")
    TRANSCRIPTS_FOLDER = os.getenv("TRANSCRIPTS_FOLDER", "./transcripts")
    PROGRESS_FILE = "indexing_progress.json"

# ============================================================================
# Sync Functions
# ============================================================================

def get_indexed_files_from_pinecone(index) -> set:
    """
    Query Pinecone to get all unique filenames that have been indexed
    
    Note: This samples the database since Pinecone doesn't have a "get all metadata" function.
    We'll query a large sample and extract unique filenames.
    """
    print("🔍 Querying Pinecone to find indexed files...")
    print("   (This may take a minute...)")
    
    indexed_files = set()
    
    try:
        # Get index stats first
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"   Total vectors in Pinecone: {total_vectors:,}")
        
        # Fetch a sample of vectors to extract filenames
        # We'll use the fetch API to get metadata from known IDs
        # But since we don't know all IDs, we'll use query with a dummy vector
        
        # Create a dummy query vector (all zeros is fine for metadata extraction)
        dummy_vector = [0.0] * 384  # Match your embedding dimension
        
        # Query multiple times with different filters to get broad coverage
        batch_size = 10000  # Pinecone max is 10,000
        sample_size = min(batch_size, total_vectors)
        
        print(f"   Sampling up to {sample_size:,} vectors to extract filenames...")
        
        # Query Pinecone
        results = index.query(
            vector=dummy_vector,
            top_k=sample_size,
            include_metadata=True
        )
        
        # Extract unique filenames from metadata
        if results and 'matches' in results:
            for match in results['matches']:
                if 'metadata' in match and 'filename' in match['metadata']:
                    filename = match['metadata']['filename']
                    indexed_files.add(filename)
        
        print(f"   ✓ Found {len(indexed_files)} unique indexed files in sample")
        
        # If sample is much smaller than total files, warn user
        if len(indexed_files) < 100 and total_vectors > 1000:
            print(f"   ⚠️  Warning: Only found {len(indexed_files)} files in sample.")
            print(f"      This might not be comprehensive. Consider running this script")
            print(f"      multiple times or use the 'all files' option below.")
        
        return indexed_files
        
    except Exception as e:
        print(f"   ✗ Error querying Pinecone: {e}")
        return set()

def get_all_transcript_files(transcripts_folder: str) -> set:
    """Get all transcript filenames from disk"""
    print(f"\n📁 Scanning transcript folder: {transcripts_folder}")
    
    transcript_dir = Path(transcripts_folder)
    if not transcript_dir.exists():
        print(f"   ✗ Folder not found!")
        return set()
    
    all_files = [f.name for f in transcript_dir.glob("*.txt")]
    print(f"   ✓ Found {len(all_files)} transcript files on disk")
    
    return set(all_files)

def load_current_progress() -> dict:
    """Load current indexing progress file"""
    if os.path.exists(Config.PROGRESS_FILE):
        try:
            with open(Config.PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"indexed_files": [], "last_updated": None, "total_indexed": 0}

def save_progress(progress: dict):
    """Save updated progress file"""
    progress["last_updated"] = datetime.now().isoformat()
    
    # Add sync metadata
    progress["synced_from_db"] = True
    progress["sync_timestamp"] = datetime.now().isoformat()
    
    with open(Config.PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n✓ Saved updated progress to {Config.PROGRESS_FILE}")

# ============================================================================
# Main Sync Logic
# ============================================================================

def sync_progress():
    """Main function to sync progress file with Pinecone"""
    
    print("="*80)
    print("SYSK Indexing Progress Sync Tool")
    print("="*80)
    
    # Validate environment
    if not Config.PINECONE_API_KEY:
        print("\n❌ ERROR: PINECONE_API_KEY environment variable not set")
        print("   Set it with: export PINECONE_API_KEY='your-key'")
        sys.exit(1)
    
    # Connect to Pinecone
    print(f"\n🔗 Connecting to Pinecone...")
    print(f"   Index: {Config.PINECONE_INDEX_NAME}")
    
    try:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"\n❌ ERROR: Index '{Config.PINECONE_INDEX_NAME}' not found!")
            sys.exit(1)
        
        index = pc.Index(Config.PINECONE_INDEX_NAME)
        print(f"   ✓ Connected successfully")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to connect to Pinecone: {e}")
        sys.exit(1)
    
    # Get current progress
    current_progress = load_current_progress()
    current_indexed = set(current_progress.get("indexed_files", []))
    print(f"\n📊 Current indexing_progress.json:")
    print(f"   Files tracked: {len(current_indexed)}")
    
    # Show user options
    print("\n" + "="*80)
    print("SYNC OPTIONS:")
    print("="*80)
    print("1. Sample Pinecone (query 10k vectors to find indexed files)")
    print("2. Assume all transcript files are indexed (fastest)")
    print("3. Exit without changes")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        # Query Pinecone to find indexed files
        indexed_in_db = get_indexed_files_from_pinecone(index)
        
        if not indexed_in_db:
            print("\n❌ Could not retrieve indexed files from Pinecone")
            sys.exit(1)
        
        # Compare with disk
        all_files = get_all_transcript_files(Config.TRANSCRIPTS_FOLDER)
        
        # Update progress
        new_progress = {
            "indexed_files": sorted(list(indexed_in_db)),
            "total_indexed": len(indexed_in_db),
            "note": f"Synced from {len(indexed_in_db)} files found in Pinecone sample"
        }
        
        # Show comparison
        print("\n" + "="*80)
        print("COMPARISON:")
        print("="*80)
        print(f"Previously tracked: {len(current_indexed)} files")
        print(f"Found in Pinecone:  {len(indexed_in_db)} files")
        print(f"On disk:            {len(all_files)} files")
        
        added = indexed_in_db - current_indexed
        removed = current_indexed - indexed_in_db
        
        if added:
            print(f"\n✓ Will ADD {len(added)} files to progress.json")
        if removed:
            print(f"\n⚠️  Will REMOVE {len(removed)} files from progress.json")
        
    elif choice == "2":
        # Assume all files on disk are indexed
        all_files = get_all_transcript_files(Config.TRANSCRIPTS_FOLDER)
        
        if not all_files:
            print("\n❌ No transcript files found on disk")
            sys.exit(1)
        
        new_progress = {
            "indexed_files": sorted(list(all_files)),
            "total_indexed": len(all_files),
            "note": "Assumed all transcript files are indexed"
        }
        
        # Show comparison
        print("\n" + "="*80)
        print("COMPARISON:")
        print("="*80)
        print(f"Previously tracked: {len(current_indexed)} files")
        print(f"Will track:         {len(all_files)} files (all on disk)")
        
        added = all_files - current_indexed
        removed = current_indexed - all_files
        
        if added:
            print(f"\n✓ Will ADD {len(added)} files to progress.json")
            if len(added) <= 20:
                print("\nNew files to add:")
                for f in sorted(list(added))[:20]:
                    print(f"  - {f}")
        if removed:
            print(f"\n⚠️  Will REMOVE {len(removed)} files from progress.json")
    
    else:
        print("\nExiting without changes.")
        return
    
    # Confirm
    print("\n" + "="*80)
    confirm = input("Apply these changes? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        save_progress(new_progress)
        print("\n✅ Sync complete!")
        print(f"   Total files now tracked: {new_progress['total_indexed']}")
    else:
        print("\n❌ Changes cancelled.")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        sync_progress()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
