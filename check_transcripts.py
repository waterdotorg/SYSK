#!/usr/bin/env python3
"""
Quick check: Do we have a hypermiling transcript file?
"""

import os
from pathlib import Path

def check_transcripts():
    """Check transcript folder for hypermiling"""
    
    transcripts_folder = "./transcripts"
    
    print("="*80)
    print("CHECKING TRANSCRIPT FILES")
    print("="*80)
    
    if not os.path.exists(transcripts_folder):
        print(f"❌ Folder '{transcripts_folder}' not found")
        print("   Is the path correct? Try running from the same directory as your chatbot.")
        return
    
    # Get all .txt files
    files = list(Path(transcripts_folder).glob("*.txt"))
    print(f"\n✓ Found {len(files)} transcript files")
    
    # Search for hypermiling
    print("\nSearching for 'hypermiling' in filenames...")
    hypermiling_files = [f for f in files if 'hypermiling' in f.name.lower()]
    
    if hypermiling_files:
        print(f"✓ Found {len(hypermiling_files)} file(s):")
        for f in hypermiling_files:
            print(f"\n  📄 {f.name}")
            print(f"     Size: {f.stat().st_size} bytes")
            
            # Check content
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                    print(f"     Length: {len(content)} characters")
                    
                    # Check if 'hypermiling' appears in content
                    count = content.lower().count('hypermiling')
                    print(f"     'hypermiling' appears {count} times in content")
                    
                    # Show first few lines
                    lines = content.split('\n')[:10]
                    print(f"\n     First few lines:")
                    for line in lines:
                        if line.strip():
                            print(f"       {line[:80]}")
            except Exception as e:
                print(f"     ❌ Error reading file: {e}")
    else:
        print("❌ No files with 'hypermiling' in filename")
        print("\n   Checking for related terms in filenames...")
        
        related_terms = ['fuel', 'efficiency', 'gas', 'mileage', 'mpg']
        for term in related_terms:
            matches = [f for f in files if term in f.name.lower()]
            if matches:
                print(f"\n   '{term}' found in {len(matches)} filename(s):")
                for f in matches[:3]:
                    print(f"     - {f.name}")
        
        print("\n   Checking content of ALL files (this may take a moment)...")
        hypermiling_in_content = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if 'hypermiling' in content.lower():
                        hypermiling_in_content.append(f)
            except:
                pass
        
        if hypermiling_in_content:
            print(f"\n   ✓ Found 'hypermiling' in content of {len(hypermiling_in_content)} file(s):")
            for f in hypermiling_in_content:
                print(f"     📄 {f.name}")
        else:
            print("\n   ❌ 'hypermiling' not found in any transcript content")
            print("   This episode may not have been scraped yet!")
    
    # Show some sample filenames
    print("\n" + "-"*80)
    print("SAMPLE TRANSCRIPT FILENAMES (first 10):")
    for i, f in enumerate(files[:10], 1):
        print(f"{i}. {f.name}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    
    if hypermiling_files or hypermiling_in_content:
        print("""
✓ Hypermiling transcript EXISTS
  → Episode is in your transcripts folder
  → Problem is likely with indexing or search
  → Run: python debug_search.py to check ChromaDB
  → Make sure you clicked "Index" or "Re-index" in the app
        """)
    else:
        print("""
❌ Hypermiling transcript NOT FOUND
  → Episode may not have been scraped
  → Check your spreadsheet to confirm episode exists
  → Run the scraper to download this episode:
    python sysk_scraper_v5_rss.py
  → After scraping, click "Index" in the chatbot to add it to the database
        """)


if __name__ == "__main__":
    check_transcripts()
