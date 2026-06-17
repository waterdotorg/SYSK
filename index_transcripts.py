#!/usr/bin/env python3
"""
Offline transcript indexer for the SYSK RAG project.

This replaces the in-app "Index Next Batch" button (COWORK.md Priorities #1-#3):

  #1  Pinecone is the single source of truth. We ask Pinecone what is already
      indexed instead of trusting the drift-prone indexing_progress.json.
  #2  Index by diff, newest-first. We process only (disk - Pinecone), ordered by
      publish date descending, so a weekly run touches just the genuinely new
      episodes instead of walking a positional cursor through the whole backlog.
  #3  Indexing runs offline. This is a standalone CLI (run locally or from a
      GitHub Action); the Streamlit app stays read-only against Pinecone.

Usage
-----
    export PINECONE_API_KEY="..."
    python3 index_transcripts.py                 # index everything new
    python3 index_transcripts.py --dry-run       # show what would be indexed
    python3 index_transcripts.py --limit 50      # cap files this run
    python3 index_transcripts.py --transcripts-folder ./transcripts

Exit codes: 0 success (including "nothing to do"), 1 on error.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from transcript_processor import TranscriptProcessor
import pinecone_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_transcripts")

DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sysk-transcripts")
DEFAULT_TRANSCRIPTS_FOLDER = os.getenv("TRANSCRIPTS_FOLDER", "./transcripts")


def build_chunk_records(processor: TranscriptProcessor, file_path: Path):
    """Parse + chunk one transcript file into (ids, documents, metadatas).

    The vector ID and metadata shape MUST match the historical app logic so that
    IDs stay stable and pinecone_state can recover filenames from them.
    Returns (ids, documents, metadatas) or (None, None, None) if unusable.
    """
    parsed = processor.parse_transcript_file(str(file_path))
    if not parsed:
        return None, None, None

    chunks = processor.chunk_by_time(parsed['transcript'], parsed['metadata'])
    if not chunks:
        return None, None, None

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    for chunk in chunks:
        chunk_id = f"{parsed['metadata']['filename']}_{chunk['chunk_id']}"

        chunk_metadata = {
            'title': parsed['metadata'].get('title', 'Unknown'),
            'date': parsed['metadata'].get('date', 'Unknown'),
            'duration': parsed['metadata'].get('duration', 'Unknown'),
            'episode_type': parsed['metadata'].get('episode_type', 'Full Episode'),
            'filename': parsed['metadata'].get('filename', ''),
            'time': f"{chunk.get('start_time', '00:00:00')} - {chunk.get('end_time', '00:00:00')}",
            'chunk_id': chunk['chunk_id'],
        }
        for url_key in ('episode_url', 'audio_url', 'transcript_url'):
            if url_key in parsed['metadata']:
                chunk_metadata[url_key] = parsed['metadata'][url_key]

        ids.append(chunk_id)
        documents.append(chunk['text'])
        metadatas.append(chunk_metadata)

    return ids, documents, metadatas


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline diff-based SYSK transcript indexer.")
    parser.add_argument("--transcripts-folder", default=DEFAULT_TRANSCRIPTS_FOLDER,
                        help="Folder containing *.txt transcript files.")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME,
                        help="Pinecone index name.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of NEW files to index this run (0 = no limit).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the diff without writing to Pinecone.")
    args = parser.parse_args()

    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY is not set. export PINECONE_API_KEY='...' and retry.")
        return 1

    transcripts_dir = Path(args.transcripts_folder)
    if not transcripts_dir.exists():
        logger.error("Transcripts folder not found: %s", transcripts_dir)
        return 1

    disk_files = {f.name for f in transcripts_dir.glob("*.txt")}
    logger.info("Found %d transcript files on disk.", len(disk_files))

    # --- Connect to Pinecone and ask it what is already indexed (source of truth) ---
    # Imported here so --help works without the heavy sentence-transformers import.
    from pinecone_vector_db import PineconeVectorDatabase

    logger.info("Connecting to Pinecone index '%s' and loading the embedding model...", args.index_name)
    db = PineconeVectorDatabase(index_name=args.index_name)

    logger.info("Enumerating indexed episodes from Pinecone (source of truth)...")
    indexed_files = pinecone_state.get_indexed_filenames(db.index)
    logger.info("Pinecone already has %d distinct episodes indexed.", len(indexed_files))

    # --- Diff, newest-first (Priority #2) ---
    to_index = pinecone_state.diff_to_index(disk_files, indexed_files)

    # Sanity check: files Pinecone knows about that are missing on disk (informational).
    orphans = sorted(set(indexed_files) - disk_files)
    if orphans:
        logger.warning("%d episode(s) are in Pinecone but no longer on disk (e.g. %s).",
                       len(orphans), orphans[0])

    if not to_index:
        logger.info("Nothing to index — Pinecone is up to date with disk. ✅")
        return 0

    if args.limit and args.limit > 0:
        capped = to_index[:args.limit]
        logger.info("%d new episode(s) to index; capping to %d this run.", len(to_index), len(capped))
        to_index = capped
    else:
        logger.info("%d new episode(s) to index (newest first).", len(to_index))

    logger.info("First few to index: %s", ", ".join(to_index[:5]))

    if args.dry_run:
        logger.info("--dry-run set; not writing to Pinecone. Would index %d episode(s).", len(to_index))
        return 0

    processor = TranscriptProcessor()
    indexed_count = 0
    chunk_count = 0
    errors: List[str] = []
    start = time.time()

    for i, filename in enumerate(to_index, 1):
        file_path = transcripts_dir / filename
        try:
            ids, documents, metadatas = build_chunk_records(processor, file_path)
            if not ids:
                errors.append(f"{filename}: no chunks created (empty/too short?)")
                logger.warning("[%d/%d] %s -> no chunks, skipped.", i, len(to_index), filename)
                continue

            db.add(documents=documents, metadatas=metadatas, ids=ids)
            indexed_count += 1
            chunk_count += len(ids)
            logger.info("[%d/%d] %s -> %d chunks upserted.", i, len(to_index), filename, len(ids))
        except Exception as e:  # keep going; one bad file shouldn't stop the run
            errors.append(f"{filename}: {e}")
            logger.error("[%d/%d] %s -> ERROR: %s", i, len(to_index), filename, e)

    elapsed = time.time() - start
    logger.info("Done. Indexed %d episode(s), %d chunks, in %.1fs.", indexed_count, chunk_count, elapsed)
    if errors:
        logger.warning("%d file(s) had errors:", len(errors))
        for err in errors:
            logger.warning("  - %s", err)

    # Non-fatal: report success even with per-file errors so a weekly job keeps moving.
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(1)
