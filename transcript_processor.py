"""
Transcript parsing and chunking for the SYSK RAG project.

This module is intentionally free of any Streamlit dependency so that both the
Streamlit app (sysk_rag_chatbot.py) and the offline indexer (index_transcripts.py)
share ONE implementation of parsing/chunking. Keeping a single source of truth for
chunk IDs is essential: the Pinecone vector ID is built as
``f"{filename}_{chunk_id}"`` and the rest of the system derives indexed-state from
those IDs.
"""

import os
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Chunking defaults (mirror the values in sysk_rag_chatbot.Config)
# ----------------------------------------------------------------------------
CHUNK_DURATION_SECONDS = 180   # 3 minutes per time-based chunk
CHUNK_SIZE_CHARS = 2000        # characters per character-based chunk
CHUNK_OVERLAP_CHARS = 200      # overlap between character-based chunks


class TranscriptProcessor:
    """Process SYSK transcript files into chunks."""

    def __init__(self, chunk_duration_seconds=None, chunk_size_chars=None,
                 chunk_overlap_chars=None):
        self.chunk_duration = chunk_duration_seconds or CHUNK_DURATION_SECONDS
        self.chunk_size = chunk_size_chars or CHUNK_SIZE_CHARS
        self.chunk_overlap = chunk_overlap_chars or CHUNK_OVERLAP_CHARS

    def parse_transcript_file(self, filepath: str) -> Optional[Dict]:
        """Parse a transcript file and extract metadata."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract metadata from header
            lines = content.split('\n')
            metadata = {}
            transcript_start = 0

            for i, line in enumerate(lines):
                if line.startswith("Episode:"):
                    metadata['title'] = line.replace("Episode:", "").strip()
                elif line.startswith("Date:"):
                    metadata['date'] = line.replace("Date:", "").strip()
                elif line.startswith("Duration:"):
                    duration_str = line.replace("Duration:", "").strip()
                    metadata['duration'] = duration_str
                elif line.startswith("Episode URL:"):
                    metadata['episode_url'] = line.replace("Episode URL:", "").strip()
                elif line.startswith("Audio URL:"):
                    metadata['audio_url'] = line.replace("Audio URL:", "").strip()
                elif line.startswith("Transcript URL:"):
                    metadata['transcript_url'] = line.replace("Transcript URL:", "").strip()
                elif "=" * 40 in line:
                    transcript_start = i + 1
                    break

            # Get transcript content after header
            transcript_text = '\n'.join(lines[transcript_start:]).strip()

            # Determine episode type
            if metadata.get('title', '').startswith("Short Stuff"):
                metadata['episode_type'] = "Short Stuff"
            else:
                metadata['episode_type'] = "Full Episode"

            # Extract filename for unique ID
            metadata['filename'] = os.path.basename(filepath)

            return {
                'metadata': metadata,
                'transcript': transcript_text,
                'filepath': filepath
            }

        except Exception as e:
            logger.warning("Error parsing %s: %s", filepath, e)
            return None

    def parse_timestamp(self, timestamp_str: str) -> int:
        """Convert timestamp string (HH:MM:SS) to seconds."""
        try:
            parts = timestamp_str.strip().split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + int(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + int(s)
            else:
                return 0
        except Exception:
            return 0

    def chunk_by_time(self, transcript_text: str, metadata: Dict) -> List[Dict]:
        """Chunk transcript by time intervals, with fallback for episodes without timestamps."""

        # Fix transcripts that don't have proper line breaks
        transcript_text = re.sub(r'(\d{2}:\d{2}:\d{2})([A-Za-z])', r'\n\1\n\2', transcript_text)

        # Count timestamps to determine chunking strategy
        timestamp_count = len(re.findall(r'\d{2}:\d{2}:\d{2}', transcript_text))

        # If episode has very few timestamps, use character-based chunking
        if timestamp_count < 5:
            return self.chunk_by_characters(transcript_text, metadata)

        # Otherwise use time-based chunking
        chunks = []
        lines = transcript_text.split('\n')

        current_chunk = []
        current_start_time = 0
        current_start_timestamp = "00:00:00"
        chunk_id = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a timestamp
            if re.match(r'^\d{2}:\d{2}:\d{2}$', line):
                timestamp_seconds = self.parse_timestamp(line)

                # If we've exceeded chunk duration, save current chunk
                if timestamp_seconds - current_start_time >= self.chunk_duration and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_time': current_start_timestamp,
                        'end_time': line,
                        'chunk_id': chunk_id,
                        'metadata': metadata
                    })

                    # Start new chunk
                    current_chunk = []
                    current_start_time = timestamp_seconds
                    current_start_timestamp = line
                    chunk_id += 1

                current_chunk.append(line)
            else:
                current_chunk.append(line)

        # Save final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_time': current_start_timestamp,
                'end_time': current_start_timestamp,  # Last known timestamp
                'chunk_id': chunk_id,
                'metadata': metadata
            })

        return chunks if chunks else [{'text': transcript_text, 'start_time': '00:00:00', 'end_time': '00:00:00', 'chunk_id': 0, 'metadata': metadata}]

    def chunk_by_characters(self, transcript_text: str, metadata: Dict) -> List[Dict]:
        """Fallback chunking for episodes without timestamps."""
        chunks = []

        for i in range(0, len(transcript_text), self.chunk_size - self.chunk_overlap):
            chunk_text = transcript_text[i:i + self.chunk_size]

            chunks.append({
                'text': chunk_text,
                'start_time': '00:00:00',  # No timestamp available
                'end_time': '00:00:00',
                'chunk_id': len(chunks),
                'metadata': metadata
            })

        return chunks if chunks else [{'text': transcript_text, 'start_time': '00:00:00', 'end_time': '00:00:00', 'chunk_id': 0, 'metadata': metadata}]
