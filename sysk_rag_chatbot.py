"""
SYSK Podcast RAG Chatbot with Admin Interface
Retrieval-Augmented Generation chatbot for Stuff You Should Know podcast transcripts
Built following the Water Portal Assistant architecture pattern

NEW: PIN-protected admin interface for batch database indexing
Access via: ?admin=true URL parameter
"""

import streamlit as st
import os
import re
from datetime import datetime
from pathlib import Path
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import hashlib
from typing import List, Dict, Optional, Tuple
import json

# NEW: Hybrid search module
from hybrid_search import HybridSearcher

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_COLLECTION = "sysk_transcripts"
    CLAUDE_MODEL = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS = 4096
    TOP_K_RESULTS = 5
    CONVERSATION_HISTORY_LENGTH = 5
    
    # Chunking parameters (configurable via UI)
    # For time-based chunking (episodes with timestamps)
    CHUNK_DURATION_SECONDS = 180  # 3 minutes per chunk
    
    # For character-based chunking (episodes without timestamps) 
    CHUNK_SIZE_CHARS = 2000  # Characters per chunk
    CHUNK_OVERLAP_CHARS = 200  # Character overlap between chunks
    
    # Hybrid search defaults
    SEARCH_MODE = "Smart"  # Smart, Hybrid, Semantic, Keyword
    SEMANTIC_WEIGHT = 0.5
    KEYWORD_WEIGHT = 0.5
    TITLE_WEIGHT = 2.0  # Multiplier for title matches
    
    # NEW: Admin configuration
    ADMIN_PIN_HASH = "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4"  # Default: "1234"
    INDEXING_PROGRESS_FILE = "indexing_progress.json"

# ============================================================================
# Admin Authentication
# ============================================================================

def hash_pin(pin: str) -> str:
    """Hash PIN for secure storage"""
    return hashlib.sha256(pin.encode()).hexdigest()

def check_admin_access() -> bool:
    """Check if user should see admin interface"""
    query_params = st.query_params
    return query_params.get("admin") == "true"

def load_admin_pin_hash() -> str:
    """Load admin PIN hash from config file or use default"""
    config_file = "admin_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("admin_pin_hash", Config.ADMIN_PIN_HASH)
        except:
            pass
    return Config.ADMIN_PIN_HASH

def save_admin_pin_hash(pin_hash: str):
    """Save new admin PIN hash to config file"""
    config_file = "admin_config.json"
    config = {"admin_pin_hash": pin_hash, "updated": datetime.now().isoformat()}
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def verify_and_change_pin(old_pin: str, new_pin: str, confirm_pin: str) -> tuple:
    """
    Verify old PIN and change to new PIN
    Returns: (success: bool, message: str)
    """
    current_hash = load_admin_pin_hash()
    
    # Verify old PIN
    if hash_pin(old_pin) != current_hash:
        return False, "❌ Current PIN is incorrect"
    
    # Validate new PIN
    if not new_pin or len(new_pin) < 4:
        return False, "❌ New PIN must be at least 4 characters"
    
    if new_pin != confirm_pin:
        return False, "❌ New PIN entries don't match"
    
    if new_pin == old_pin:
        return False, "❌ New PIN must be different from old PIN"
    
    # Save new PIN
    new_hash = hash_pin(new_pin)
    save_admin_pin_hash(new_hash)
    
    return True, "✅ PIN changed successfully!"

def sync_progress_from_database(vector_db: 'VectorDatabase', transcripts_folder: str):
    """
    Rebuild indexing_progress.json from existing ChromaDB data
    Useful when database exists but progress file is missing/empty
    """
    # Get all unique filenames from ChromaDB
    all_metadata = vector_db.collection.get()['metadatas']
    
    if not all_metadata:
        return {
            "indexed_files": [],
            "total_indexed": 0,
            "last_updated": None
        }
    
    # Extract unique filenames
    indexed_files = set()
    for metadata in all_metadata:
        filename = metadata.get('filename', '')
        if filename:
            indexed_files.add(filename)
    
    # Create progress structure
    progress = {
        "indexed_files": sorted(list(indexed_files)),
        "total_indexed": len(indexed_files),
        "last_updated": datetime.now().isoformat(),
        "synced_from_db": True
    }
    
    return progress

# ============================================================================
# Admin Database Builder Functions
# ============================================================================

def load_indexing_progress() -> Dict:
    """Load which files have been indexed"""
    if os.path.exists(Config.INDEXING_PROGRESS_FILE):
        with open(Config.INDEXING_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"indexed_files": [], "last_updated": None, "total_indexed": 0}

def save_indexing_progress(progress: Dict):
    """Save indexing progress"""
    progress["last_updated"] = datetime.now().isoformat()
    with open(Config.INDEXING_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_transcript_files(transcripts_folder: str) -> List[Path]:
    """Get all transcript files"""
    transcript_dir = Path(transcripts_folder)
    if not transcript_dir.exists():
        return []
    return sorted([f for f in transcript_dir.glob("*.txt")])

def batch_index_transcripts(processor: 'TranscriptProcessor', 
                            vector_db: 'VectorDatabase',
                            transcripts_folder: str,
                            batch_size: int = 50,
                            progress_callback=None) -> Dict:
    """
    Index transcripts in batches with progress tracking
    
    Args:
        processor: TranscriptProcessor instance
        vector_db: VectorDatabase instance
        transcripts_folder: Path to transcripts directory
        batch_size: Number of files to process per batch
        progress_callback: Optional callback for progress updates (current, total)
        
    Returns:
        Dict with indexing results
    """
    progress = load_indexing_progress()
    all_files = get_transcript_files(transcripts_folder)
    
    # Filter out already indexed files
    indexed_set = set(progress.get("indexed_files", []))
    files_to_index = [f for f in all_files if f.name not in indexed_set]
    
    if not files_to_index:
        return {
            "status": "complete",
            "message": "All files already indexed",
            "total_files": len(all_files),
            "indexed": len(indexed_set)
        }
    
    # Get batch
    batch_files = files_to_index[:batch_size]
    
    # Index the batch
    results = {
        "status": "processing",
        "processed": 0,
        "errors": []
    }
    
    total_in_batch = len(batch_files)
    
    for idx, file_path in enumerate(batch_files, 1):
        # Call progress callback if provided
        if progress_callback:
            progress_callback(idx, total_in_batch)
        
        try:
            # Parse transcript file
            parsed = processor.parse_transcript_file(str(file_path))
            if not parsed:
                results["errors"].append({
                    "file": file_path.name,
                    "error": "Failed to parse file"
                })
                continue
            
            # Create chunks
            chunks = processor.chunk_by_time(
                parsed['transcript'], 
                parsed['metadata']
            )
            
            # Add to ChromaDB
            if chunks:
                for chunk in chunks:
                    chunk_id = f"{parsed['metadata']['filename']}_{chunk['chunk_id']}"
                    
                    # Prepare metadata for storage
                    chunk_metadata = {
                        'title': parsed['metadata'].get('title', 'Unknown'),
                        'date': parsed['metadata'].get('date', 'Unknown'),
                        'duration': parsed['metadata'].get('duration', 'Unknown'),
                        'episode_type': parsed['metadata'].get('episode_type', 'Full Episode'),
                        'filename': parsed['metadata'].get('filename', ''),
                        'time': f"{chunk.get('start_time', '00:00:00')} - {chunk.get('end_time', '00:00:00')}",
                        'chunk_id': chunk['chunk_id']
                    }
                    
                    # Add URLs if available
                    for url_key in ['episode_url', 'audio_url', 'transcript_url']:
                        if url_key in parsed['metadata']:
                            chunk_metadata[url_key] = parsed['metadata'][url_key]
                    
                    vector_db.collection.add(
                        documents=[chunk['text']],
                        metadatas=[chunk_metadata],
                        ids=[chunk_id]
                    )
                
                # Track progress
                progress["indexed_files"].append(file_path.name)
                progress["total_indexed"] = len(progress["indexed_files"])
                results["processed"] += 1
                
        except Exception as e:
            results["errors"].append({
                "file": file_path.name,
                "error": str(e)
            })
    
    # Save progress
    save_indexing_progress(progress)
    
    results["total_files"] = len(all_files)
    results["total_indexed"] = len(progress["indexed_files"])
    results["remaining"] = len(all_files) - len(progress["indexed_files"])
    
    return results

# ============================================================================
# Document Processing (Your existing class)
# ============================================================================

class TranscriptProcessor:
    """Process SYSK transcript files into chunks"""
    
    def __init__(self, chunk_duration_seconds=None, chunk_size_chars=None, chunk_overlap_chars=None):
        self.chunk_duration = chunk_duration_seconds or Config.CHUNK_DURATION_SECONDS
        self.chunk_size = chunk_size_chars or Config.CHUNK_SIZE_CHARS
        self.chunk_overlap = chunk_overlap_chars or Config.CHUNK_OVERLAP_CHARS
    
    def parse_transcript_file(self, filepath: str) -> Optional[Dict]:
        """Parse a transcript file and extract metadata"""
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
            st.warning(f"Error parsing {filepath}: {e}")
            return None
    
    def parse_timestamp(self, timestamp_str: str) -> int:
        """Convert timestamp string (HH:MM:SS) to seconds"""
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
        except:
            return 0
    
    def chunk_by_time(self, transcript_text: str, metadata: Dict) -> List[Dict]:
        """Chunk transcript by time intervals, with fallback for episodes without timestamps"""
        
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
        """Fallback chunking for episodes without timestamps"""
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

# ============================================================================
# Vector Database (Your existing class - abbreviated)
# ============================================================================

class VectorDatabase:
    """ChromaDB vector database for transcript storage and retrieval"""
    
    def __init__(self, collection_name: str = Config.CHROMA_COLLECTION, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            all_metadata = self.collection.get()['metadatas']
            
            full_episodes = sum(1 for m in all_metadata if m.get('episode_type') == 'Full Episode')
            short_stuff = sum(1 for m in all_metadata if m.get('episode_type') == 'Short Stuff')
            unique_episodes = len(set(m.get('filename', '') for m in all_metadata))
            
            return {
                'total_chunks': len(all_metadata),
                'total_episodes': unique_episodes,
                'full_episodes': full_episodes,
                'short_stuff': short_stuff
            }
        except:
            return {
                'total_chunks': 0,
                'total_episodes': 0,
                'full_episodes': 0,
                'short_stuff': 0
            }

# ============================================================================
# RAG System (Your existing class - abbreviated for space)
# ============================================================================

class RAGSystem:
    """Complete RAG system combining retrieval and generation"""
    
    def __init__(self, vector_db: VectorDatabase, api_key: str):
        self.vector_db = vector_db
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        
        # NEW: Initialize hybrid searcher
        self.hybrid_searcher = HybridSearcher(vector_db.collection)
    
    def retrieve_context(self, query: str, episode_type_filter: Optional[str] = None,
                        search_mode: str = "Smart", semantic_weight: float = 0.5,
                        keyword_weight: float = 0.5, title_weight: float = 2.0,
                        top_k: int = Config.TOP_K_RESULTS) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context with hybrid search"""
        
        # Execute search based on mode
        if search_mode == "Smart":
            results, method_used = self.hybrid_searcher.search_with_fallback(
                query, n_results=top_k, title_weight=title_weight
            )
        elif search_mode == "Hybrid":
            results = self.hybrid_searcher.hybrid_search(
                query, n_results=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                title_weight=title_weight
            )
        elif search_mode == "Semantic":
            results = self.hybrid_searcher.semantic_search(query, n_results=top_k)
        elif search_mode == "Keyword":
            results = self.hybrid_searcher.keyword_search(query, n_results=top_k, title_weight=title_weight)
        else:
            results = self.hybrid_searcher.semantic_search(query, n_results=top_k)
        
        # Apply episode type filter if specified
        if episode_type_filter and results['ids'] and results['ids'][0]:
            filtered_ids = []
            filtered_docs = []
            filtered_meta = []
            
            for i, metadata in enumerate(results['metadatas'][0]):
                if metadata.get('episode_type') == episode_type_filter:
                    filtered_ids.append(results['ids'][0][i])
                    filtered_docs.append(results['documents'][0][i])
                    filtered_meta.append(metadata)
            
            results = {
                'ids': [filtered_ids],
                'documents': [filtered_docs],
                'metadatas': [filtered_meta]
            }
        
        # Format context and sources
        if not results['ids'] or not results['ids'][0]:
            return "No relevant episodes found.", []
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            context_parts.append(f"[Episode {i}] {metadata.get('title', 'Unknown')}")
            context_parts.append(f"Date: {metadata.get('date', 'Unknown')}")
            context_parts.append(f"Time: {metadata.get('time', '00:00:00')}")
            context_parts.append(f"Content: {doc}\n")
            
            sources.append({
                'title': metadata.get('title', 'Unknown'),
                'date': metadata.get('date', 'Unknown'),
                'time': metadata.get('time', '00:00:00'),
                'episode_type': metadata.get('episode_type', 'Full Episode'),
                'audio_url': metadata.get('audio_url'),
                'episode_url': metadata.get('episode_url'),
                'transcript_url': metadata.get('transcript_url'),
                'match_type': results.get('match_types', [None])[i-1] if 'match_types' in results else None
            })
        
        return "\n".join(context_parts), sources
    
    def generate_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using Claude"""
        
        # Build conversation history for context
        history_text = ""
        if conversation_history:
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"\nUser: {exchange['query']}\nAssistant: {exchange['response']}\n"
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions about Stuff You Should Know podcast episodes.
        
Your role:
- Answer questions using the provided episode transcripts
- Be conversational and friendly
- Cite specific episodes when referencing information
- If the context doesn't contain relevant information, say so
- Reference timestamps when they're helpful

Format:
- Use natural language, not bullet points unless listing multiple items
- Keep responses concise but informative
- Mention episode titles naturally in your response"""
        
        user_prompt = f"""Based on these SYSK episode transcripts, please answer the question.

Previous conversation:
{history_text}

Episode Context:
{context}

Question: {query}

Answer:"""
        
        try:
            message = self.anthropic_client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=Config.CLAUDE_MAX_TOKENS,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============================================================================
# Admin Interface
# ============================================================================

def show_admin_interface():
    """PIN-protected admin interface for database management"""
    
    st.set_page_config(
        page_title="SYSK Admin",
        page_icon="🔧",
        layout="wide"
    )
    
    # Check authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        # PIN entry screen
        st.title("🔒 Admin Access")
        st.write("Enter your PIN to access database management")
        
        pin = st.text_input("PIN", type="password", max_chars=20, key="login_pin")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Submit", type="primary"):
                if hash_pin(pin) == load_admin_pin_hash():
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect PIN")
        
        st.stop()
    
    # ========================================================================
    # AUTHENTICATED ADMIN INTERFACE
    # ========================================================================
    
    # Initialize systems if not already
    if 'admin_vector_db' not in st.session_state:
        st.session_state.admin_vector_db = VectorDatabase()
    
    if 'admin_processor' not in st.session_state:
        st.session_state.admin_processor = TranscriptProcessor()
    
    if 'admin_rag_system' not in st.session_state:
        if not Config.ANTHROPIC_API_KEY:
            st.error("⚠️ ANTHROPIC_API_KEY not set")
            st.stop()
        st.session_state.admin_rag_system = RAGSystem(
            st.session_state.admin_vector_db, 
            Config.ANTHROPIC_API_KEY
        )
    
    vector_db = st.session_state.admin_vector_db
    processor = st.session_state.admin_processor
    rag_system = st.session_state.admin_rag_system
    
    # Initialize chat state for admin
    if 'admin_messages' not in st.session_state:
        st.session_state.admin_messages = []
    if 'admin_conversation_history' not in st.session_state:
        st.session_state.admin_conversation_history = []
    
    # ========================================================================
    # SIDEBAR - ALL ADMIN TOOLS
    # ========================================================================
    
    st.sidebar.title("🔧 Admin Tools")
    
    # Logout button at top
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.sidebar.divider()
    
    # ---- Configuration ----
    st.sidebar.markdown("### ⚙️ Configuration")
    transcripts_folder = st.sidebar.text_input("Transcripts Folder", value="./transcripts")
    
    # ---- Database Status ----
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Database Status")
    
    progress = load_indexing_progress()
    all_files = get_transcript_files(transcripts_folder)
    remaining = len(all_files) - progress.get("total_indexed", 0)
    db_count = vector_db.collection.count()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Files", len(all_files))
        st.metric("Indexed", progress.get("total_indexed", 0))
    with col2:
        st.metric("Remaining", remaining)
        st.metric("DB Docs", db_count)
    
    # Progress bar
    if len(all_files) > 0:
        progress_pct = progress.get("total_indexed", 0) / len(all_files)
        st.sidebar.progress(progress_pct, text=f"{progress_pct*100:.1f}%")
    
    if progress.get("last_updated"):
        st.sidebar.caption(f"📅 {progress['last_updated']}")
    
    # ---- Batch Indexing ----
    st.sidebar.divider()
    st.sidebar.markdown("### ⚡ Batch Indexing")
    
    batch_size = st.sidebar.slider(
        "Batch Size", 
        min_value=10, 
        max_value=200, 
        value=50,
        help="Files to index per batch"
    )
    
    if remaining > 0:
        if st.sidebar.button("🚀 Index Next Batch", type="primary", use_container_width=True):
            # Create placeholders for progress
            progress_text = st.sidebar.empty()
            progress_bar = st.sidebar.empty()
            
            # Progress callback
            def update_progress(current, total):
                progress_text.text(f"Processing file {current} of {total}")
                progress_bar.progress(current / total)
            
            # Run indexing with progress updates
            results = batch_index_transcripts(
                processor, 
                vector_db,
                transcripts_folder,
                batch_size=batch_size,
                progress_callback=update_progress
            )
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
            
            if results["status"] == "complete":
                st.sidebar.success("✅ " + results["message"])
            else:
                st.sidebar.success(f"✅ Indexed {results['processed']} files")
                st.sidebar.info(f"📝 {results['remaining']} remaining")
                
                if results.get("errors"):
                    with st.sidebar.expander("⚠️ Errors"):
                        for error in results["errors"]:
                            st.error(f"{error['file']}: {error['error']}")
            
            # Auto-sync progress from DB
            progress = sync_progress_from_database(vector_db, transcripts_folder)
            save_indexing_progress(progress)
            
            st.rerun()
    else:
        st.sidebar.success("✅ All files indexed!")
    
    # ---- Database Management ----
    st.sidebar.divider()
    st.sidebar.markdown("### 🔧 Database Management")
    
    # Sync button - useful when DB exists but progress is missing
    if st.sidebar.button("🔄 Sync Progress from DB", use_container_width=True, 
                         help="Rebuild progress tracker from existing database"):
        with st.spinner("Syncing progress from database..."):
            progress = sync_progress_from_database(vector_db, transcripts_folder)
            save_indexing_progress(progress)
            st.sidebar.success(f"✅ Synced {progress['total_indexed']} files")
            st.rerun()
    
    if st.sidebar.button("🔄 Reset Progress", use_container_width=True):
        if os.path.exists(Config.INDEXING_PROGRESS_FILE):
            os.remove(Config.INDEXING_PROGRESS_FILE)
        st.sidebar.success("✅ Progress reset")
        
        # Auto-sync from DB if DB has content
        if vector_db.collection.count() > 0:
            progress = sync_progress_from_database(vector_db, transcripts_folder)
            save_indexing_progress(progress)
        
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Database", use_container_width=True):
        all_ids = vector_db.collection.get()['ids']
        if all_ids:
            total_docs = len(all_ids)
            batch_size = 5000  # ChromaDB batch limit
            
            # Create progress indicators
            progress_text = st.sidebar.empty()
            progress_bar = st.sidebar.empty()
            
            # Delete in batches with progress
            for i in range(0, total_docs, batch_size):
                batch_ids = all_ids[i:i + batch_size]
                vector_db.collection.delete(ids=batch_ids)
                
                current = min(i + batch_size, total_docs)
                progress_text.text(f"Deleting: {current}/{total_docs}")
                progress_bar.progress(current / total_docs)
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
            
            st.sidebar.success(f"✅ Deleted {total_docs} docs")
            
            # Auto-sync (should now show 0 files)
            progress = sync_progress_from_database(vector_db, transcripts_folder)
            save_indexing_progress(progress)
            
            st.rerun()
        else:
            st.sidebar.info("Database empty")
    
    if st.sidebar.button("♻️ Full Rebuild", use_container_width=True):
        # Clear database in batches with progress
        all_ids = vector_db.collection.get()['ids']
        if all_ids:
            total_docs = len(all_ids)
            batch_size = 5000
            
            # Create progress indicators
            progress_text = st.sidebar.empty()
            progress_bar = st.sidebar.empty()
            
            for i in range(0, total_docs, batch_size):
                batch_ids = all_ids[i:i + batch_size]
                vector_db.collection.delete(ids=batch_ids)
                
                current = min(i + batch_size, total_docs)
                progress_text.text(f"Clearing: {current}/{total_docs}")
                progress_bar.progress(current / total_docs)
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
        
        # Reset progress
        if os.path.exists(Config.INDEXING_PROGRESS_FILE):
            os.remove(Config.INDEXING_PROGRESS_FILE)
        
        # Auto-sync (should show 0 everything)
        progress = sync_progress_from_database(vector_db, transcripts_folder)
        save_indexing_progress(progress)
        
        st.sidebar.success("✅ Ready for fresh indexing")
        st.rerun()
    
    # ---- Change PIN ----
    st.sidebar.divider()
    st.sidebar.markdown("### 🔐 Change Admin PIN")
    
    with st.sidebar.expander("Change PIN"):
        old_pin = st.text_input("Current PIN", type="password", key="old_pin")
        new_pin = st.text_input("New PIN", type="password", key="new_pin")
        confirm_pin = st.text_input("Confirm New PIN", type="password", key="confirm_pin")
        
        if st.button("Update PIN", use_container_width=True):
            success, message = verify_and_change_pin(old_pin, new_pin, confirm_pin)
            if success:
                st.success(message)
                st.info("Please re-login with your new PIN")
                # Log out after successful change
                st.session_state.admin_authenticated = False
                st.rerun()
            else:
                st.error(message)
    
    # ---- Indexed Files List ----
    st.sidebar.divider()
    if progress.get("indexed_files"):
        with st.sidebar.expander(f"📝 Indexed Files ({len(progress['indexed_files'])})"):
            files_list = sorted(progress["indexed_files"])
            
            # Show first 50
            display_count = min(50, len(files_list))
            for filename in files_list[:display_count]:
                st.caption(filename)
            
            if len(files_list) > display_count:
                st.caption(f"...and {len(files_list) - display_count} more")
            
            # Download full list
            st.download_button(
                "📥 Download List",
                data="\n".join(files_list),
                file_name="indexed_files.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # ---- Search Settings ----
    st.sidebar.divider()
    st.sidebar.markdown("### 🔍 Search Settings")
    search_mode = st.sidebar.selectbox(
        "Search Mode",
        ["Smart", "Hybrid", "Semantic", "Keyword"],
        help="Search method for queries"
    )
    
    top_k = st.sidebar.slider("Number of Results", 3, 20, 10)
    
    with st.sidebar.expander("⚙️ Advanced Settings"):
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.1)
        keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.5, 0.1)
        title_weight = st.slider("Title Match Boost", 0.0, 5.0, 2.0, 0.5)
    
    episode_type_filter = st.sidebar.selectbox(
        "Episode Type",
        options=["All", "Full Episode", "Short Stuff"]
    )
    
    # ========================================================================
    # MAIN AREA - SEARCH INTERFACE (same as public)
    # ========================================================================
    
    st.title("🎙️ SYSK Search - Admin Mode")
    st.caption(f"🔧 Admin • {db_count} chunks indexed")
    
    # Check if indexed
    if db_count == 0:
        st.info("👈 Use sidebar to index transcripts")
        st.stop()
    
    # Display conversation
    for message in st.session_state.admin_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                        st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**")
                        st.caption(f"{source['date']} • {source['time']}")
                        
                        if source.get('audio_url'):
                            st.markdown(f"🎧 [Listen]({source['audio_url']})")
                        if source.get('episode_url'):
                            st.markdown(f"🔗 [Episode]({source['episode_url']})")
                        if source.get('transcript_url'):
                            st.markdown(f"📝 [Transcript]({source['transcript_url']})")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about SYSK episodes..."):
        st.session_state.admin_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Searching..."):
                context, sources = rag_system.retrieve_context(
                    prompt,
                    episode_type_filter=episode_type_filter if episode_type_filter != "All" else None,
                    search_mode=search_mode,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    title_weight=title_weight,
                    top_k=top_k
                )
                
                response = rag_system.generate_response(
                    prompt, 
                    context, 
                    st.session_state.admin_conversation_history[-Config.CONVERSATION_HISTORY_LENGTH:]
                )
                
                st.markdown(response)
                
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)})"):
                        for i, source in enumerate(sources, 1):
                            episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                            st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**")
                            st.caption(f"{source['date']} • {source['time']}")
                            
                            if source.get('audio_url'):
                                st.markdown(f"🎧 [Listen]({source['audio_url']})")
                            if source.get('episode_url'):
                                st.markdown(f"🔗 [Episode]({source['episode_url']})")
                            if source.get('transcript_url'):
                                st.markdown(f"📝 [Transcript]({source['transcript_url']})")
                            st.markdown("---")
                
                st.session_state.admin_conversation_history.append({
                    'query': prompt,
                    'response': response
                })
                
                st.session_state.admin_messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
    
    # Clear conversation
    if len(st.session_state.admin_messages) > 0:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.admin_messages = []
                st.session_state.admin_conversation_history = []
                st.rerun()

# ============================================================================
# Main Search Interface (Your existing interface)
# ============================================================================

def show_search_interface():
    """Main public search interface - clean chat with NO sidebar"""
    
    st.set_page_config(
        page_title="SYSK Search",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with sidebar collapsed
    )
    
    # Hide sidebar completely with CSS
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    
    # Initialize systems
    if 'rag_system' not in st.session_state:
        vector_db = VectorDatabase()
        
        if not Config.ANTHROPIC_API_KEY:
            st.error("⚠️ ANTHROPIC_API_KEY not set. Please set it in environment variables or .streamlit/secrets.toml")
            st.stop()
        
        st.session_state.rag_system = RAGSystem(vector_db, Config.ANTHROPIC_API_KEY)
    
    # Default search parameters (no sidebar controls for public users)
    search_mode = "Smart"
    top_k = 10
    semantic_weight = 0.5
    keyword_weight = 0.5
    title_weight = 2.0
    episode_type_filter = "All"
    
    # Main chat interface
    st.title("🎙️ Stuff You Should Know - Podcast Assistant")
    
    st.caption(f"🧠 Smart Search Mode • 🎙️ {st.session_state.rag_system.vector_db.collection.count()} chunks indexed")
    
    st.markdown("Ask me anything about SYSK episodes!")
    
    # Check if indexed
    if st.session_state.rag_system.vector_db.collection.count() == 0:
        st.info("👆 No episodes indexed yet. Contact admin to index transcripts.")
        st.stop()
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                        
                        match_badge = ""
                        if 'match_type' in source:
                            match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄'}
                            match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                        
                        st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                        st.caption(f"{source['date']} • {source['time']}")
                        
                        if source.get('audio_url'):
                            st.markdown(f"🎧 [Listen to Episode]({source['audio_url']})")
                        if source.get('episode_url'):
                            st.markdown(f"🔗 [Episode Page]({source['episode_url']})")
                        if source.get('transcript_url'):
                            st.markdown(f"📝 [Transcript]({source['transcript_url']})")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about SYSK episodes..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Searching with {search_mode} mode..."):
                # Retrieve context
                context, sources = st.session_state.rag_system.retrieve_context(
                    prompt,
                    episode_type_filter=episode_type_filter if episode_type_filter != "All" else None,
                    search_mode=search_mode,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    title_weight=title_weight,
                    top_k=top_k
                )
                
                # Generate response
                response = st.session_state.rag_system.generate_response(
                    prompt, 
                    context, 
                    st.session_state.conversation_history[-Config.CONVERSATION_HISTORY_LENGTH:]
                )
                
                st.markdown(response)
                
                # Display sources
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} found)"):
                        for i, source in enumerate(sources, 1):
                            episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                            
                            match_badge = ""
                            if 'match_type' in source:
                                match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄'}
                                match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                            
                            st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                            st.caption(f"{source['date']} • {source['time']}")
                            
                            if source.get('audio_url'):
                                st.markdown(f"🎧 [Listen to Episode]({source['audio_url']})")
                            if source.get('episode_url'):
                                st.markdown(f"🔗 [Episode Page]({source['episode_url']})")
                            if source.get('transcript_url'):
                                st.markdown(f"📝 [Transcript]({source['transcript_url']})")
                            st.markdown("---")
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    'query': prompt,
                    'response': response
                })
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
    
    # Clear conversation button in main area (no sidebar for public users)
    if len(st.session_state.messages) > 0:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

# ============================================================================
# Main App Routing
# ============================================================================

def main():
    """Route to admin or search interface based on URL parameter"""
    if check_admin_access():
        show_admin_interface()
    else:
        show_search_interface()

if __name__ == "__main__":
    main()
