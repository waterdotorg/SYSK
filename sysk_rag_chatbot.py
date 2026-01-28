"""
SYSK Podcast RAG Chatbot
Retrieval-Augmented Generation chatbot for Stuff You Should Know podcast transcripts
Built following the Water Portal Assistant architecture pattern
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
import requests
import zipfile

# NEW: Hybrid search module
from hybrid_search import HybridSearcher

# ============================================================================
# Database Setup (for cloud deployment with Git LFS)
# ============================================================================

@st.cache_resource(show_spinner=False)
def setup_database_if_needed(_version="v2"):  # Change version to force re-extraction
    """
    Extract ChromaDB from chroma_db.zip if not present locally.
    Only runs once per app session (cached).
    The zip file is stored in the repo using Git LFS.
    
    Args:
        _version: Internal version to force cache invalidation when database updates
    """
    db_path = "./chroma_db"
    zip_path = "./chroma_db.zip"
    
    # Delete old database if exists (to force re-extraction of updated zip)
    if os.path.exists(db_path):
        import shutil
        st.info("🔄 Removing old database to extract updated version...")
        shutil.rmtree(db_path)
    
    # Skip the exists check - always extract when this function runs
    
    # Debug: Check what files exist
    st.info(f"🔍 Looking for database zip at: {os.path.abspath(zip_path)}")
    
    # Check if zip file exists (from Git LFS)
    if not os.path.exists(zip_path):
        # List current directory contents for debugging
        files = os.listdir(".")
        st.error(f"❌ Database zip file not found at {zip_path}")
        st.info(f"📁 Files in current directory: {files[:20]}")  # Show first 20 files
        st.info("💡 Ensure chroma_db.zip is in the repository with Git LFS")
        return False
    
    # Check file size to ensure LFS actually downloaded it (not just a pointer)
    file_size = os.path.getsize(zip_path)
    if file_size < 1000:  # LFS pointer files are tiny (~100 bytes)
        st.error(f"❌ Database file appears to be an LFS pointer ({file_size} bytes)")
        st.info("💡 Streamlit Cloud may not have Git LFS enabled. The file needs to be ~900MB.")
        return False
    
    st.info(f"✓ Found zip file ({file_size / (1024**3):.2f} GB)")
    
    try:
        with st.spinner("📥 Extracting database (first load only, ~1-2 minutes)..."):
            # Extract database
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            
            st.success("✓ Database extracted successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to extract database: {e}")
        return False

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    
    @staticmethod
    def get_api_key():
        """Get API key from Streamlit secrets (cloud) or environment (local)"""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            return st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError):
            # Fallback to environment variable (for local development)
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                st.error("⚠️ API key not found. Please configure ANTHROPIC_API_KEY in secrets or environment.")
                st.stop()
            return api_key
    
    @staticmethod
    def get_database_url():
        """Get database download URL from secrets (optional, for cloud with external storage)"""
        try:
            return st.secrets.get("DATABASE_URL", None)
        except (KeyError, FileNotFoundError, AttributeError):
            return None
    
    ANTHROPIC_API_KEY = None  # Will be set dynamically
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_COLLECTION = "sysk_transcripts"
    CLAUDE_MODEL = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS = 4096
    TOP_K_RESULTS = 3  # Reduced from 5 for better performance on Streamlit Cloud
    CONVERSATION_HISTORY_LENGTH = 3  # Reduced from 5 for memory optimization
    
    # Chunking parameters (configurable via UI)
    # For time-based chunking (episodes with timestamps)
    CHUNK_DURATION_SECONDS = 180  # 3 minutes per chunk
    
    # For character-based chunking (episodes without timestamps) 
    CHUNK_SIZE_CHARS = 2000  # Characters per chunk
    CHUNK_OVERLAP_CHARS = 200  # Character overlap between chunks
    
    # NEW: Hybrid search defaults
    SEARCH_MODE = "Smart"  # Smart, Hybrid, Semantic, Keyword
    SEMANTIC_WEIGHT = 0.5
    KEYWORD_WEIGHT = 0.5
    TITLE_WEIGHT = 2.0  # Multiplier for title matches (1.0 = equal to content, 5.0 = heavy boost)

# ============================================================================
# Document Processing
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
        # Insert line breaks before and after timestamps (HH:MM:SS pattern)
        transcript_text = re.sub(r'(\d{2}:\d{2}:\d{2})([A-Za-z])', r'\n\1\n\2', transcript_text)
        
        # Count timestamps to determine chunking strategy
        timestamp_count = len(re.findall(r'\d{2}:\d{2}:\d{2}', transcript_text))
        
        # If episode has very few timestamps (older episodes), use character-based chunking
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
                'end_time': "END",
                'chunk_id': chunk_id,
                'metadata': metadata
            })
        
        return chunks
    
    def chunk_by_characters(self, transcript_text: str, metadata: Dict) -> List[Dict]:
        """Fallback chunking for episodes without timestamps - use fixed character count"""
        chunks = []
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        
        # Split into sentences to avoid cutting mid-sentence
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_time': f"Chunk {chunk_id + 1}",
                    'end_time': f"Chunk {chunk_id + 1}",
                    'chunk_id': chunk_id,
                    'metadata': metadata
                })
                
                # Start new chunk with overlap (keep last few sentences)
                overlap_text = ' '.join(current_chunk[-3:]) if len(current_chunk) > 3 else ''
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_time': f"Chunk {chunk_id + 1}",
                'end_time': f"Chunk {chunk_id + 1}",
                'chunk_id': chunk_id,
                'metadata': metadata
            })
        
        return chunks

# ============================================================================
# Vector Database Management
# ============================================================================

class VectorDatabase:
    """Manage ChromaDB vector database"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        # Get existing collection without specifying embedding function
        # (will use whatever was set when collection was created)
        try:
            self.collection = self.client.get_collection(
                name=Config.CHROMA_COLLECTION
            )
        except Exception:
            # If collection doesn't exist, create it with embedding function
            self.collection = self.client.create_collection(
                name=Config.CHROMA_COLLECTION,
                embedding_function=self.embedding_function
            )
    
    def get_document_hash(self, filepath: str) -> str:
        """Generate hash of document for change detection"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def add_chunks(self, chunks: List[Dict], force_reindex: bool = False):
        """Add chunks to vector database"""
        for chunk in chunks:
            doc_id = f"{chunk['metadata']['filename']}_chunk_{chunk['chunk_id']}"
            
            # Check if already indexed
            existing = self.collection.get(ids=[doc_id])
            if existing['ids']:
                if force_reindex:
                    # Delete existing entry before re-adding
                    self.collection.delete(ids=[doc_id])
                else:
                    continue
            
            # Add to database
            self.collection.add(
                documents=[chunk['text']],
                metadatas=[{
                    'title': chunk['metadata']['title'],
                    'date': chunk['metadata']['date'],
                    'duration': chunk['metadata']['duration'],
                    'episode_type': chunk['metadata']['episode_type'],
                    'filename': chunk['metadata']['filename'],
                    'episode_url': chunk['metadata'].get('episode_url', ''),
                    'audio_url': chunk['metadata'].get('audio_url', ''),
                    'transcript_url': chunk['metadata'].get('transcript_url', ''),
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'chunk_id': str(chunk['chunk_id'])
                }],
                ids=[doc_id]
            )
    
    def search(self, query: str, episode_type_filter: Optional[str] = None, n_results: int = Config.TOP_K_RESULTS) -> Dict:
        """Search for relevant chunks using semantic search"""
        where_filter = None
        if episode_type_filter and episode_type_filter != "All":
            where_filter = {"episode_type": episode_type_filter}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics - memory efficient version"""
        count = self.collection.count()
        
        # Estimate episodes without loading all data (memory efficient)
        # Average ~19 chunks per episode based on your data
        estimated_episodes = count // 19
        
        return {
            'total_chunks': count,
            'total_episodes': estimated_episodes,
            'short_stuff': 0,  # Estimation mode
            'full_episodes': estimated_episodes
        }

# ============================================================================
# RAG System
# ============================================================================

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.get_api_key())
        self.vector_db = VectorDatabase()
        
        # NEW: Initialize hybrid searcher
        self.hybrid_searcher = HybridSearcher(self.vector_db.collection)
        self.search_mode = Config.SEARCH_MODE
        self.semantic_weight = Config.SEMANTIC_WEIGHT
        self.keyword_weight = Config.KEYWORD_WEIGHT
        self.title_weight = Config.TITLE_WEIGHT
    
    def retrieve_context(self, query: str, episode_type_filter: Optional[str] = None,
                        search_mode: Optional[str] = None, 
                        semantic_weight: Optional[float] = None,
                        keyword_weight: Optional[float] = None,
                        title_weight: Optional[float] = None,
                        top_k: int = Config.TOP_K_RESULTS) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context using hybrid search"""
        
        # Use provided parameters or fall back to instance defaults
        mode = search_mode or self.search_mode
        sem_weight = semantic_weight or self.semantic_weight
        key_weight = keyword_weight or self.keyword_weight
        t_weight = title_weight or self.title_weight
        
        # Execute search based on selected mode
        if mode == "Smart":
            results, method = self.hybrid_searcher.search_with_fallback(
                query, n_results=top_k, title_weight=t_weight
            )
            search_method_used = f"smart_{method}"
        elif mode == "Hybrid":
            results = self.hybrid_searcher.hybrid_search(
                query, n_results=top_k,
                semantic_weight=sem_weight,
                keyword_weight=key_weight,
                title_weight=t_weight
            )
            search_method_used = "hybrid"
        elif mode == "Keyword":
            results = self.hybrid_searcher.keyword_search(
                query, n_results=top_k, title_weight=t_weight
            )
            search_method_used = "keyword"
        else:  # "Semantic"
            results = self.hybrid_searcher.semantic_search(query, n_results=top_k)
            search_method_used = "semantic"
        
        # Apply episode type filter if needed (post-search filtering)
        if episode_type_filter and results['ids'] and results['ids'][0]:
            filtered_ids = []
            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                if meta.get('episode_type') == episode_type_filter:
                    filtered_ids.append(results['ids'][0][i])
                    filtered_docs.append(results['documents'][0][i])
                    filtered_metas.append(meta)
                    filtered_dists.append(results['distances'][0][i])
            
            results = {
                'ids': [filtered_ids],
                'documents': [filtered_docs],
                'metadatas': [filtered_metas],
                'distances': [filtered_dists]
            }
        
        context_parts = []
        sources = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context_parts.append(f"[Source {i+1}] Episode: {metadata['title']}")
                context_parts.append(f"Date: {metadata['date']} | Time: {metadata['start_time']}")
                if metadata.get('episode_url'):
                    context_parts.append(f"Episode Page: {metadata['episode_url']}")
                if metadata.get('audio_url'):
                    context_parts.append(f"Audio (Listen): {metadata['audio_url']}")
                if metadata.get('transcript_url'):
                    context_parts.append(f"Transcript: {metadata['transcript_url']}")
                context_parts.append(f"Content: {doc}")
                context_parts.append("---")
                
                source = {
                    'title': metadata['title'],
                    'date': metadata['date'],
                    'time': metadata['start_time'],
                    'episode_url': metadata.get('episode_url', ''),
                    'audio_url': metadata.get('audio_url', ''),
                    'transcript_url': metadata.get('transcript_url', ''),
                    'episode_type': metadata.get('episode_type', 'Full Episode'),
                    'search_method': search_method_used
                }
                
                # Add match information if available
                if 'sources' in results and i < len(results['sources']):
                    source['match_type'] = results['sources'][i]
                if 'match_types' in results and i < len(results['match_types']):
                    source['match_details'] = results['match_types'][i]
                
                sources.append(source)
        
        context = '\n'.join(context_parts)
        return context, sources
    
    def generate_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using Claude"""
        
        system_prompt = """You are a helpful assistant for the "Stuff You Should Know" (SYSK) podcast. 
You help users find information from podcast episodes by answering questions based on the transcript content provided.

When answering:
- Be conversational and engaging, matching the friendly tone of the podcast
- Cite specific episodes when referencing information
- If the context doesn't contain relevant information, say so honestly
- Include episode titles and dates when mentioning specific episodes
- If multiple episodes discuss a topic, mention that and highlight key differences

The context provided contains excerpts from SYSK podcast transcripts with episode titles, dates, and timestamps."""

        # Build conversation history for Claude
        messages = []
        for msg in conversation_history:
            messages.append({"role": "user", "content": msg['query']})
            messages.append({"role": "assistant", "content": msg['response']})
        
        # Add current query with context
        current_message = f"""Context from SYSK episodes:

{context}

User question: {query}

Please answer based on the context provided above."""
        
        messages.append({"role": "user", "content": current_message})
        
        # Call Claude API
        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=Config.CLAUDE_MAX_TOKENS,
            system=system_prompt,
            messages=messages
        )
        
        return response.content[0].text

# ============================================================================
# Streamlit Application
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False

def index_transcripts(transcripts_folder: str, force_reindex: bool = False, 
                     chunk_duration: int = None, chunk_size: int = None, chunk_overlap: int = None):
    """Index all transcript files"""
    processor = TranscriptProcessor(chunk_duration, chunk_size, chunk_overlap)
    vector_db = st.session_state.rag_system.vector_db
    
    transcript_files = list(Path(transcripts_folder).glob("*.txt"))
    
    if not transcript_files:
        st.error(f"No transcript files found in {transcripts_folder}")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_chunks = 0
    
    for idx, filepath in enumerate(transcript_files):
        status_text.text(f"Processing {idx+1}/{len(transcript_files)}: {filepath.name}")
        
        # Parse transcript
        parsed = processor.parse_transcript_file(str(filepath))
        if not parsed:
            continue
        
        # Chunk by time
        chunks = processor.chunk_by_time(parsed['transcript'], parsed['metadata'])
        
        # Add to vector database
        vector_db.add_chunks(chunks, force_reindex)
        total_chunks += len(chunks)
        
        progress_bar.progress((idx + 1) / len(transcript_files))
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✓ Indexed {len(transcript_files)} episodes ({total_chunks} chunks)")
    st.session_state.indexed = True

def display_sources(sources: List[Dict]):
    """Display source episodes in sidebar"""
    if sources:
        st.sidebar.markdown("### 📚 Sources Referenced")
        seen_episodes = set()
        for source in sources:
            episode_key = f"{source['title']}_{source['date']}"
            if episode_key not in seen_episodes:
                seen_episodes.add(episode_key)
                episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                st.sidebar.markdown(f"{episode_type_emoji} **{source['title']}**")
                st.sidebar.markdown(f"*{source['date']} • {source['time']}*")
                if source.get('audio_url'):
                    st.sidebar.markdown(f"🎧 [Listen]({source['audio_url']})")
                if source.get('episode_url'):
                    st.sidebar.markdown(f"🔗 [Page]({source['episode_url']})")
                st.sidebar.markdown("---")

def main():
    st.set_page_config(
        page_title="SYSK Podcast Assistant",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Extract database from Git LFS zip file if needed
    if not setup_database_if_needed():
        st.error("❌ Database not available. Please ensure chroma_db.zip is in the repository.")
        st.stop()
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("🎙️ SYSK Assistant")
    st.sidebar.markdown("Ask questions about **Stuff You Should Know** podcast episodes!")
    
    # API Key check (removed - now handled in Config.get_api_key())
    
    # NEW: Search Settings Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Search Settings")
    
    search_mode = st.sidebar.radio(
        "Search Mode:",
        ["Smart", "Hybrid", "Semantic", "Keyword"],
        index=0,
        help="""
**Smart**: Auto-fallback (tries semantic, then keyword if needed)
**Hybrid**: Combines both with adjustable weights
**Semantic**: AI-powered contextual search (original method)
**Keyword**: Direct text matching for exact terms
        """
    )
    
    # Show weights only for Hybrid mode
    semantic_weight = Config.SEMANTIC_WEIGHT
    keyword_weight = Config.KEYWORD_WEIGHT
    
    if search_mode == "Hybrid":
        st.sidebar.markdown("#### Search Weights")
        semantic_weight = st.sidebar.slider(
            "Semantic (AI Context)",
            0.0, 1.0, 0.5, 0.1,
            help="Higher = more AI contextual understanding"
        )
        keyword_weight = st.sidebar.slider(
            "Keyword (Exact Match)",
            0.0, 1.0, 0.5, 0.1,
            help="Higher = more exact text matching"
        )
    
    # Number of results
    top_k = st.sidebar.slider(
        "Results to Retrieve",
        min_value=1,
        max_value=10,  # Reduced from 15 for better performance
        value=Config.TOP_K_RESULTS,
        help="Number of transcript chunks to retrieve (lower = faster)"
    )
    
    # NEW: Title Weight Control
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        title_weight = st.slider(
            "Title Match Weight",
            min_value=1.0,
            max_value=10.0,
            value=Config.TITLE_WEIGHT,
            step=0.5,
            help="""
How much to boost title matches over content matches:
• 1.0 = Equal weight (title = content)
• 2.0 = Balanced (mix of title and content results) ✓ RECOMMENDED
• 5.0 = Heavy title boost (mostly title matches)
• 10.0 = Maximum (only title matches)

Lower values give more diverse results across episodes.
Higher values focus on episodes where the term is in the title.
            """
        )
        st.caption(f"Current: {title_weight:.1f}x boost for title matches")
    
    # Show search info
    with st.sidebar.expander("ℹ️ Search Mode Info", expanded=False):
        if search_mode == "Smart":
            st.write("**Smart Mode** automatically chooses the best search:")
            st.write("1. Tries semantic search first")
            st.write("2. Falls back to keyword if results are poor")
            st.write("3. Uses hybrid if both are mediocre")
            st.write("")
            st.write("✅ Best for most users!")
        elif search_mode == "Hybrid":
            st.write("**Hybrid Mode** combines both:")
            st.write(f"• Semantic: {semantic_weight:.0%}")
            st.write(f"• Keyword: {keyword_weight:.0%}")
            st.write("")
            st.write("Adjust weights to fine-tune results")
        elif search_mode == "Semantic":
            st.write("**Semantic Mode** (original):")
            st.write("• AI-powered contextual search")
            st.write("• Best for natural language questions")
            st.write("• May miss exact technical terms")
        else:  # Keyword
            st.write("**Keyword Mode**:")
            st.write("• Direct text matching")
            st.write("• Best for exact terms/phrases")
            st.write("• Fast and precise")
    
    # Indexing section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Document Management")
    
    transcripts_folder = st.sidebar.text_input(
        "Transcripts Folder",
        value="./transcripts",
        help="Path to folder containing transcript .txt files"
    )
    
    # Chunking parameters
    st.sidebar.markdown("#### Chunking Settings")
    
    chunk_duration = st.sidebar.number_input(
        "Time Chunk Duration (seconds)",
        min_value=60,
        max_value=600,
        value=Config.CHUNK_DURATION_SECONDS,
        step=30,
        help="Duration for time-based chunks (episodes with timestamps)"
    )
    
    chunk_size = st.sidebar.number_input(
        "Character Chunk Size",
        min_value=500,
        max_value=5000,
        value=Config.CHUNK_SIZE_CHARS,
        step=100,
        help="Characters per chunk (episodes without timestamps)"
    )
    
    chunk_overlap = st.sidebar.number_input(
        "Character Overlap",
        min_value=0,
        max_value=1000,
        value=Config.CHUNK_OVERLAP_CHARS,
        step=50,
        help="Overlap between character chunks"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📥 Index", use_container_width=True):
            with st.spinner("Indexing transcripts..."):
                index_transcripts(transcripts_folder, force_reindex=False,
                                chunk_duration=chunk_duration,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)
    
    with col2:
        if st.button("🔄 Re-index", use_container_width=True):
            with st.spinner("Re-indexing all transcripts..."):
                index_transcripts(transcripts_folder, force_reindex=True,
                                chunk_duration=chunk_duration,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)
    
    # Database management
    st.sidebar.markdown("#### Database Management")
    if st.sidebar.button("🗑️ Delete Database", use_container_width=True):
        if st.sidebar.checkbox("⚠️ Confirm deletion"):
            try:
                import shutil
                shutil.rmtree("./chroma_db")
                st.sidebar.success("✓ Database deleted! Restart the app to reinitialize.")
                st.session_state.indexed = False
            except Exception as e:
                st.sidebar.error(f"Error deleting database: {e}")
    
    # Database stats
    if st.session_state.indexed or st.session_state.rag_system.vector_db.collection.count() > 0:
        stats = st.session_state.rag_system.vector_db.get_stats()
        st.sidebar.success(f"✓ {stats['total_episodes']} episodes indexed")
        st.sidebar.caption(f"🎙️ {stats['full_episodes']} Full Episodes")
        st.sidebar.caption(f"⚡ {stats['short_stuff']} Short Stuff")
        st.sidebar.caption(f"📦 {stats['total_chunks']} total chunks")
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")
    episode_type_filter = st.sidebar.selectbox(
        "Episode Type",
        options=["All", "Full Episode", "Short Stuff"],
        help="Filter by episode type"
    )
    
    # Example prompts
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Example Questions")
    examples = [
        "What episodes discuss artificial intelligence?",
        "Tell me about the episode on data centers",
        "What did Josh and Chuck say about sleep?",
        "Find episodes about space exploration",
        "What are some Short Stuff episodes about technology?"
    ]
    for example in examples:
        if st.sidebar.button(example, use_container_width=True):
            st.session_state.query_input = example
    
    # Main chat interface
    st.title("🎙️ Stuff You Should Know - Podcast Assistant")
    
    # NEW: Show current search mode
    mode_emoji = {"Smart": "🧠", "Hybrid": "🔄", "Semantic": "🎯", "Keyword": "📝"}
    st.caption(f"{mode_emoji.get(search_mode, '🔍')} Search Mode: **{search_mode}**")
    
    st.markdown("Ask me anything about SYSK episodes!")
    
    # Check if indexed
    if st.session_state.rag_system.vector_db.collection.count() == 0:
        st.info("👆 Click **Index** in the sidebar to get started!")
        st.stop()
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                        
                        # Show match information if available
                        match_badge = ""
                        if 'match_type' in source:
                            match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄'}
                            match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                        
                        st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                        st.caption(f"{source['date']} • {source['time']}")
                        
                        if 'match_details' in source and source['match_details'] != 'semantic_only':
                            st.caption(f"Match: {source['match_details']}")
                        
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
                try:
                    # Retrieve context with search parameters
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
                    
                except MemoryError:
                    st.error("❌ Out of memory. Try reducing the number of results to 3 or fewer.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.info("💡 Try: Reduce results, use Keyword search mode, or simplify your query")
                    st.stop()
                
                st.markdown(response)
                
                # Display sources
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} found)"):
                        for i, source in enumerate(sources, 1):
                            episode_type_emoji = "⚡" if source['episode_type'] == "Short Stuff" else "🎙️"
                            
                            # NEW: Show match information
                            match_badge = ""
                            if 'match_type' in source:
                                match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄'}
                                match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                            
                            st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                            st.caption(f"{source['date']} • {source['time']}")
                            
                            # NEW: Show match details for hybrid/keyword searches
                            if 'match_details' in source and source['match_details'] != 'semantic_only':
                                st.caption(f"Match: {source['match_details']}")
                            
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
                
                # Display sources in sidebar
                display_sources(sources)
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()
