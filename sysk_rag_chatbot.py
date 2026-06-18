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
from sentence_transformers import SentenceTransformer
import hashlib
from typing import List, Dict, Optional, Tuple
import json
import random

# Pinecone for persistent vector storage
from pinecone_vector_db import VectorDatabase

# NEW: Hybrid search module
from hybrid_search import HybridSearcher

# Shared, Streamlit-free transcript parsing/chunking (also used by index_transcripts.py)
from transcript_processor import TranscriptProcessor

# Pinecone as the single source of truth for "what is indexed" (Priority #1)
import pinecone_state

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    PINECONE_INDEX_NAME = "sysk-transcripts"  # Changed from underscore to hyphen
    CLAUDE_MODEL = "claude-sonnet-4-6"
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
    """Load admin PIN hash from Streamlit secrets (cloud) or config file (local)"""
    # Try Streamlit secrets first (persistent across reboots in cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "ADMIN_PIN_HASH" in st.secrets:
            return st.secrets["ADMIN_PIN_HASH"]
    except:
        pass
    
    # Fall back to config file (for local development or if not in secrets)
    config_file = "admin_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("admin_pin_hash", Config.ADMIN_PIN_HASH)
        except:
            pass
    
    # Default PIN hash (1234)
    return Config.ADMIN_PIN_HASH

def save_admin_pin_hash(pin_hash: str):
    """Save new admin PIN hash to config file"""
    config_file = "admin_config.json"
    config = {"admin_pin_hash": pin_hash, "updated": datetime.now().isoformat()}
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Note: Cannot programmatically update Streamlit Cloud secrets
    # User must manually add ADMIN_PIN_HASH to secrets in Streamlit Cloud UI

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

# (removed def sync_progress_from_database — superseded by Pinecone-sourced state / offline indexer)

# ============================================================================
# Admin Database Builder Functions
# ============================================================================

# (removed def load_indexing_progress — superseded by Pinecone-sourced state / offline indexer)

# (removed def save_indexing_progress — superseded by Pinecone-sourced state / offline indexer)

def get_transcript_files(transcripts_folder: str) -> List[Path]:
    """Get all transcript files"""
    transcript_dir = Path(transcripts_folder)
    if not transcript_dir.exists():
        return []
    return sorted([f for f in transcript_dir.glob("*.txt")])

@st.cache_data(ttl=900, show_spinner="Reading indexed state from Pinecone…")
def get_indexing_state(_vector_db: 'VectorDatabase', transcripts_folder: str,
                       db_count: int) -> Dict:
    """Derive indexing state directly from Pinecone (the single source of truth).

    Replaces the old indexing_progress.json ledger (Priority #1). Returns the set
    of episodes indexed in Pinecone, the files on disk, and the newest-first diff
    of what still needs indexing — without maintaining any local ledger.

    Cached (st.cache_data) because enumerating all vector IDs from Pinecone takes
    20-30s and Streamlit re-runs the whole script on every interaction (e.g. each
    search). Cache keys:
      * ``_vector_db`` is prefixed with ``_`` so Streamlit does NOT try to hash it.
      * ``transcripts_folder`` and ``db_count`` ARE hashed, so the cache auto-
        invalidates whenever the chunk count changes (i.e. after (re)indexing),
        with a 15-minute TTL as a backstop. Use the sidebar "Refresh" button
        (get_indexing_state.clear()) to force an immediate recompute.
    """
    disk_files = {f.name for f in get_transcript_files(transcripts_folder)}
    try:
        indexed_files = pinecone_state.get_indexed_filenames(_vector_db.collection.index)
    except Exception as e:
        st.warning(f"Could not read indexed state from Pinecone: {e}")
        indexed_files = set()

    to_index = pinecone_state.diff_to_index(disk_files, indexed_files)
    return {
        "disk_files": disk_files,
        "indexed_files": indexed_files,
        "to_index": to_index,
        "total_disk": len(disk_files),
        "total_indexed": len(indexed_files),
        "remaining": len(to_index),
    }

# (removed def batch_index_transcripts — superseded by Pinecone-sourced state / offline indexer)

# ============================================================================
# Document Processing (Your existing class)
# ============================================================================

# (removed class TranscriptProcessor — superseded by Pinecone-sourced state / offline indexer)

# ============================================================================
# Vector Database - Now using Pinecone (see pinecone_vector_db.py)
# ============================================================================
# VectorDatabase class is imported from pinecone_vector_db.py

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
        # Note: For Pinecone, we avoid pure keyword search as it requires fetching all docs
        if search_mode == "Smart":
            # Use semantic search as primary (keyword fallback doesn't work well with Pinecone)
            results = self.hybrid_searcher.semantic_search(query, n_results=top_k)
            method_used = "semantic"
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
            # Keyword-only doesn't work with Pinecone (requires get_all)
            # Fall back to semantic search
            st.warning("⚠️ Keyword-only search not available with Pinecone. Using semantic search.")
            results = self.hybrid_searcher.semantic_search(query, n_results=top_k)
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
    # Indexing state is derived directly from Pinecone (the single source of
    # truth). There is no longer an indexing_progress.json ledger to drift.
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Database Status")

    # db_count (a fast describe_index_stats call) doubles as the cache key for
    # get_indexing_state: when the chunk count changes, the cached state recomputes.
    db_count = vector_db.collection.count()
    state = get_indexing_state(vector_db, transcripts_folder, db_count)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Files on disk", state["total_disk"])
        st.metric("Indexed (Pinecone)", state["total_indexed"])
    with col2:
        st.metric("Remaining", state["remaining"])
        st.metric("DB Chunks", db_count)

    # Progress bar
    if state["total_disk"] > 0:
        progress_pct = min(state["total_indexed"] / state["total_disk"], 1.0)
        st.sidebar.progress(progress_pct, text=f"{progress_pct*100:.1f}%")

    st.sidebar.caption("Source of truth: Pinecone (cached ~15 min)")
    if st.sidebar.button("🔄 Refresh indexed state", use_container_width=True,
                         help="Re-read indexed episodes from Pinecone now (clears the cache)"):
        get_indexing_state.clear()
        st.rerun()

    # ---- Indexing (now an offline job) ----
    st.sidebar.divider()
    st.sidebar.markdown("### ⚡ Indexing")

    if state["remaining"] == 0:
        st.sidebar.success("✅ Pinecone is up to date with disk.")
    else:
        st.sidebar.info(
            f"📝 {state['remaining']} new episode(s) to index. "
            "Indexing now runs as an offline job (it no longer happens in this app)."
        )
        st.sidebar.markdown("Run the indexer locally or via GitHub Actions:")
        st.sidebar.code(
            "cd /Users/kevinbridges/POC/SYSK\n"
            "export PINECONE_API_KEY='...'\n"
            "python3 index_transcripts.py",
            language="bash",
        )
        with st.sidebar.expander(f"⏭️ Next up, newest-first ({state['remaining']})"):
            for fname in state["to_index"][:50]:
                st.caption(fname)
            if state["remaining"] > 50:
                st.caption(f"...and {state['remaining'] - 50} more")

    # ---- Database Management ----
    st.sidebar.divider()
    st.sidebar.markdown("### 🔧 Database Management")
    st.sidebar.caption(
        "Destructive operations. After clearing, re-index with the offline "
        "`index_transcripts.py` job."
    )

    confirm_destructive = st.sidebar.checkbox(
        "I understand these actions are destructive", key="confirm_destructive"
    )

    if st.sidebar.button("🗑️ Clear Database", use_container_width=True,
                         disabled=not confirm_destructive):
        with st.spinner("Clearing database..."):
            vector_db.collection.delete(delete_all=True)
        st.sidebar.success("✅ Database cleared. Run index_transcripts.py to rebuild.")
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
                
                # Check if running in cloud
                is_cloud = os.getenv("STREAMLIT_RUNTIME_ENV") or "streamlit.app" in os.getenv("HOSTNAME", "")
                
                if is_cloud:
                    st.warning("⚠️ **Cloud Deployment:** To make this PIN persistent across reboots, add it to Streamlit Cloud Secrets:")
                    st.code(f'ADMIN_PIN_HASH = "{hash_pin(new_pin)}"', language="toml")
                    st.caption("1. Go to share.streamlit.io\n2. Your app → Settings → Secrets\n3. Add the line above\n4. Click Save")
                
                st.info("Please re-login with your new PIN")
                # Log out after successful change
                st.session_state.admin_authenticated = False
                st.rerun()
            else:
                st.error(message)
    
    # ---- Indexed Files List ----
    st.sidebar.divider()
    if state.get("indexed_files"):
        with st.sidebar.expander(f"📝 Indexed Files ({len(state['indexed_files'])})"):
            files_list = sorted(state["indexed_files"])

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
        help="Smart: Auto-selects best method | Hybrid: Combines both | Semantic: AI understanding | Keyword: Exact matches"
    )
    
    top_k = st.sidebar.slider("Number of Results", 3, 20, 10)
    
    with st.sidebar.expander("⚙️ Advanced Search Settings"):
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.1,
                                    help="Weight for AI similarity search")
        keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.5, 0.1,
                                   help="Weight for exact text matching")
        title_weight = st.slider("Title Match Boost", 0.0, 5.0, 2.0, 0.5,
                                 help="Multiplier when query matches episode title")
    
    episode_type_filter = st.sidebar.selectbox(
        "Episode Type Filter",
        options=["All", "Full Episode", "Short Stuff"],
        help="Filter results by episode type"
    )
    
    # ---- Example Prompts ----
    st.sidebar.divider()
    st.sidebar.markdown("### 💡 Example Questions")
    examples = [
        "What episodes discuss artificial intelligence?",
        "Tell me about the episode on data centers",
        "What did Josh and Chuck say about sleep?",
        "Find episodes about space exploration",
        "What are some Short Stuff episodes about technology?"
    ]
    
    for example in examples:
        if st.sidebar.button(example, use_container_width=True, key=f"example_{hash(example)}"):
            # Add example as a message
            st.session_state.admin_messages.append({"role": "user", "content": example})
            st.rerun()
    
    # ---- Admin Info ----
    
    # ========================================================================
    # MAIN AREA - SEARCH INTERFACE (same as public)
    # ========================================================================
    
    st.title("🎙️ SYSK Search - Admin Mode")
    
    # Show search mode and stats
    mode_emoji = {"Smart": "🧠", "Hybrid": "🔄", "Semantic": "🎯", "Keyword": "📝"}
    st.caption(f"{mode_emoji.get(search_mode, '🔍')} Search Mode: **{search_mode}** • {db_count} chunks indexed")
    
    # Add scrolling ticker of random episode titles
    if db_count > 0:
        try:
            # Sample episodes by doing simple semantic searches
            sample_queries = ["the", "how", "what", "work", "history"]
            
            unique_episodes = {}
            
            for query_word in sample_queries[:2]:  # Just use 2 queries
                try:
                    results = vector_db.collection.query(
                        query_texts=[query_word],
                        n_results=50
                    )
                    
                    # Extract unique titles
                    for metadata in results.get('metadatas', [[]])[0]:
                        title = metadata.get('title', '')
                        episode_type = metadata.get('episode_type', 'Full Episode')
                        if title and title not in unique_episodes:
                            unique_episodes[title] = episode_type
                            if len(unique_episodes) >= 15:
                                break
                except:
                    continue
                
                if len(unique_episodes) >= 15:
                    break
            
            if unique_episodes:
                import random
                if len(unique_episodes) > 15:
                    sample_titles = random.sample(list(unique_episodes.items()), 15)
                else:
                    sample_titles = list(unique_episodes.items())
                
                # Create ticker HTML with emojis
                ticker_items = []
                for title, ep_type in sample_titles:
                    emoji = "⚡" if ep_type == "Short Stuff" else "🎙️"
                    ticker_items.append(f"{emoji} {title}")
                
                ticker_text = " • ".join(ticker_items)
                
                # CSS for scrolling ticker
                st.markdown(f"""
                    <style>
                    .ticker-wrapper {{
                        width: 100%;
                        overflow: hidden;
                        background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 50%, #1f1f1f 100%);
                        padding: 10px 0;
                        margin: 15px 0;
                        border-radius: 5px;
                    }}
                    
                    .ticker {{
                        display: inline-block;
                        white-space: nowrap;
                        animation: scroll 60s linear infinite;
                        padding-left: 100%;
                        color: #e0e0e0;
                        font-size: 14px;
                        font-weight: 500;
                    }}
                    
                    @keyframes scroll {{
                        0% {{ transform: translateX(0); }}
                        100% {{ transform: translateX(-100%); }}
                    }}
                    
                    .ticker:hover {{
                        animation-play-state: paused;
                    }}
                    </style>
                    
                    <div class="ticker-wrapper">
                        <div class="ticker">{ticker_text}</div>
                    </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            # Silently fail if ticker can't be generated
            pass
    
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
                        
                        # Show match type badge
                        match_badge = ""
                        if 'match_type' in source and source['match_type']:
                            match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄', 
                                         'title_match': '🏷️', 'title_and_content': '🏷️📝'}
                            match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                        
                        st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                        st.caption(f"{source['date']} • {source['time']}")
                        
                        # Show match details if available
                        if 'match_type' in source and source['match_type'] not in ['semantic_only', None]:
                            st.caption(f"Match type: {source['match_type']}")
                        
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
            with st.spinner(f"Searching with {search_mode} mode..."):
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
                            
                            # Show match type badge
                            match_badge = ""
                            if 'match_type' in source and source['match_type']:
                                match_icons = {'semantic': '🎯', 'keyword': '📝', 'both': '🔄',
                                             'title_match': '🏷️', 'title_and_content': '🏷️📝'}
                                match_badge = f" {match_icons.get(source['match_type'], '🔍')}"
                            
                            st.markdown(f"**{i}.** {episode_type_emoji} **{source['title']}**{match_badge}")
                            st.caption(f"{source['date']} • {source['time']}")
                            
                            # Show match details if available
                            if 'match_type' in source and source['match_type'] not in ['semantic_only', None]:
                                st.caption(f"Match type: {source['match_type']}")
                            
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
    top_k = 5  # Show top 5 most relevant results
    semantic_weight = 0.5
    keyword_weight = 0.5
    title_weight = 2.0
    episode_type_filter = "All"
    
    # Main chat interface
    st.title("🎙️ Stuff You Should Know - Podcast Assistant")
    
    st.caption(f"🧠 Smart Search Mode • 🎙️ {st.session_state.rag_system.vector_db.collection.count()} chunks indexed")
    
    # Add scrolling ticker of random episode titles
    if st.session_state.rag_system.vector_db.collection.count() > 0:
        try:
            # Sample episodes by doing a simple semantic search for common words
            # This is more reliable than dummy vector queries
            sample_queries = ["the", "how", "what", "work", "history"]
            
            unique_episodes = {}
            
            for query_word in sample_queries[:2]:  # Just use 2 queries to be fast
                try:
                    results = st.session_state.rag_system.vector_db.collection.query(
                        query_texts=[query_word],
                        n_results=50
                    )
                    
                    # Extract unique titles from results
                    for metadata in results.get('metadatas', [[]])[0]:
                        title = metadata.get('title', '')
                        episode_type = metadata.get('episode_type', 'Full Episode')
                        if title and title not in unique_episodes:
                            unique_episodes[title] = episode_type
                            if len(unique_episodes) >= 15:
                                break
                except:
                    continue
                
                if len(unique_episodes) >= 15:
                    break
            
            if unique_episodes:
                # Get up to 15 episodes
                import random
                if len(unique_episodes) > 15:
                    sample_titles = random.sample(list(unique_episodes.items()), 15)
                else:
                    sample_titles = list(unique_episodes.items())
                
                # Create ticker HTML with emojis
                ticker_items = []
                for title, ep_type in sample_titles:
                    emoji = "⚡" if ep_type == "Short Stuff" else "🎙️"
                    ticker_items.append(f"{emoji} {title}")
                
                ticker_text = " • ".join(ticker_items)
                
                # CSS for scrolling ticker
                st.markdown(f"""
                    <style>
                    .ticker-wrapper {{
                        width: 100%;
                        overflow: hidden;
                        background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 50%, #1f1f1f 100%);
                        padding: 10px 0;
                        margin: 15px 0;
                        border-radius: 5px;
                    }}
                    
                    .ticker {{
                        display: inline-block;
                        white-space: nowrap;
                        animation: scroll 60s linear infinite;
                        padding-left: 100%;
                        color: #e0e0e0;
                        font-size: 14px;
                        font-weight: 500;
                    }}
                    
                    @keyframes scroll {{
                        0% {{ transform: translateX(0); }}
                        100% {{ transform: translateX(-100%); }}
                    }}
                    
                    .ticker:hover {{
                        animation-play-state: paused;
                    }}
                    </style>
                    
                    <div class="ticker-wrapper">
                        <div class="ticker">{ticker_text}</div>
                    </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            # Silently fail if ticker can't be generated
            pass
    
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
