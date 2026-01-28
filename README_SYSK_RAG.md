# SYSK Podcast RAG Chatbot

AI-powered chatbot for exploring Stuff You Should Know podcast transcripts using Retrieval-Augmented Generation (RAG).

## Features

- 🎙️ **Semantic Search**: Ask natural language questions about SYSK episodes
- ⚡ **Episode Filtering**: Filter by "Short Stuff" vs "Full Episode"
- 📚 **Source Citations**: See which episodes information came from
- 💬 **Conversational Memory**: Context-aware follow-up questions
- 🕐 **Time-based Chunking**: 3-minute segments for precise retrieval
- 📊 **Episode Statistics**: Track indexed episodes and chunks

## Architecture

Built following the Water Portal Assistant pattern:

- **Frontend**: Streamlit (single-file application)
- **Vector Database**: ChromaDB (embedded, persistent)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude Sonnet 4 via Anthropic API
- **Chunking**: Time-based (3-minute intervals with timestamps)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### 2. Set Anthropic API Key

**Option A: Environment Variable**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Option B: Streamlit Secrets** (for deployment)

Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

### 3. Prepare Transcripts

Place your SYSK transcript `.txt` files in a folder (default: `./transcripts`)

Expected format:
```
Episode: [Title]
Date: YYYY-MM-DD
Duration: X minutes
Transcript URL: [URL]
================================================================================
00:00:01
Josh: [dialogue]
00:00:15
Chuck: [dialogue]
...
```

### 4. Run Locally

```bash
streamlit run sysk_rag_chatbot.py
```

The app will open at `http://localhost:8501`

## Usage

### First Time Setup

1. **Index Transcripts**: Click "📥 Index" in the sidebar
2. **Wait for Indexing**: Progress bar will show indexing status
3. **Start Chatting**: Ask questions about SYSK episodes!

### Example Questions

- "What episodes discuss artificial intelligence?"
- "Tell me about the episode on data centers"
- "What did Josh and Chuck say about sleep?"
- "Find Short Stuff episodes about technology"
- "What are some recent episodes about science?"

### Filters

- **Episode Type**: Filter by "Full Episode" or "Short Stuff"

### Features

- **Conversation History**: Last 5 exchanges maintained for context
- **Source Citations**: Click "📚 View Sources" to see referenced episodes
- **Clear Conversation**: Reset chat with "🗑️ Clear Conversation"
- **Re-index**: Update database with new transcripts using "🔄 Re-index"

## Deployment to Streamlit Community Cloud

### 1. GitHub Setup

Create a new GitHub repository with:
```
sysk-rag-chatbot/
├── sysk_rag_chatbot.py
├── requirements_rag.txt
├── .streamlit/
│   └── secrets.toml  (local only - don't commit!)
└── README.md
```

Create `.gitignore`:
```
.streamlit/secrets.toml
chroma_db/
transcripts/
__pycache__/
*.pyc
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `sysk_rag_chatbot.py`
4. Add secrets in Streamlit Cloud settings:
   ```
   ANTHROPIC_API_KEY = "your-api-key-here"
   ```
5. Deploy!

### 3. Upload Transcripts

**Option A**: Include in repository (if manageable size)
- Add `transcripts/` folder to repo
- May hit GitHub file limits with 1800+ files

**Option B**: Upload via Streamlit interface
- Keep transcripts local
- Upload and index through the UI
- ChromaDB will persist the indexed data

## Cost Estimation

### Storage
- **ChromaDB**: Local/embedded (no cost)
- **Streamlit Community Cloud**: Free tier (1GB storage)

### API Usage
- **Anthropic Claude API**: Pay-per-use
  - Input: ~$3 per million tokens
  - Output: ~$15 per million tokens
  - Estimated: $0.10-0.50 per conversation session
  - Monthly (moderate use): $10-30

### Deployment
- **Streamlit Community Cloud**: Free
- **Total Infrastructure Cost**: $0

## Technical Details

### Chunking Strategy

Transcripts are chunked by **3-minute time intervals** to:
- Preserve conversational context
- Enable precise timestamp citations
- Balance chunk size for retrieval quality

### Vector Search

- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Similarity**: Cosine similarity
- **Top-K**: 5 most relevant chunks per query

### Response Generation

1. Query converted to embedding
2. ChromaDB searches for top 5 similar chunks
3. Chunks + conversation history sent to Claude
4. Claude generates contextual response with citations

### Metadata Tracked

Each chunk includes:
- Episode title
- Date
- Duration
- Episode type (Full/Short Stuff)
- Start/end timestamps
- Filename

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
Set the API key in environment or `.streamlit/secrets.toml`

### "No transcript files found"
Check the transcripts folder path in sidebar

### "Cold start delays"
First load on Streamlit Cloud may take 10-15 seconds

### Database issues
Click "🔄 Re-index" to rebuild the database

## Enhancements (Future)

Following the Water Portal pattern, potential enhancements:

1. **Website Widget**: Embeddable chat widget for websites
2. **FastAPI Wrapper**: REST API for programmatic access
3. **Advanced Filters**: Search by date range, topic, guest appearances
4. **Analytics**: Track popular queries and episode interest
5. **Auto-transcription**: Integrate with Whisper API for new episodes

## Architecture Comparison to Water Portal

| Feature | Water Portal | SYSK Chatbot |
|---------|-------------|--------------|
| Frontend | Streamlit | Streamlit |
| Vector DB | ChromaDB | ChromaDB |
| Embeddings | sentence-transformers | sentence-transformers |
| LLM | Claude Sonnet 4 | Claude Sonnet 4 |
| Chunking | Semantic | Time-based |
| Document Type | Mixed (PDF/DOCX/PPTX) | Text transcripts |
| Use Case | Organizational docs | Personal knowledge |

## License

Personal use project for exploring SYSK podcast content.

---

Built with ❤️ following the Water Portal Assistant architecture pattern.
