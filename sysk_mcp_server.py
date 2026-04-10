#!/usr/bin/env python3
"""
SYSK MCP Server - Local stdio Transport
Exposes the Stuff You Should Know podcast archive as MCP tools
for use with Claude Desktop and other MCP-compatible AI clients.

Usage:
    python sysk_mcp_server.py

Configure in Claude Desktop's claude_desktop_config.json.
"""

import json
import os
import re
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ── Logging (stderr only — stdout is reserved for MCP stdio protocol) ──────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [SYSK-MCP] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
PINECONE_INDEX     = os.environ.get("PINECONE_INDEX_NAME", "sysk-transcripts")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384
DEFAULT_TOP_K      = 5
MAX_TOP_K          = 20
TITLE_WEIGHT       = 2.0   # Multiplier for title-matched results
KEYWORD_MIN_SCORE  = 0.3   # Minimum keyword score to include a result


# ── Lazy-loaded singletons (initialised once via lifespan) ─────────────────────
_pinecone_index    = None
_embedding_model   = None


@asynccontextmanager
async def app_lifespan(app):
    """Load heavy dependencies once at server start."""
    global _pinecone_index, _embedding_model

    log.info("Loading sentence-transformer embedding model…")
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model loaded ✓")
    except Exception as exc:
        log.error("Failed to load embedding model: %s", exc)

    log.info("Connecting to Pinecone…")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
        stats = _pinecone_index.describe_index_stats()
        log.info(
            "Pinecone connected ✓  vectors=%s", stats.get("total_vector_count", "?")
        )
    except Exception as exc:
        log.error("Failed to connect to Pinecone: %s", exc)

    yield {}

    log.info("SYSK MCP server shutting down.")


mcp = FastMCP("sysk_mcp", lifespan=app_lifespan)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _require_deps() -> Optional[str]:
    """Return an error string if core dependencies aren't ready."""
    if _embedding_model is None:
        return "Embedding model not loaded. Check server logs."
    if _pinecone_index is None:
        return "Pinecone not connected. Check PINECONE_API_KEY and PINECONE_INDEX_NAME."
    return None


def _embed(text: str) -> List[float]:
    """Convert text to a 384-dim embedding vector."""
    return _embedding_model.encode(text, normalize_embeddings=True).tolist()


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_score(query: str, chunk_text: str, title: str) -> float:
    """
    Simple keyword relevance score in [0, 1].
    Counts what fraction of query tokens appear in the chunk or title.
    """
    tokens = _normalize(query).split()
    if not tokens:
        return 0.0
    norm_chunk = _normalize(chunk_text)
    norm_title = _normalize(title)
    hits = sum(1 for t in tokens if t in norm_chunk or t in norm_title)
    return hits / len(tokens)


def _apply_title_weight(results: List[Dict], query: str) -> List[Dict]:
    """Boost scores for chunks whose episode title contains query terms."""
    tokens = _normalize(query).split()
    for r in results:
        title = _normalize(r.get("title", ""))
        if any(t in title for t in tokens):
            r["score"] = r.get("score", 0) * TITLE_WEIGHT
    return results


def _semantic_search(query: str, top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
    """Query Pinecone with a semantic embedding vector."""
    vector = _embed(query)
    kwargs: Dict[str, Any] = {"vector": vector, "top_k": top_k, "include_metadata": True}
    if filters:
        kwargs["filter"] = filters
    response = _pinecone_index.query(**kwargs)
    return [
        {
            "score": float(m.score),
            "title": m.metadata.get("title", m.metadata.get("episode_title", "Unknown")),
            "date":  m.metadata.get("date", ""),
            "chunk": m.metadata.get("text", ""),
            "duration": m.metadata.get("duration", m.metadata.get("duration_minutes", "")),
            "episode_type": m.metadata.get("episode_type", "Full Episode"),
            "timestamp": m.metadata.get("time", m.metadata.get("timestamp_range", "")),
            "source_file": m.metadata.get("filename", m.metadata.get("source_file", "")),
            "episode_url": m.metadata.get("episode_url", ""),
            "audio_url": m.metadata.get("audio_url", ""),
        }
        for m in response.matches
    ]


def _keyword_search(query: str, top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Keyword search via Pinecone metadata filter + client-side scoring.
    Falls back gracefully when no filter matches.
    """
    tokens = _normalize(query).split()
    # Fetch a broad set then score locally (Pinecone free tier has no full-text)
    vector = _embed(query)
    kwargs: Dict[str, Any] = {
        "vector": vector, "top_k": min(top_k * 4, 100), "include_metadata": True
    }
    if filters:
        kwargs["filter"] = filters
    response = _pinecone_index.query(**kwargs)

    scored = []
    for m in response.matches:
        chunk_text = m.metadata.get("text", "")
        title      = m.metadata.get("title", m.metadata.get("episode_title", ""))
        ks = _keyword_score(query, chunk_text, title)
        if ks >= KEYWORD_MIN_SCORE:
            scored.append({
                "score": ks,
                "title": title,
                "date":  m.metadata.get("date", ""),
                "chunk": chunk_text,
                "duration": m.metadata.get("duration", m.metadata.get("duration_minutes", "")),
                "episode_type": m.metadata.get("episode_type", "Full Episode"),
                "timestamp": m.metadata.get("time", m.metadata.get("timestamp_range", "")),
                "source_file": m.metadata.get("filename", m.metadata.get("source_file", "")),
                "episode_url": m.metadata.get("episode_url", ""),
                "audio_url": m.metadata.get("audio_url", ""),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _hybrid_search(
    query: str,
    top_k: int,
    semantic_weight: float = 0.6,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """Combine semantic + keyword scores, deduplicate by title+timestamp."""
    semantic = _semantic_search(query, top_k * 2, filters)
    keyword  = _keyword_search(query, top_k * 2, filters)

    # Merge by a unique key
    merged: Dict[str, Dict] = {}
    for r in semantic:
        key = f"{r['title']}::{r['timestamp']}"
        r["score"] = r["score"] * semantic_weight
        merged[key] = r

    kw_weight = 1.0 - semantic_weight
    for r in keyword:
        key = f"{r['title']}::{r['timestamp']}"
        if key in merged:
            merged[key]["score"] += r["score"] * kw_weight
        else:
            r["score"] = r["score"] * kw_weight
            merged[key] = r

    results = list(merged.values())
    results = _apply_title_weight(results, query)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _format_results_markdown(results: List[Dict], query: str) -> str:
    """Render search results as readable Markdown with full episode metadata and links."""
    if not results:
        return f"No results found for **{query}**. Try different keywords or switch search modes."

    lines = [f"## Search Results for: *{query}*\n", f"Found **{len(results)}** relevant passages.\n"]
    for i, r in enumerate(results, 1):
        score_pct = f"{min(r['score'] * 100, 100):.0f}%"
        duration  = f" · {r['duration']}" if r.get("duration") else ""
        ts        = f" · \u23f1 {r['timestamp']}" if r.get("timestamp") and r['timestamp'] != "00:00:00 - 00:00:00" else ""
        ep_url    = f"\n\U0001f3a7 [Listen on iHeart]({r['episode_url']})" if r.get("episode_url") else ""
        audio_url = f" · [MP3]({r['audio_url']})" if r.get("audio_url") else ""

        excerpt = r['chunk'][:500].strip()
        ellipsis = '\u2026' if len(r['chunk']) > 500 else ''
        lines.append(
            f"### {i}. {r['title']}\n"
            f"\U0001f4c5 {r['date']}{duration} \u00b7 {r['episode_type']} \u00b7 Relevance: {score_pct}{ts}"
            f"{ep_url}{audio_url}\n\n"
            f"> {excerpt}{ellipsis}\n"
        )
    return "\n".join(lines)


def _format_results_json(results: List[Dict]) -> str:
    """Render search results as a JSON string."""
    return json.dumps(
        [
            {
                "episode_title": r["title"],
                "date": r["date"],
                "duration": r["duration"],
                "episode_type": r["episode_type"],
                "timestamp_range": r["timestamp"],
                "episode_url": r.get("episode_url", ""),
                "audio_url": r.get("audio_url", ""),
                "relevance_score": round(r["score"], 4),
                "excerpt": r["chunk"][:800],
            }
            for r in results
        ],
        indent=2,
    )


def _build_pinecone_filter(
    episode_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Optional[Dict]:
    """Construct a Pinecone metadata filter dict from optional parameters."""
    f: Dict[str, Any] = {}
    if episode_type and episode_type.lower() != "all":
        f["episode_type"] = {"$eq": episode_type}
    if date_from:
        f.setdefault("date", {})["$gte"] = date_from
    if date_to:
        f.setdefault("date", {})["$lte"] = date_to
    return f if f else None


# ── Input models ───────────────────────────────────────────────────────────────

class SearchMode(str, Enum):
    HYBRID   = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON     = "json"


class EpisodeType(str, Enum):
    ALL        = "all"
    FULL       = "Full Episode"
    SHORT      = "Short Stuff"


class SearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    query: str = Field(
        ...,
        description="Natural language question or topic to search for in SYSK transcripts. "
                    "Examples: 'how does Bitcoin work', 'episodes about ancient Rome', 'hypermiling tips'",
        min_length=2,
        max_length=500,
    )
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search strategy: 'hybrid' (recommended) combines semantic + keyword; "
                    "'semantic' uses AI understanding only; 'keyword' uses exact term matching.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="Number of results to return (1–20).",
        ge=1,
        le=MAX_TOP_K,
    )
    episode_type: EpisodeType = Field(
        default=EpisodeType.ALL,
        description="Filter by episode type: 'all', 'Full Episode', or 'Short Stuff'.",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Earliest episode date to include (YYYY-MM-DD). Example: '2015-01-01'",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Latest episode date to include (YYYY-MM-DD). Example: '2020-12-31'",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    semantic_weight: float = Field(
        default=0.6,
        description="In hybrid mode, weight given to semantic results (0.0–1.0). "
                    "Higher = more conceptual; lower = more keyword-focused.",
        ge=0.0,
        le=1.0,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for structured data.",
    )


class EpisodeLookupInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    title: str = Field(
        ...,
        description="Episode title to search for (partial matches supported). "
                    "Example: 'Tulsa Race Massacre' or 'How Caffeine Works'",
        min_length=2,
        max_length=300,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for structured data.",
    )


class StatsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable summary, 'json' for raw data.",
    )


# ── MCP Tools ──────────────────────────────────────────────────────────────────

@mcp.tool(
    name="sysk_search",
    annotations={
        "title": "Search SYSK Podcast Transcripts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def sysk_search(params: SearchInput) -> str:
    """Search the Stuff You Should Know podcast transcript archive by topic or question.

    This is the primary search tool. It searches 40,000+ transcript chunks from 700+
    indexed SYSK episodes using a combination of semantic (AI-powered) and keyword
    matching. Returns relevant passages with episode titles, dates, and excerpts.

    Best used for:
    - Finding what Josh & Chuck said about a specific topic
    - Discovering episodes that touch on a concept
    - Researching how the show covered an event or subject over time

    Args:
        params (SearchInput): Search parameters including:
            - query (str): The topic or question to search for
            - mode (str): 'hybrid' | 'semantic' | 'keyword'
            - top_k (int): Number of results (1–20, default 5)
            - episode_type (str): 'all' | 'Full Episode' | 'Short Stuff'
            - date_from (str): Optional start date filter (YYYY-MM-DD)
            - date_to (str): Optional end date filter (YYYY-MM-DD)
            - semantic_weight (float): Hybrid mode semantic ratio (0.0–1.0)
            - response_format (str): 'markdown' | 'json'

    Returns:
        str: Formatted list of relevant transcript passages with episode metadata.
    """
    err = _require_deps()
    if err:
        return f"Error: {err}"

    try:
        filters = _build_pinecone_filter(
            episode_type=params.episode_type.value if params.episode_type != EpisodeType.ALL else None,
            date_from=params.date_from,
            date_to=params.date_to,
        )

        if params.mode == SearchMode.HYBRID:
            results = _hybrid_search(params.query, params.top_k, params.semantic_weight, filters)
        elif params.mode == SearchMode.SEMANTIC:
            results = _semantic_search(params.query, params.top_k, filters)
            results = _apply_title_weight(results, params.query)
        else:  # KEYWORD
            results = _keyword_search(params.query, params.top_k, filters)

        if params.response_format == ResponseFormat.JSON:
            return _format_results_json(results)
        return _format_results_markdown(results, params.query)

    except Exception as exc:
        log.error("sysk_search error: %s", exc, exc_info=True)
        return f"Error performing search: {exc}. Please try again or simplify your query."


@mcp.tool(
    name="sysk_find_episode",
    annotations={
        "title": "Find a Specific SYSK Episode",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def sysk_find_episode(params: EpisodeLookupInput) -> str:
    """Look up a specific Stuff You Should Know episode by title.

    Use this when you already know (or approximately know) the name of an episode
    and want to find its metadata and a content preview. Supports partial title matching.

    Args:
        params (EpisodeLookupInput): Lookup parameters including:
            - title (str): Episode title to search for (partial match OK)
            - response_format (str): 'markdown' | 'json'

    Returns:
        str: Episode details including full title, date, duration, type, and a
             content excerpt from the transcript.
    """
    err = _require_deps()
    if err:
        return f"Error: {err}"

    try:
        # Embed the title and do a high-recall semantic search
        results = _semantic_search(params.title, top_k=10)
        results = _apply_title_weight(results, params.title)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate: keep one chunk per unique episode title
        seen_titles: set = set()
        unique_episodes = []
        for r in results:
            norm = _normalize(r["title"])
            if norm not in seen_titles:
                seen_titles.add(norm)
                unique_episodes.append(r)

        if not unique_episodes:
            return (
                f"No episodes found matching '{params.title}'. "
                "Try a shorter or different part of the title."
            )

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                [
                    {
                        "episode_title": ep["title"],
                        "date": ep["date"],
                        "duration_minutes": ep["duration"],
                        "episode_type": ep["episode_type"],
                        "excerpt": ep["chunk"][:600],
                    }
                    for ep in unique_episodes[:5]
                ],
                indent=2,
            )

        lines = [f"## Episodes matching: *{params.title}*\n"]
        for i, ep in enumerate(unique_episodes[:5], 1):
            duration = f" · {ep['duration']} min" if ep.get("duration") else ""
            lines.append(
                f"### {i}. {ep['title']}\n"
                f"📅 {ep['date']}{duration} · {ep['episode_type']}\n\n"
                f"> {ep['chunk'][:400].strip()}…\n"
            )
        return "\n".join(lines)

    except Exception as exc:
        log.error("sysk_find_episode error: %s", exc, exc_info=True)
        return f"Error looking up episode: {exc}"


@mcp.tool(
    name="sysk_get_stats",
    annotations={
        "title": "Get SYSK Archive Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def sysk_get_stats(params: StatsInput) -> str:
    """Return statistics about the SYSK podcast archive index.

    Provides a summary of what's currently indexed: total episodes, chunks,
    Pinecone index health, and coverage information.

    Args:
        params (StatsInput): Options including:
            - response_format (str): 'markdown' | 'json'

    Returns:
        str: Archive statistics including vector count, episode count, and index status.
    """
    err = _require_deps()
    if err:
        return f"Error: {err}"

    try:
        stats = _pinecone_index.describe_index_stats()
        total_vectors   = stats.get("total_vector_count", 0)
        namespaces      = stats.get("namespaces", {})
        index_fullness  = stats.get("index_fullness", 0)

        # Estimate episode count from known avg chunks/episode
        est_episodes = round(total_vectors / 58) if total_vectors else 0

        data = {
            "total_chunks_indexed": total_vectors,
            "estimated_episodes_indexed": est_episodes,
            "total_episodes_in_archive": "2,700+",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIM,
            "index_name": PINECONE_INDEX,
            "index_fullness_pct": round(index_fullness * 100, 1),
            "namespaces": list(namespaces.keys()) or ["default"],
            "retrieved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        return (
            f"## 🎙️ SYSK Archive Statistics\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Chunks indexed | {data['total_chunks_indexed']:,} |\n"
            f"| Episodes indexed (est.) | {data['estimated_episodes_indexed']:,} |\n"
            f"| Total episodes in archive | {data['total_episodes_in_archive']} |\n"
            f"| Embedding model | `{data['embedding_model']}` |\n"
            f"| Vector dimensions | {data['embedding_dimensions']} |\n"
            f"| Index fullness | {data['index_fullness_pct']}% |\n"
            f"| Retrieved | {data['retrieved_at']} |\n"
        )

    except Exception as exc:
        log.error("sysk_get_stats error: %s", exc, exc_info=True)
        return f"Error fetching stats: {exc}"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        log.warning(
            "Missing environment variables: %s  —  set them before starting the server.",
            ", ".join(missing),
        )

    log.info("Starting SYSK MCP server (stdio transport)…")
    mcp.run()  # defaults to stdio transport
