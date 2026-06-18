"""
Cross-encoder reranker for the SYSK RAG project (COWORK.md Priority #4).

Bi-encoder semantic search (MiniLM embeddings) is fast but coarse: it embeds the
query and each chunk independently. A cross-encoder scores the (query, chunk) pair
jointly, which is much more accurate at ordering — but too slow to run over the
whole corpus. The standard pattern, which this implements:

    retrieve a candidate pool (top ~30) with the bi-encoder
        -> rerank those candidates with the cross-encoder
            -> pass the top 5 to Claude

Design notes:
  * Streamlit-free, so the app and the eval harness share one implementation.
  * The model is loaded lazily and cached (first call pays the load cost).
  * Any failure (model can't download/load, predict error) falls back to the
    original order and logs a warning — retrieval must never hard-fail because of
    the reranker.
  * Default model is ms-marco-MiniLM-L-6-v2: small (~80MB) and fast, the best
    latency/quality tradeoff for Streamlit Community Cloud's free tier.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache: Dict[str, object] = {}


def _get_model(model_name: str):
    """Lazily load and cache a CrossEncoder model."""
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder  # heavy import, deferred
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank_order(query: str, documents: List[str],
                 model_name: str = DEFAULT_MODEL) -> List[int]:
    """Return indices of ``documents`` sorted by cross-encoder relevance (desc).

    Falls back to the original order on any error so callers never crash.
    """
    n = len(documents)
    if n == 0:
        return []
    try:
        model = _get_model(model_name)
        pairs = [[query, (doc or "")] for doc in documents]
        scores = model.predict(pairs)
        return sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
    except Exception as e:
        logger.warning("Reranker unavailable (%s); keeping original order.", e)
        return list(range(n))


def rrf_fuse(orderings: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal-rank fusion of several rankings.

    Each ordering is a list of item indices, best first. An item's fused score is
    sum(1 / (k + rank)) across rankings (rank is 1-based). Items ranked highly by
    multiple rankers rise; an item one ranker places badly is only dampened, not
    dominated. Returns indices sorted by fused score (desc).
    """
    from collections import defaultdict
    scores = defaultdict(float)
    for order in orderings:
        for rank, idx in enumerate(order, 1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def rrf_rerank_results(query: str, results: Dict, top_n: Optional[int] = None,
                       k: int = 60, model_name: str = DEFAULT_MODEL) -> Dict:
    """Fuse the semantic order with the cross-encoder order via RRF, keep top_n.

    The incoming ``results`` are assumed to already be in semantic rank order
    (that is how the search layer returns them). We compute the cross-encoder
    order over the same candidates and RRF-fuse the two. Falls back gracefully:
    if the cross-encoder can't load, rerank_order returns the identity order and
    the fusion reduces to the semantic order.
    """
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    m = min(len(ids), len(docs), len(metas))
    if m == 0:
        return results
    ids, docs, metas = ids[:m], docs[:m], metas[:m]

    semantic_order = list(range(m))                  # already semantic-ranked
    ce_order = rerank_order(query, docs, model_name)  # cross-encoder ranking
    fused = rrf_fuse([semantic_order, ce_order], k=k)
    if top_n is not None:
        fused = fused[:top_n]

    return {
        "ids": [[ids[i] for i in fused]],
        "documents": [[docs[i] for i in fused]],
        "metadatas": [[metas[i] for i in fused]],
        "reranked": True,
        "fusion": "rrf",
    }


def rerank_results(query: str, results: Dict, top_n: Optional[int] = None,
                   model_name: str = DEFAULT_MODEL) -> Dict:
    """Reorder a Chroma-style results dict by cross-encoder score; truncate to top_n.

    ``results`` is the shape returned by the search layer:
    ``{"ids": [[...]], "documents": [[...]], "metadatas": [[...]]}``.
    Returns the same shape (with a ``"reranked": True`` marker). If there are no
    results, the input is returned unchanged.
    """
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    m = min(len(ids), len(docs), len(metas))
    if m == 0:
        return results

    ids, docs, metas = ids[:m], docs[:m], metas[:m]
    order = rerank_order(query, docs, model_name)
    if top_n is not None:
        order = order[:top_n]

    return {
        "ids": [[ids[i] for i in order]],
        "documents": [[docs[i] for i in order]],
        "metadatas": [[metas[i] for i in order]],
        "reranked": True,
    }
