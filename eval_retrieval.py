#!/usr/bin/env python3
"""
Retrieval eval harness for the SYSK RAG project (COWORK.md Priority #8).

Measures how well retrieval surfaces the known-correct episode for a set of
golden questions, so weight/chunk tuning — and, next, the reranker (#4) — becomes
measurement instead of guesswork.

Metrics (per run, over the golden set):
  * recall@k for k in {1,3,5,10,20,30}: fraction of questions whose correct
    episode appears within the top-k retrieved *episodes* (chunks are de-duped to
    episodes, preserving rank order).
  * MRR: mean reciprocal rank of the first correct episode (0 if not found).

Why recall@20/30 matters: the planned reranker retrieves a candidate pool
(top 20-30) and reorders it. recall@30 is the ceiling the reranker can achieve;
recall@5 is today's end-quality. Comparing both before/after proves the reranker.

Modes:
  * semantic (default): one Pinecone query per question (fast). This is the
    production "Smart" path and the reranker's candidate source.
  * hybrid (--include-hybrid): also evaluates hybrid search across one or more
    title-boost values. NOTE: hybrid/keyword search calls collection.get(), which
    fetches the ENTIRE index (~45k vectors) per query — very slow. Opt-in only.

Usage:
    export PINECONE_API_KEY='...'
    ./venv/bin/python3 eval_retrieval.py
    ./venv/bin/python3 eval_retrieval.py --top-k 30
    ./venv/bin/python3 eval_retrieval.py --include-hybrid --title-boosts 1 2 5

Writes a timestamped JSON + Markdown report to eval/results/. Reads nothing from,
and writes nothing to, Pinecone beyond read-only queries.
"""

import argparse
import json
import logging
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pinecone_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("eval_retrieval")

RECALL_KS = [1, 3, 5, 10, 20, 30]
DEFAULT_GOLDEN = "eval/golden_questions.json"
DEFAULT_INDEX = os.getenv("PINECONE_INDEX_NAME", "sysk-transcripts")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_golden(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) else data
    cleaned = []
    for q in questions:
        expected = [_nfc(e) for e in q.get("expected_episodes", []) if e]
        if not q.get("question") or not expected:
            logger.warning("Skipping malformed golden item: %s", q.get("id", q))
            continue
        cleaned.append({"id": q.get("id", ""), "question": q["question"],
                        "expected": set(expected), "note": q.get("note", "")})
    return cleaned


def ranked_episodes(results: Dict) -> List[str]:
    """De-dupe retrieved chunks to a rank-ordered list of unique episode filenames."""
    metas = (results.get("metadatas") or [[]])
    ids = (results.get("ids") or [[]])
    metas0 = metas[0] if metas else []
    ids0 = ids[0] if ids else []
    out, seen = [], set()
    for i, meta in enumerate(metas0):
        filename = (meta or {}).get("filename")
        if not filename and i < len(ids0):
            filename = pinecone_state.filename_from_vector_id(ids0[i])
        if not filename:
            continue
        filename = _nfc(filename)
        if filename not in seen:
            seen.add(filename)
            out.append(filename)
    return out


def first_hit_rank(episodes: List[str], expected: set) -> Optional[int]:
    """1-based rank of the first episode that matches an expected filename, else None."""
    for idx, ep in enumerate(episodes, 1):
        if ep in expected:
            return idx
    return None


def _replace_documents(results: Dict, text_provider) -> Dict:
    """Return a copy of results with each chunk's document replaced by full text.

    text_provider(metadata, vector_id) -> full chunk text (or None to keep the
    original truncated document).
    """
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    new_docs = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        vid = ids[i] if i < len(ids) else ""
        full = text_provider(meta, vid)
        new_docs.append(full if full else doc)
    return {"ids": [ids], "documents": [new_docs], "metadatas": [metas]}


def evaluate(searcher, golden: List[Dict], top_k: int, mode: str,
             text_provider=None, **search_kwargs) -> Dict:
    """Run one config over the golden set and return metrics + per-question detail."""
    per_question = []
    for q in golden:
        if mode == "semantic":
            results = searcher.semantic_search(q["question"], n_results=top_k)
        elif mode == "reranked":
            # Same candidate pool as semantic, reordered by the cross-encoder.
            import reranker
            results = searcher.semantic_search(q["question"], n_results=top_k)
            if text_provider is not None:
                results = _replace_documents(results, text_provider)  # rerank on FULL text
            results = reranker.rerank_results(q["question"], results, top_n=top_k,
                                              model_name=search_kwargs.get("model_name",
                                                                          reranker.DEFAULT_MODEL))
        elif mode == "hybrid":
            results = searcher.hybrid_search(q["question"], n_results=top_k, **search_kwargs)
        else:
            raise ValueError(mode)
        eps = ranked_episodes(results)
        rank = first_hit_rank(eps, q["expected"])
        per_question.append({
            "id": q["id"], "question": q["question"], "note": q["note"],
            "rank": rank, "top_episode": eps[0] if eps else None,
            "expected": sorted(q["expected"]),
        })

    n = len(golden) or 1
    recall = {k: sum(1 for r in per_question if r["rank"] and r["rank"] <= k) / n
              for k in RECALL_KS if k <= top_k}
    mrr = sum((1.0 / r["rank"]) for r in per_question if r["rank"]) / n
    return {"mode": mode, "search_kwargs": search_kwargs, "n_questions": len(golden),
            "recall": recall, "mrr": mrr, "per_question": per_question}


def print_report(result: Dict):
    label = result["mode"]
    if result["search_kwargs"]:
        label += " " + ", ".join(f"{k}={v}" for k, v in result["search_kwargs"].items())
    print("\n" + "=" * 72)
    print(f"CONFIG: {label}   (n={result['n_questions']})")
    print("=" * 72)
    print("  " + "  ".join(f"R@{k}={v:.0%}" for k, v in result["recall"].items())
          + f"   MRR={result['mrr']:.3f}")
    max_k = max(result["recall"].keys()) if result["recall"] else 0
    misses = [r for r in result["per_question"] if not r["rank"]]
    weak = [r for r in result["per_question"] if r["rank"] and r["rank"] > 5]
    if misses:
        print(f"\n  MISSES ({len(misses)}) — correct episode not in top {max_k}:")
        for r in misses:
            print(f"    [{r['id']}] {r['question']}")
            print(f"        top result: {r['top_episode']}")
    if weak:
        print(f"\n  WEAK (found but rank 6+):")
        for r in weak:
            print(f"    [{r['id']}] rank {r['rank']}: {r['question']}")
    if not misses and not weak:
        print("\n  All golden questions hit within top 5. ✅")


def main() -> int:
    p = argparse.ArgumentParser(description="SYSK retrieval eval harness")
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--index-name", default=DEFAULT_INDEX)
    p.add_argument("--top-k", type=int, default=30, help="Candidate pool size to retrieve")
    p.add_argument("--rerank", action="store_true",
                   help="Also evaluate cross-encoder reranking of the semantic candidate pool")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="Cross-encoder model for --rerank")
    p.add_argument("--rerank-fulltext", action="store_true",
                   help="Rerank on FULL chunk text re-read from disk (not the 1000-char "
                        "Pinecone copy). Requires --transcripts-folder to be present.")
    p.add_argument("--transcripts-folder", default="transcripts",
                   help="Transcript folder for --rerank-fulltext")
    p.add_argument("--include-hybrid", action="store_true",
                   help="Also evaluate hybrid search (SLOW: fetches the whole index per query)")
    p.add_argument("--title-boosts", type=float, nargs="*", default=[2.0],
                   help="Title-boost values to sweep in hybrid mode")
    p.add_argument("--out-dir", default="eval/results")
    args = p.parse_args()

    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set. export PINECONE_API_KEY='...' and retry.")
        return 1
    if not Path(args.golden).exists():
        logger.error("Golden set not found: %s", args.golden)
        return 1

    golden = load_golden(args.golden)
    logger.info("Loaded %d golden questions from %s", len(golden), args.golden)

    # Heavy imports here so --help stays fast.
    from pinecone_vector_db import PineconeVectorDatabase
    from hybrid_search import HybridSearcher

    logger.info("Connecting to Pinecone '%s' and loading the embedding model...", args.index_name)
    db = PineconeVectorDatabase(index_name=args.index_name)
    searcher = HybridSearcher(db)  # HybridSearcher expects an object exposing .query()/.get()

    results = []
    logger.info("Evaluating SEMANTIC retrieval (top_k=%d)...", args.top_k)
    results.append(evaluate(searcher, golden, args.top_k, "semantic"))

    if args.rerank:
        logger.info("Evaluating RERANKED retrieval (cross-encoder %s)...", args.rerank_model)
        results.append(evaluate(searcher, golden, args.top_k, "reranked",
                                model_name=args.rerank_model))

    if args.rerank_fulltext:
        from transcript_processor import TranscriptProcessor
        proc = TranscriptProcessor()
        tdir = Path(args.transcripts_folder)
        _text_cache: Dict[str, Dict[int, str]] = {}

        def text_provider(meta, vid):
            meta = meta or {}
            fn = meta.get("filename") or pinecone_state.filename_from_vector_id(vid)
            cid = meta.get("chunk_id")
            if fn is None or cid is None:
                return None
            try:
                cid = int(cid)  # Pinecone may return numeric metadata as float
            except (TypeError, ValueError):
                return None
            if fn not in _text_cache:
                _text_cache[fn] = proc.chunk_text_map(str(tdir / fn))
            return _text_cache[fn].get(cid)

        logger.info("Evaluating RERANKED+FULLTEXT retrieval (cross-encoder %s)...", args.rerank_model)
        r = evaluate(searcher, golden, args.top_k, "reranked",
                     text_provider=text_provider, model_name=args.rerank_model)
        r["mode"] = "reranked+fulltext"
        results.append(r)

    if args.include_hybrid:
        logger.warning("Hybrid mode fetches the entire index per query — this is slow.")
        for tb in args.title_boosts:
            logger.info("Evaluating HYBRID retrieval (title_boost=%s)...", tb)
            results.append(evaluate(searcher, golden, args.top_k, "hybrid",
                                    semantic_weight=0.5, keyword_weight=0.5,
                                    title_boost=tb, title_weight=tb))

    for r in results:
        print_report(r)

    # Persist results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"timestamp_utc": stamp, "index": args.index_name, "top_k": args.top_k,
               "golden_file": args.golden, "results": results}
    json_path = out_dir / f"eval_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f"eval_{stamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# SYSK retrieval eval — {stamp}\n\n")
        f.write(f"- Index: `{args.index_name}` · top_k={args.top_k} · "
                f"{len(golden)} golden questions\n\n")
        f.write("| Config | " + " | ".join(f"R@{k}" for k in RECALL_KS) + " | MRR |\n")
        f.write("|" + "---|" * (len(RECALL_KS) + 2) + "\n")
        for r in results:
            tb = r["search_kwargs"].get("title_boost")
            cfg = r["mode"] + (f" (tb={tb})" if tb is not None else "")
            cells = [f"{r['recall'].get(k, float('nan')):.0%}" if k in r["recall"] else "—"
                     for k in RECALL_KS]
            f.write(f"| {cfg} | " + " | ".join(cells) + f" | {r['mrr']:.3f} |\n")
    logger.info("Saved results: %s  and  %s", json_path, md_path)
    print(f"\nSaved: {json_path}\n       {md_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(1)
