#!/usr/bin/env python3
"""
SYSK v2 indexer: embeds chunker_v2 chunks with bge-small-en-v1.5 and upserts
them into a NEW Pinecone index ("sysk-v2"), leaving the live "sysk-transcripts"
index untouched until we cut over.

Why a new model: all-MiniLM-L6-v2 reads only 256 tokens, so most of each old
3-minute chunk was invisible to semantic search. bge-small-en-v1.5 reads 512
tokens, keeps the same 384 dimensions and size, and v2 chunks (~200 words,
max ~315) fit entirely.

Usage (needs PINECONE_API_KEY; first run downloads the model, ~130 MB):
  python3 index_v2.py --dry-run           # show what would be indexed
  python3 index_v2.py --limit 20          # index the 20 newest missing episodes
  python3 index_v2.py                     # index everything missing
  python3 index_v2.py --rebuild FILE ...  # re-chunk + re-embed specific episodes
  python3 index_v2.py --query "pirate radio"   # quick semantic test
"""
import argparse
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import chunker_v2
import pinecone_state  # reuse its tolerant index.list() page/ID parsing

INDEX_NAME = os.getenv("SYSK_V2_INDEX", "sysk-v2")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
# bge v1.5 retrieval convention: prefix QUERIES (not passages) with this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
UPSERT_BATCH = 100
METADATA_KEYS = ("filename", "title", "date", "start_sec", "end_sec", "approx_time",
                 "speakers_labeled", "segment_title", "kind", "episode_kind", "rerun_of",
                 "episode_url", "audio_url", "text")


def api_key():
    key = os.getenv("PINECONE_API_KEY", "")
    if not key:
        s = Path(".streamlit/secrets.toml")
        if s.exists():
            m = re.search(r'^PINECONE_API_KEY\s*=\s*"([^"]+)"', s.read_text(), re.M)
            key = m.group(1) if m else ""
    if not key:
        sys.exit("PINECONE_API_KEY not set (env or .streamlit/secrets.toml).")
    return key


def connect(create=True):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=api_key())
    names = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in names:
        if not create:
            sys.exit(f"Index {INDEX_NAME} does not exist yet.")
        print(f"Creating Pinecone index {INDEX_NAME} ({DIMENSION}d, cosine)...")
        pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        for _ in range(60):
            if INDEX_NAME in [i.name for i in pc.list_indexes()]:
                break
            time.sleep(2)
        time.sleep(10)  # serverless indexes take a moment before accepting writes
    return pc.Index(INDEX_NAME)


def indexed_ids(index, prefix=None):
    kwargs = {"prefix": prefix} if prefix else {}
    for page in index.list(**kwargs):
        for item in pinecone_state._iter_page_items(page):
            vid = pinecone_state._coerce_id(item)
            if vid:
                yield vid


def indexed_filenames(index):
    return {pinecone_state._nfc(chunker_v2.filename_from_id(v)) for v in indexed_ids(index)}


def load_model():
    """Load the embedding model. SYSK_EMBED_DEVICE=mps|cuda opts into a GPU (default cpu)
    (cpu avoids Apple-GPU memory build-up entirely, at some speed cost)."""
    from sentence_transformers import SentenceTransformer
    device = os.getenv("SYSK_EMBED_DEVICE", "cpu")  # Apple MPS hit OOM on long runs
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"Embedding on device: {model.device}")
    return model


def free_accelerator_memory():
    """Release cached GPU memory between episodes (PyTorch's Apple MPS backend
    otherwise keeps growing its cache until it runs out)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def to_record(c, vec):
    md = {k: c[k] for k in METADATA_KEYS if k in c and c[k] is not None}
    md["text"] = md.get("text", "")[:3000]
    md["segment_topics"] = c.get("segment_topics", [])[:12]
    return {"id": c["id"], "values": vec, "metadata": md}


def index_files(index, model, files, rebuild=False):
    total = 0
    t0 = time.time()
    for n, path in enumerate(files, 1):
        chunks = [c for c in chunker_v2.chunk_episode(path) if c["kind"] != "ad"]
        if not chunks:
            print(f"[{n}/{len(files)}] {path.name}: no chunks, skipped")
            continue
        if rebuild:  # drop this episode's old vectors first (chunk count may change)
            old = list(indexed_ids(index, prefix=urllib.parse.quote(path.name, safe="") + "#"))
            for i in range(0, len(old), 1000):
                index.delete(ids=old[i:i + 1000])
        vecs = model.encode([c["embed_text"] for c in chunks], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False).tolist()
        free_accelerator_memory()
        recs = [to_record(c, v) for c, v in zip(chunks, vecs)]
        for i in range(0, len(recs), UPSERT_BATCH):
            index.upsert(vectors=recs[i:i + UPSERT_BATCH])
        total += len(recs)
        if n % 25 == 0 or n == len(files):
            rate = n / max(time.time() - t0, 1)
            print(f"[{n}/{len(files)}] {total:,} chunks so far "
                  f"(~{(len(files) - n) / rate / 60:.0f} min left)")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", nargs="+", type=Path)
    ap.add_argument("--query")
    a = ap.parse_args()

    if a.query:
        index, model = connect(create=False), load_model()
        q = model.encode(QUERY_PREFIX + a.query, normalize_embeddings=True).tolist()
        res = index.query(vector=q, top_k=10, include_metadata=True)
        for m in res["matches"]:
            md = m["metadata"]
            approx = "~" if md.get("approx_time") else ""
            print(f"{m['score']:.3f}  {md.get('title')} @ {approx}{int(md.get('start_sec', 0)) // 60}m"
                  f"  [{md.get('segment_title', '')}]")
        return

    index = connect()
    if a.rebuild:
        print(f"Rebuilt {index_files(index, load_model(), a.rebuild, rebuild=True):,} chunks.")
        return

    disk = sorted(chunker_v2.TRANSCRIPTS.glob("*.txt"), key=lambda p: p.name, reverse=True)
    have = indexed_filenames(index)
    todo = [p for p in disk if pinecone_state._nfc(p.name) not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(disk)} transcripts on disk, {len(have)} already in {INDEX_NAME}, {len(todo)} to index.")
    no_profile = [p.name for p in todo if not chunker_v2.load_profile(p.name)]
    if no_profile:
        print(f"Note: {len(no_profile)} of these have no episode profile yet "
              f"(chunks will lack segment titles); e.g. {no_profile[0]}")
    if a.dry_run or not todo:
        return
    n = index_files(index, load_model(), todo)
    print(f"Done: {n:,} chunks upserted into {INDEX_NAME}.")


if __name__ == "__main__":
    main()
