#!/usr/bin/env python3
"""
SYSK v2 hybrid search: finds EPISODES (not just chunks) by fusing three signals
with Reciprocal Rank Fusion (RRF):

  1. Dense:     bge-small-en-v1.5 vectors in Pinecone "sysk-v2" (meaning)
  2. Keyword:   SQLite FTS5 BM25 over transcript chunks (exact words, names)
  3. Profiles:  SQLite FTS5 BM25 over Haiku episode profiles
                (title, summary, topics, entities, segment titles)

Each signal ranks episodes by their best hit; RRF combines the ranks, so an
episode found by any one signal can surface, and one found by all three wins.
Results keep the best passages per episode (with start times) for display.

The keyword DB (sysk_fts.db, ~260 MB) is built from transcripts + profiles on
first use and is not committed.

CLI:
  python3 search_v2.py "cannibalism survival"            # top 20 episodes
  python3 search_v2.py "cannibalism" --episodes 50 --no-dense
  python3 search_v2.py --rebuild-fts
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path

import chunker_v2

FTS_DB = os.getenv("SYSK_FTS_DB", "sysk_fts.db")
RRF_K = 60
STOP = set("""a an the and or of to in on at for with from by about as is are was were be been
being it its this that these those what which who whom why how when where do does did
have has had not no can could would should will may might than then there their they them
he she his her you your we our i me my all any some more most much many very just also into
over under up down out so if but s t vs""".split())


# ----------------------------------------------------------------- keyword DB
def build_fts(db_path=FTS_DB):
    """Build chunk FTS (via chunker_v2) plus an episode-profile FTS table."""
    n_chunks = chunker_v2.build_fts(db_path)
    db = sqlite3.connect(db_path)
    db.executescript("""
        DROP TABLE IF EXISTS episodes;
        DROP TABLE IF EXISTS episodes_fts;
        CREATE TABLE episodes (filename TEXT PRIMARY KEY, title TEXT, date TEXT,
            episode_kind TEXT, rerun_of TEXT, summary TEXT, episode_url TEXT);
        CREATE VIRTUAL TABLE episodes_fts USING fts5(
            title, topics, entities, segments, summary,
            content='', tokenize='porter unicode61 remove_diacritics 2');
    """)
    n_eps = 0
    for path in sorted(chunker_v2.TRANSCRIPTS.glob("*.txt")):
        p = chunker_v2.load_profile(path.name)
        if not p:
            continue
        n_eps += 1
        db.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?)",
                   (path.name, p.get("title", ""), p.get("date", ""), p.get("episode_kind", ""),
                    p.get("rerun_of") or "", p.get("summary", ""), p.get("episode_url", "")))
        segs = " | ".join(f"{s.get('title', '')}: {' '.join(s.get('topics', []))}"
                          for s in p.get("segments", []) if isinstance(s, dict))
        db.execute("INSERT INTO episodes_fts(rowid, title, topics, entities, segments, summary) "
                   "VALUES ((SELECT rowid FROM episodes WHERE filename=?),?,?,?,?,?)",
                   (path.name, p.get("title", ""), " ".join(p.get("topics", [])),
                    " ".join(e.get("name", "") for e in p.get("entities", []) if isinstance(e, dict)),
                    segs, p.get("summary", "")))
    db.commit()
    db.close()
    return n_chunks, n_eps


def _match_expr(query):
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def keyword_chunks(db, query, limit=300):
    m = _match_expr(query)
    if not m:
        return []
    return db.execute("""
        SELECT c.id, c.filename, c.title, c.date, c.start_sec, c.segment_title, c.text,
               bm25(chunks_fts, 4.0, 2.0, 2.0, 1.0) AS s
        FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?""", (m, limit)).fetchall()


def keyword_episodes(db, query, limit=200):
    m = _match_expr(query)
    if not m:
        return []
    return [r[0] for r in db.execute("""
        SELECT e.filename, bm25(episodes_fts, 6.0, 4.0, 3.0, 2.0, 1.0) AS s
        FROM episodes_fts JOIN episodes e ON e.rowid = episodes_fts.rowid
        WHERE episodes_fts MATCH ? ORDER BY s LIMIT ?""", (m, limit)).fetchall()]


# ------------------------------------------------------------------- search
class HybridSearch:
    def __init__(self, use_dense=True, db_path=FTS_DB):
        if not Path(db_path).exists():
            print(f"Building keyword index {db_path} (first run, ~10 s)...", file=sys.stderr)
            build_fts(db_path)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.index = self.model = None
        if use_dense:
            import index_v2
            self.index = index_v2.connect(create=False)
            self.model = index_v2.load_model()
            self.query_prefix = index_v2.QUERY_PREFIX

    def dense_chunks(self, query, top_k=300):
        q = self.model.encode(self.query_prefix + query, normalize_embeddings=True).tolist()
        res = self.index.query(vector=q, top_k=top_k, include_metadata=True)
        out = []
        for m in res["matches"]:
            md = m["metadata"]
            out.append((m["id"], md.get("filename", ""), md.get("title", ""), md.get("date", ""),
                        int(md.get("start_sec", 0)), md.get("segment_title", ""), md.get("text", ""),
                        float(m["score"])))
        return out

    def search(self, query, n_episodes=20, passages_per_episode=3):
        """Return a ranked list of episodes, each with its best passages."""
        rankings, passages = [], {}

        def add_chunk_ranking(rows):
            order = OrderedDict()
            for r in rows:
                fn = chunker_v2.filename_from_id(r[0]) if not r[1] else r[1]
                order.setdefault(fn, None)
                passages.setdefault(fn, OrderedDict()).setdefault(
                    r[0], {"id": r[0], "start_sec": r[4], "segment_title": r[5], "text": r[6]})
            rankings.append(list(order))

        if self.index is not None:
            add_chunk_ranking(self.dense_chunks(query))
        add_chunk_ranking(keyword_chunks(self.db, query))
        rankings.append(keyword_episodes(self.db, query))

        scores = {}
        for ranking in rankings:
            for rank, fn in enumerate(ranking):
                scores[fn] = scores.get(fn, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scores, key=scores.get, reverse=True)[:n_episodes]

        meta = {r[0]: r for r in self.db.execute(
            f"SELECT filename, title, date, episode_kind, rerun_of, episode_url FROM episodes "
            f"WHERE filename IN ({','.join('?' * len(top))})", top)} if top else {}
        results = []
        for fn in top:
            m = meta.get(fn, (fn, fn, "", "", "", ""))
            results.append({"filename": fn, "title": m[1], "date": m[2], "episode_kind": m[3],
                            "rerun_of": m[4], "episode_url": m[5], "score": round(scores[fn], 5),
                            "passages": list(passages.get(fn, {}).values())[:passages_per_episode]})
        return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no-dense", action="store_true", help="Keyword + profile signals only (no Pinecone).")
    ap.add_argument("--rebuild-fts", action="store_true")
    a = ap.parse_args()
    if a.rebuild_fts:
        print("Indexed %d chunks and %d episode profiles." % build_fts())
        return
    if not a.query:
        ap.print_help()
        return
    hs = HybridSearch(use_dense=not a.no_dense)
    for i, r in enumerate(hs.search(a.query, a.episodes), 1):
        tag = " [rerun]" if r["episode_kind"] == "rerun" else ""
        print(f"{i:2d}. {r['date']}  {r['title']}{tag}")
        for p in r["passages"][:1]:
            print(f"      @{p['start_sec'] // 60}m [{p['segment_title']}] {p['text'][:110]!r}")


if __name__ == "__main__":
    main()
