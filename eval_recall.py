#!/usr/bin/env python3
"""
"Find EVERY episode on a topic" eval: old index (v1) vs new hybrid search (v2).

The old eval asked "is THE right episode in the top k?". The real goal is
exhaustive: every episode that substantively discusses a topic. This harness:

  label  Builds SILVER labels: Claude Haiku 5.5 reads the whole catalog of
         episode profiles (title, summary, topics, segment titles) for each topic
         and lists every episode where the topic is the main subject ("main") or
         a meaningful segment ("substantial"). Saved to eval/recall_topics.json for
         human review. Cost: well under $1 for 15 topics (catalog is prompt-cached).

  run    For each topic, ranks episodes with
           v1       old index (MiniLM, 3-min chunks, "sysk-transcripts")
           v1-chat  what the old chatbot actually saw (top 20 chunks)
           v2       new hybrid search (bge dense + BM25 chunks + BM25 profiles)
         and reports episode recall at 10/25/50 against the labels.

Usage:
  python3 eval_recall.py label            # once (re-run --force to relabel)
  python3 eval_recall.py run
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
from pathlib import Path

import chunker_v2
import enrich_episodes as ee  # api_key() + post() helpers (Anthropic API, stdlib only)

LABELS = Path("eval/recall_topics.json")
CHUNK_EPISODES = 150  # ~65-85K tokens per call: stays inside Haiku 5.5's cheaper <=100K-token tier

TOPICS = [
    ("cannibalism", "cannibalism"),
    ("prohibition", "Prohibition and bootlegging"),
    ("cia", "the CIA and its operations"),
    ("nuclear", "nuclear weapons"),
    ("cryptids", "Bigfoot and other cryptids"),
    ("serial_killers", "serial killers"),
    ("shipwrecks", "shipwrecks and disasters at sea"),
    ("caffeine", "coffee and caffeine"),
    ("pirates", "pirates and piracy"),
    ("vampires", "vampires"),
    ("depression", "the Great Depression"),
    ("space", "NASA and space exploration"),
    ("cults", "cults"),
    ("psychedelics", "psychedelic drugs"),
    ("cold_war", "the Cold War"),
]

LABEL_TOOL = {
    "name": "report_episodes",
    "description": "Report which catalog entries discuss the topic.",
    "input_schema": {"type": "object", "properties": {
        "main": {"type": "array", "items": {"type": "integer"},
                 "description": "IDs where the topic is the episode's main subject."},
        "substantial": {"type": "array", "items": {"type": "integer"},
                        "description": "IDs where the topic gets a meaningful segment or extended discussion, but is not the main subject."},
    }, "required": ["main", "substantial"]},
}


def catalog():
    rows = []
    for path in sorted(chunker_v2.TRANSCRIPTS.glob("*.txt")):
        p = chunker_v2.load_profile(path.name)
        if not p:
            continue
        segs = "; ".join(s.get("title", "") for s in p.get("segments", [])
                         if isinstance(s, dict) and s.get("kind") == "content")
        rows.append((path.name, f"{p.get('date', '')} | {p.get('title', '')} | {p.get('summary', '')} "
                                f"| Topics: {', '.join(p.get('topics', []))} | Segments: {segs}"))
    return rows


def _ids(value):
    """Normalize the model's ID list: it sometimes returns a JSON string, or IDs
    as strings like "123" or "[123]"."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = re.findall(r"\d+", value)
    if not isinstance(value, list):
        value = [value]
    out = []
    for v in value:
        m = re.search(r"\d+", str(v))
        if m:
            out.append(int(m.group()))
    return out


def label_chunk(lines, offset, topics):
    """One catalog chunk, all topics sequentially (catalog block is prompt-cached)."""
    block = "\n".join(f"[{offset + i}] {text}" for i, (_, text) in enumerate(lines))
    out = {}
    for key, desc in topics:
        payload = {
            "model": ee.MODEL, "max_tokens": 4000,
            "system": [{"type": "text", "text":
                        "You label podcast episodes for a search evaluation. Each catalog line is "
                        "[ID] date | title | summary | topics | segment titles.\n\n" + block,
                        "cache_control": {"type": "ephemeral"}}],
            "tools": [LABEL_TOOL], "tool_choice": {"type": "tool", "name": "report_episodes"},
            "messages": [{"role": "user", "content":
                          f"Topic: {desc}\n\nList every catalog ID where this topic is the main subject "
                          f"('main') or gets a meaningful segment or extended discussion ('substantial'). "
                          f"Exclude passing mentions. Be thorough: check every line."}],
        }
        msg = json.loads(ee.post("/messages", payload))
        block_out = next((b for b in msg.get("content", []) if b.get("type") == "tool_use"), {})
        data = block_out.get("input", {})
        out[key] = {k: [i for i in _ids(data.get(k)) if offset <= i < offset + len(lines)]
                    for k in ("main", "substantial")}
    return out


def cmd_label(force=False):
    if LABELS.exists() and not force:
        sys.exit(f"{LABELS} exists; use --force to relabel.")
    cat = catalog()
    chunks = [(cat[i:i + CHUNK_EPISODES], i) for i in range(0, len(cat), CHUNK_EPISODES)]
    print(f"Labeling {len(TOPICS)} topics over {len(cat)} episodes in {len(chunks)} catalog chunks...")
    merged = {k: {"main": set(), "substantial": set()} for k, _ in TOPICS}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(lambda c: label_chunk(c[0], c[1], TOPICS), chunks):
            for k, v in res.items():
                for kind in ("main", "substantial"):
                    merged[k][kind].update(v[kind])
    out = {"_README": "Silver labels from Claude Haiku 5.5 reading episode profiles. Review and edit "
                      "freely: move filenames between lists or delete false positives.",
           "created": dt.date.today().isoformat(), "topics": {}}
    for key, desc in TOPICS:
        main = sorted(cat[i][0] for i in merged[key]["main"])
        sub = sorted(cat[i][0] for i in merged[key]["substantial"] - merged[key]["main"])
        out["topics"][key] = {"query": desc, "main": main, "substantial": sub}
        print(f"  {key:15s} main={len(main):3d}  substantial={len(sub):3d}")
    LABELS.parent.mkdir(exist_ok=True)
    LABELS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {LABELS}. Skim it for obvious mistakes before running the comparison.")


# ------------------------------------------------------------------- run
class V1:
    """The old index exactly as the live app queries it."""
    def __init__(self):
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        import index_v2, pinecone_state
        self.ps = pinecone_state
        self.index = Pinecone(api_key=index_v2.api_key()).Index("sysk-transcripts")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def episodes(self, query, top_k=300):
        res = self.index.query(vector=self.model.encode(query).tolist(), top_k=top_k, include_metadata=True)
        order = []
        for m in res["matches"]:
            fn = m["metadata"].get("filename") or self.ps.filename_from_vector_id(m["id"])
            fn = self.ps._nfc(fn or "")
            if fn and fn not in order:
                order.append(fn)
        return order


def recall(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / len(gold) if gold else float("nan")


def precision(ranked, gold, k):
    """Share of the top k that are relevant (fair for topics with more than k episodes)."""
    top = ranked[:k]
    return len(set(top) & gold) / len(top) if top else 0.0


def r_precision(ranked, gold):
    """Recall at k = |gold|: 100% means the first |gold| results are exactly the gold set."""
    return recall(ranked, gold, len(gold))


def cmd_run():
    import search_v2, pinecone_state
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["topics"]
    v1, v2 = V1(), search_v2.HybridSearch()
    rows, totals = [], {}
    print(f"{'topic':15s} {'gold':>4s} | {'old-chat':>8s} | {'P@25 v1':>7s} {'v2':>4s} | "
          f"{'R-prec v1':>9s} {'v2':>4s} | {'R@100 v1':>8s} {'v2':>4s}")
    for key, t in labels.items():
        gold = {pinecone_state._nfc(f) for f in t["main"] + t["substantial"]}
        main = {pinecone_state._nfc(f) for f in t["main"]}
        r1 = v1.episodes(t["query"], top_k=1000)
        r1chat = v1.episodes(t["query"], top_k=20)
        r2 = [pinecone_state._nfc(r["filename"]) for r in v2.search(t["query"], n_episodes=150)]
        row = {"topic": key, "gold": len(gold), "main": len(main),
               "v1chat_recall": recall(r1chat, gold, 999),
               "v1_p25": precision(r1, gold, 25), "v2_p25": precision(r2, gold, 25),
               "v1_rprec": r_precision(r1, gold), "v2_rprec": r_precision(r2, gold),
               "v1_r100": recall(r1, gold, 100), "v2_r100": recall(r2, gold, 100),
               "v1_main_r25": recall(r1, main, 25), "v2_main_r25": recall(r2, main, 25),
               "v1_ranked_episodes": len(r1),
               "v2_missed_top100": sorted(gold - set(r2[:100]))}
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                totals.setdefault(k, []).append(v)
        print(f"{key:15s} {len(gold):4d} | {row['v1chat_recall']:8.0%} | {row['v1_p25']:7.0%} {row['v2_p25']:4.0%} | "
              f"{row['v1_rprec']:9.0%} {row['v2_rprec']:4.0%} | {row['v1_r100']:8.0%} {row['v2_r100']:4.0%}")
    avg = {k: sum(v) / len(v) for k, v in totals.items()}
    print("-" * 78)
    print(f"{'AVERAGE':15s} {'':4s} | {avg['v1chat_recall']:8.0%} | {avg['v1_p25']:7.0%} {avg['v2_p25']:4.0%} | "
          f"{avg['v1_rprec']:9.0%} {avg['v2_rprec']:4.0%} | {avg['v1_r100']:8.0%} {avg['v2_r100']:4.0%}")
    print(f"Main-subject episodes in top 25:  v1 {avg['v1_main_r25']:.0%}   v2 {avg['v2_main_r25']:.0%}")
    print("Key: old-chat = share of relevant episodes the old chatbot's top-20 chunks touched; "
          "P@25 = relevant share of the top 25; R-prec = recall at k = #relevant; R@100 = recall in top 100.")
    out = Path("eval/results") / f"recall_{dt.datetime.now():%Y%m%dT%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"average": avg, "topics": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Details (including episodes v2 missed) saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["label", "run"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cmd_label(a.force) if a.cmd == "label" else cmd_run()
