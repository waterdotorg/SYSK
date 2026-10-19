#!/usr/bin/env python3
"""
SYSK v2 chunker: turns a transcript (+ its episode profile, if present) into
search chunks, and builds the SQLite FTS5 keyword index from them.

Chunks follow speaker turns and run ~60 to 90 seconds (capped by word count so
every chunk fits the embedding model's 512-token window). Each chunk carries:
  * the episode title and the profile's segment title/topics (contextual prefix
    for embedding, and display metadata),
  * start/end seconds, so the UI can play audio from that moment,
  * the episode kind and rerun_of, so reruns can be grouped with originals.

Chunk IDs are deterministic ("<url-quoted filename>#<nnnn>"), so Pinecone and
the keyword index refer to the same chunks.

CLI:
  python3 chunker_v2.py --stats                 # chunk counts / sizes, no writes
  python3 chunker_v2.py --show FILE             # print one episode's chunks
  python3 chunker_v2.py --build-fts sysk_fts.db # (re)build the keyword index
"""
import argparse
import json
import re
import sqlite3
import urllib.parse
from pathlib import Path

TRANSCRIPTS = Path("transcripts")
ENRICHMENT = Path("enrichment")

TARGET_SECONDS = 75     # start a new chunk after this much audio...
MIN_SECONDS = 40        # ...but never before this much
MAX_WORDS = 300         # hard cap (~400 tokens) so chunk + prefix fit in 512 tokens
OVERLAP_TURNS = 1       # repeat the last turn at the start of the next chunk

HEADER_KEYS = {"Episode": "title", "Date": "date", "Duration": "duration",
               "Episode URL": "episode_url", "Audio URL": "audio_url",
               "Transcript URL": "transcript_url"}
TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")


def fix_mojibake(line: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'greatestâ\x80\x94')."""
    if "â" in line or "Ã" in line:
        try:
            return line.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return line


def to_seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def fmt(sec: int) -> str:
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_transcript(path: Path):
    """Return (metadata, [(start_sec, 'Speaker: text'), ...])."""
    raw = path.read_text(encoding="utf-8")
    head, _, body = raw.partition("=" * 80)
    meta = {"filename": path.name}
    for line in head.splitlines():
        k, sep, v = line.partition(": ")
        if sep and k in HEADER_KEYS:
            meta[HEADER_KEYS[k]] = v.strip()
    turns, ts = [], None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if TS_RE.match(line):
            ts = to_seconds(line)
        else:
            turns.append((ts if ts is not None else (turns[-1][0] if turns else 0), fix_mojibake(line)))
            ts = None
    return meta, turns


def load_profile(filename: str):
    p = ENRICHMENT / (Path(filename).stem + ".json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def chunk_id(filename: str, n: int) -> str:
    return f"{urllib.parse.quote(filename, safe='')}#{n:04d}"


def filename_from_id(cid: str) -> str:
    return urllib.parse.unquote(cid.rsplit("#", 1)[0])


def _segment_for(profile, sec: int):
    if not profile:
        return None
    best = None
    for s in profile.get("segments", []):
        try:
            if to_seconds(s["start"]) <= sec:
                best = s
        except Exception:
            continue
    return best


SENT_RE = re.compile(r"(?<=[.!?])\s+")
CHUNK_WORDS_UNTIMED = 200   # words per chunk when a transcript has no timestamps


def _duration_seconds(meta) -> int:
    m = re.search(r"(\d+)", meta.get("duration", ""))
    return int(m.group(1)) * 60 if m else 0


def _split_long(text: str, max_words: int):
    """Split an over-long turn into sentence groups of <= max_words."""
    speaker, sep, body = text.partition(": ")
    prefix = f"{speaker}: " if sep and len(speaker) < 20 else ""
    body = body if prefix else text
    parts, cur, n = [], [], 0
    for sent in SENT_RE.split(body):
        w = len(sent.split())
        if cur and n + w > max_words:
            parts.append(" ".join(cur)); cur, n = [], 0
        cur.append(sent); n += w
    if cur:
        parts.append(" ".join(cur))
    return [prefix + parts[0]] + [prefix + p for p in parts[1:]] if parts else [text]


def _timed_units(turns):
    """Timed transcript -> (start_sec, text) units, splitting over-long turns and
    interpolating their start times up to the next turn."""
    units = []
    for i, (sec, text) in enumerate(turns):
        if len(text.split()) <= MAX_WORDS:
            units.append((sec, text)); continue
        pieces = _split_long(text, MAX_WORDS // 2)
        nxt = turns[i + 1][0] if i + 1 < len(turns) else sec + 30
        step = max(nxt - sec, 1) / len(pieces)
        units += [(int(sec + k * step), p) for k, p in enumerate(pieces)]
    return units


def _untimed_units(turns, duration_sec):
    """Untimed transcript (one big block) -> ~200-word units with start times
    estimated from word position. Speaker labels are dropped: the source does
    not say who is talking."""
    text = " ".join(re.sub(r"^(Josh|Chuck|Speaker \d+): ", "", t) for _, t in turns)
    sents = SENT_RE.split(text)
    total = max(sum(len(x.split()) for x in sents), 1)
    units, cur, n, before = [], [], 0, 0
    for sent in sents:
        w = len(sent.split())
        if cur and n + w > CHUNK_WORDS_UNTIMED:
            units.append((int(before / total * duration_sec), " ".join(cur)))
            before += n; cur, n = [], 0
        cur.append(sent); n += w
    if cur:
        units.append((int(before / total * duration_sec), " ".join(cur)))
    return units


def chunk_episode(path: Path):
    """Return chunk dicts for one transcript file."""
    meta, turns = parse_transcript(path)
    if not turns:
        return []
    profile = load_profile(path.name)
    timed = len({sec for sec, _ in turns}) > 5
    if timed:
        units = _timed_units(turns)
        groups, cur = [], []
        for u in units:
            words = sum(len(t.split()) for _, t in cur)
            span = (u[0] - cur[0][0]) if cur else 0
            if cur and (span >= max(TARGET_SECONDS, MIN_SECONDS) or
                        words + len(u[1].split()) > MAX_WORDS):
                groups.append(cur)
                cur = cur[-OVERLAP_TURNS:] if OVERLAP_TURNS else []
                if sum(len(t.split()) for _, t in cur) + len(u[1].split()) > MAX_WORDS:
                    cur = []
            cur.append(u)
        if cur:
            groups.append(cur)
    else:
        groups = [[u] for u in _untimed_units(turns, _duration_seconds(meta))]

    chunks = []
    for n, g in enumerate(groups):
        start = g[0][0]
        end = groups[n + 1][0][0] if n + 1 < len(groups) else g[-1][0] + 15
        text = "\n".join(t for _, t in g)
        seg = _segment_for(profile, start)
        c = {
            "id": chunk_id(path.name, n),
            "filename": path.name,
            "title": meta.get("title", path.stem),
            "date": meta.get("date", ""),
            "start_sec": start,
            "end_sec": max(end, start + 1),
            "start": fmt(start),
            "approx_time": not timed,       # True: start time estimated from word position
            "speakers_labeled": timed,      # False: source transcript has no speaker labels
            "text": text,
            "segment_title": seg.get("title", "") if seg else "",
            "segment_topics": seg.get("topics", []) if seg else [],
            "kind": seg.get("kind", "content") if seg else "content",
            "episode_kind": (profile or {}).get("episode_kind", ""),
            "rerun_of": (profile or {}).get("rerun_of") or "",
            "episode_url": meta.get("episode_url", ""),
            "audio_url": meta.get("audio_url", ""),
        }
        ctx = c["title"] + (f" | {c['segment_title']}" if c["segment_title"] else "")
        c["embed_text"] = f"{ctx}\n{text}"
        chunks.append(c)
    return chunks


def iter_all_chunks(files=None):
    for path in files or sorted(TRANSCRIPTS.glob("*.txt")):
        for c in chunk_episode(path):
            yield c


# ---------------------------------------------------------------- keyword index
def build_fts(db_path: str, files=None) -> int:
    """(Re)build a SQLite FTS5 BM25 index over all chunks. Returns chunk count."""
    db = sqlite3.connect(db_path)
    db.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;
        CREATE TABLE chunks (id TEXT PRIMARY KEY, filename TEXT, title TEXT, date TEXT,
            start_sec INTEGER, end_sec INTEGER, kind TEXT, episode_kind TEXT,
            rerun_of TEXT, segment_title TEXT, episode_url TEXT, audio_url TEXT, text TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            title, segment_title, topics, text,
            content='', tokenize='porter unicode61 remove_diacritics 2');
    """)
    n = 0
    for c in iter_all_chunks(files):
        if c["kind"] == "ad":
            continue
        n += 1
        db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (c["id"], c["filename"], c["title"], c["date"], c["start_sec"], c["end_sec"],
                    c["kind"], c["episode_kind"], c["rerun_of"], c["segment_title"],
                    c["episode_url"], c["audio_url"], c["text"]))
        db.execute("INSERT INTO chunks_fts(rowid, title, segment_title, topics, text) "
                   "VALUES ((SELECT rowid FROM chunks WHERE id=?),?,?,?,?)",
                   (c["id"], c["title"], c["segment_title"], " ".join(c["segment_topics"]), c["text"]))
    db.commit()
    db.close()
    return n


def fts_search(db_path: str, query: str, limit: int = 50):
    """BM25 keyword search. Title/segment matches weigh more than body text."""
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    db = sqlite3.connect(db_path)
    rows = db.execute(f"""
        SELECT c.id, c.title, c.date, c.start_sec, c.segment_title, c.text,
               bm25(chunks_fts, 4.0, 2.0, 2.0, 1.0) AS score
        FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""", (match, limit)).fetchall()
    db.close()
    return [dict(zip(["id", "title", "date", "start_sec", "segment_title", "text", "score"], r))
            for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--show", type=Path)
    ap.add_argument("--build-fts", metavar="DB")
    ap.add_argument("--search", nargs=2, metavar=("DB", "QUERY"))
    a = ap.parse_args()
    if a.show:
        for c in chunk_episode(a.show):
            print(f"--- {c['id']}  {c['start']}  [{c['kind']}] {c['segment_title']}  "
                  f"({len(c['text'].split())} words)\n{c['text'][:300]}\n")
    elif a.stats:
        n = words = eps = mx = 0
        for path in sorted(TRANSCRIPTS.glob("*.txt")):
            cs = chunk_episode(path)
            eps += 1
            n += len(cs)
            for c in cs:
                w = len(c["embed_text"].split())
                words += w
                mx = max(mx, w)
        print(f"{eps} episodes -> {n:,} chunks; avg {words / max(n, 1):.0f} words, max {mx} words")
    elif a.build_fts:
        print(f"Indexed {build_fts(a.build_fts):,} chunks into {a.build_fts}")
    elif a.search:
        for r in fts_search(*a.search, limit=10):
            print(f"{r['score']:.2f}  {r['title']} @ {r['start_sec'] // 60}m  [{r['segment_title']}]")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
