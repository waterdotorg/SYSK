#!/usr/bin/env python3
"""
SYSK Deep Research: find EVERY episode that discusses a topic, pull out the
actual dialogue (with timestamps), and write a cited cross-episode summary.

Pipeline
  1. Catalog scan   Claude Haiku 5.5 reads all ~2,600 episode profiles (summary,
                    topics, segment titles) and flags episodes where the topic is
                    the main subject or a meaningful segment. (Catalog is prompt-
                    cached, so repeat questions within a few minutes are cheaper.)
  2. Safety net     v2 hybrid search adds any strong hits the scan missed.
  3. De-duplicate   Reruns are folded into their original when both are found.
  4. Read           Haiku reads each candidate's full transcript and extracts every
                    passage that discusses the topic: start time, short verbatim
                    quote, and a one-line note. Quotes are checked against the
                    transcript text ("verified").
  5. Write          Claude Sonnet 5.5 groups the passages into themes and writes the
                    summary, citing passages by ID; citations become links that play
                    the episode audio from that moment.

Output: research_out/<slug>_<timestamp>.md (report + evidence table) and .json.

Usage (needs ANTHROPIC_API_KEY or .streamlit/secrets.toml; Pinecone for step 2):
  python3 research.py "cannibalism"
  python3 research.py "What have Josh and Chuck said about Prohibition?" --max-episodes 40
  python3 research.py "cannibalism" --no-dense     # skip Pinecone safety net
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import chunker_v2
import enrich_episodes as ee
import eval_recall as er  # catalog() + _ids()

SCAN_MODEL = READ_MODEL = "claude-haiku-5-5"
WRITE_MODEL = "claude-sonnet-5-5"
PRICES = {  # USD per million tokens (Oct 2026; verify). Haiku: <=100K-token prompts.
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),   # input, output, cache write, cache read
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.10),
}
OUT_DIR = Path("research_out")
MAX_QUOTE_WORDS = 90


class Usage:
    def __init__(self):
        self.cost, self.calls = 0.0, 0

    def add(self, model, u):
        p = PRICES.get(model, (0, 0, 0, 0))
        self.calls += 1
        self.cost += (u.get("input_tokens", 0) * p[0] + u.get("output_tokens", 0) * p[1]
                      + u.get("cache_creation_input_tokens", 0) * p[2]
                      + u.get("cache_read_input_tokens", 0) * p[3]) / 1e6


USAGE = Usage()


def call(model, system, user, tool=None, max_tokens=4000):
    payload = {"model": model, "max_tokens": max_tokens, "system": system,
               "messages": [{"role": "user", "content": user}]}
    if tool:
        payload["tools"] = [tool]
        payload["tool_choice"] = {"type": "tool", "name": tool["name"]}
    msg = json.loads(ee.post("/messages", payload))
    USAGE.add(model, msg.get("usage", {}))
    if tool:
        b = next((b for b in msg.get("content", []) if b.get("type") == "tool_use"), None)
        data = dict(b["input"]) if b else {}
        for k, v in list(data.items()):
            if isinstance(v, str) and v.strip()[:1] in "[{":
                try:
                    data[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        return data
    return "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")


# ------------------------------------------------------------ 1. catalog scan
SCAN_TOOL = {
    "name": "report_episodes", "description": "Report catalog IDs relevant to the question.",
    "input_schema": {"type": "object", "properties": {
        "main": {"type": "array", "items": {"type": "integer"}},
        "substantial": {"type": "array", "items": {"type": "integer"}}},
        "required": ["main", "substantial"]}}


def scan_catalog(question, cat):
    def scan(chunk, offset):
        block = "\n".join(f"[{offset + i}] {text}" for i, (_, text) in enumerate(chunk))
        data = call(SCAN_MODEL,
                    [{"type": "text", "cache_control": {"type": "ephemeral"}, "text":
                      "You find podcast episodes for a researcher. Each catalog line is "
                      "[ID] date | title | summary | topics | segment titles.\n\n" + block}],
                    f"Research question: {question}\n\nList every catalog ID where this is the main subject "
                    f"('main') or where it gets a meaningful segment or extended discussion ('substantial'). "
                    f"Exclude passing mentions. Check every line.", SCAN_TOOL)
        return {k: [i for i in er._ids(data.get(k)) if offset <= i < offset + len(chunk)]
                for k in ("main", "substantial")}

    chunks = [(cat[i:i + er.CHUNK_EPISODES], i) for i in range(0, len(cat), er.CHUNK_EPISODES)]
    main, sub = set(), set()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(lambda c: scan(*c), chunks):
            main.update(r["main"]); sub.update(r["substantial"])
    return {cat[i][0]: "main" for i in main} | {cat[i][0]: "substantial" for i in sub - main}


# ------------------------------------------------------------- 4. reading
READ_TOOL = {
    "name": "report_passages", "description": "Report every passage that addresses the question.",
    "input_schema": {"type": "object", "properties": {
        "relevant": {"type": "boolean"},
        "passages": {"type": "array", "items": {"type": "object", "properties": {
            "start": {"type": "string", "description": "Timestamp shown on the line where the passage starts, e.g. 00:12:40"},
            "quote": {"type": "string", "description": f"Verbatim excerpt, at most {MAX_QUOTE_WORDS} words, copied exactly"},
            "speaker": {"type": "string", "description": "Josh, Chuck, guest, or unknown"},
            "point": {"type": "string", "description": "One sentence: what this passage says about the question"}},
            "required": ["start", "quote", "point"]}}},
        "required": ["relevant", "passages"]}}


def transcript_with_times(path):
    """Text for the reader: exact per-line times when the transcript has them,
    otherwise ~200-word blocks with estimated times (marked with ~)."""
    meta, turns = chunker_v2.parse_transcript(path)
    timed = len({sec for sec, _ in turns}) > 5
    if timed:
        lines = [f"[{chunker_v2.fmt(sec)}] {t}" for sec, t in turns]
    else:
        lines = [f"[~{c['start']}] {c['text']}" for c in chunker_v2.chunk_episode(path)]
    return meta, timed, "\n".join(lines)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


FILLERS = {"um", "uh", "uhm", "er", "ah", "like", "you", "know", "i", "mean"}


def _verified(quote, norm_text):
    """A quote counts as verified when at least 80% of its 4-word sequences appear
    in the transcript (tolerates dropped "um"/"uh" and small transcription quirks)."""
    words = [w for w in _norm(quote).split() if w not in FILLERS]
    text = " ".join(w for w in norm_text.split() if w not in FILLERS)
    if len(words) < 4:
        return " ".join(words) in text
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return sum(g in text for g in grams) / len(grams) >= 0.8


def read_episode(question, filename):
    path = chunker_v2.TRANSCRIPTS / filename
    meta, timed, text = transcript_with_times(path)
    data = call(READ_MODEL,
                "You extract evidence from a podcast transcript (Stuff You Should Know, hosts Josh and "
                "Chuck). Report EVERY passage that substantively addresses the research question, in "
                "order. Copy quotes exactly from the transcript, including filler words like um and uh (no paraphrase, no stitching distant "
                "lines together), at most " + str(MAX_QUOTE_WORDS) + " words each. Use the timestamp "
                "printed on the line where the passage starts. Skip passing mentions. If nothing "
                "qualifies, set relevant=false with no passages.",
                f"Research question: {question}\n\nEpisode: {meta.get('title')} ({meta.get('date')})\n\n"
                f"Transcript:\n{text}", READ_TOOL, max_tokens=6000)
    norm_text = _norm(text)
    passages = []
    for p in data.get("passages", []) if isinstance(data.get("passages"), list) else []:
        if not isinstance(p, dict) or not p.get("quote"):
            continue
        q = " ".join(p["quote"].split()[:MAX_QUOTE_WORDS])
        start = re.sub(r"[^\d:]", "", p.get("start", "")) or "00:00:00"
        try:
            sec = chunker_v2.to_seconds(start if start.count(":") == 2 else "00:" + start)
        except Exception:
            sec = 0
        passages.append({"start_sec": sec, "approx_time": not timed, "quote": q,
                         "speaker": p.get("speaker", "") if timed else "",
                         "point": p.get("point", ""), "verified": _verified(q, norm_text)})
    return {"filename": filename, "title": meta.get("title", filename), "date": meta.get("date", ""),
            "episode_url": meta.get("episode_url", ""), "audio_url": meta.get("audio_url", ""),
            "timed": timed, "passages": passages}


# ------------------------------------------------------------- 5. writing
def mmss(sec):
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def play_link(ep, p):
    """Audio URL with a media-fragment start time (browsers seek the MP3 there)."""
    base = ep.get("audio_url") or ep.get("episode_url") or ""
    return f"{base}#t={p['start_sec']}" if base else ""


def write_report(question, episodes):
    pid, lines, index = 0, [], {}
    for e_i, ep in enumerate(episodes, 1):
        for p in ep["passages"]:
            pid += 1
            index[f"P{pid}"] = (ep, p)
            approx = "~" if p["approx_time"] else ""
            lines.append(f"P{pid} | E{e_i} \"{ep['title']}\" ({ep['date'][:4]}) @ {approx}{mmss(p['start_sec'])} | "
                         f"{p['point']} | \"{p['quote']}\"")
    if not index:
        return "No passages addressing this question were found in the archive.", index
    body = call(WRITE_MODEL,
                "You write research summaries of the podcast Stuff You Should Know for a general audience. "
                "Use ONLY the evidence passages provided. Cite every claim with passage IDs in square "
                "brackets, e.g. [P3] or [P3][P17]. Quote sparingly and briefly. Note where episodes "
                "disagree or where the hosts' take changed over the years. Never invent episodes or facts. "
                "Do not use em dashes.",
                f"Research question: {question}\n\nEvidence ({len(index)} passages from {len(episodes)} episodes):\n"
                + "\n".join(lines) +
                "\n\nWrite markdown with: '## Overview' (one paragraph), '## Themes' (3 to 7 '### ' themes, "
                "each a short paragraph with citations), and '## Over the years' (a few sentences on how "
                "coverage evolved). No evidence table; it is added separately.", max_tokens=6000)

    def link(m):
        ep, p = index.get(m.group(1), (None, None))
        if not ep:
            return m.group(0)
        approx = "~" if p["approx_time"] else ""
        return f"[[{ep['title']} @ {approx}{mmss(p['start_sec'])}]({play_link(ep, p)})]"
    linked = re.sub(r"\[(P\d+)\]", link, body)
    # Same episode + time cited several times in a row ([A][A][A]) -> one link
    linked = re.sub(r"(\[\[[^\]]+\]\([^)]*\)\])(?:\1)+", r"\1", linked)
    return linked, index


def evidence_table(episodes):
    out = ["## Evidence", "",
           "Times marked ~ are estimated (older transcripts have no timestamps). "
           "Quotes marked † could not be matched word-for-word to the transcript.", ""]
    for ep in episodes:
        if not ep["passages"]:
            continue
        also = f" (also aired as: {'; '.join(ep['also_aired'])})" if ep.get("also_aired") else ""
        out.append(f"### {ep['title']} ({ep['date']}){also}")
        if ep.get("episode_url"):
            out.append(f"[Episode page]({ep['episode_url']})")
        out.append("")
        out.append("| Time | Who | What they discuss | Quote |")
        out.append("|---|---|---|---|")
        for p in ep["passages"]:
            approx = "~" if p["approx_time"] else ""
            t = f"[{approx}{mmss(p['start_sec'])}]({play_link(ep, p)})"
            q = p["quote"].replace("|", "/") + ("" if p["verified"] else " †")
            out.append(f"| {t} | {p['speaker'] or ''} | {p['point'].replace('|', '/')} | \"{q}\" |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------- quick summary
QUICK_MODEL = "claude-haiku-5-5"
# A forced tool call (same pattern as the Deep Research steps) reliably returns the
# summary; a plain text reply from Haiku came back empty.
SUMMARY_TOOL = {
    "name": "write_summary", "description": "Return the summary of the search results.",
    "input_schema": {"type": "object", "properties": {
        "summary": {"type": "string", "description": "The markdown summary with [n] citations"}},
        "required": ["summary"]}}


def quick_summary(query, results, max_episodes=10, chars_per_passage=900):
    """Short cited summary of a Quick Search result list (one fast Haiku call,
    well under a cent). Cites episodes as [n], matching the 1-based order of
    `results`, so the app can number its result list to match. Returns markdown."""
    lines = []
    for n, r in enumerate(results[:max_episodes], 1):
        for p in r.get("passages", []):
            seg = f" [{p['segment_title']}]" if p.get("segment_title") else ""
            text = re.sub(r"\s+", " ", p.get("text", ""))[:chars_per_passage]
            lines.append(f"[{n}] \"{r['title']}\" ({r['date'][:4]}){seg}: {text}")
    if not lines:
        return ""
    try:
        text = call(QUICK_MODEL,
                "You summarize search results from the podcast Stuff You Should Know for a general "
                "audience. Use ONLY the passages provided; never add outside facts. Cite every point "
                "with the episode number in square brackets, e.g. [2] or [2][5]. Older transcripts "
                "have no speaker labels, so do not attribute statements to Josh or Chuck unless the "
                "passage itself makes the speaker clear. Do not use em dashes.",
                f"Search: {query}\n\nPassages (grouped by episode number):\n" + "\n".join(lines) +
                "\n\nWrite markdown: one or two sentences giving the overall picture, then 3 to 6 "
                "bullet points with the key findings or facts, each citing its episodes. If the "
                "passages barely touch the search topic, say so in one sentence instead. "
                "Under 200 words. No headings. Return it with the write_summary tool.",
                SUMMARY_TOOL, max_tokens=2000).get("summary", "")
        text = text.strip() if isinstance(text, str) else ""
    except SystemExit as e:  # ee.post() exits on API errors; surface it instead
        raise RuntimeError(str(e)) from None
    if not text:
        raise RuntimeError("the model returned an empty response")
    return text


# ------------------------------------------------------------------ main
def fold_reruns(cands):
    """Drop a rerun when its original is also a candidate; remember the alias."""
    norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())
    profiles = {f: (chunker_v2.load_profile(f) or {}) for f in cands}
    by_title = {norm(p.get("title")): f for f, p in profiles.items() if p.get("episode_kind") != "rerun"}
    keep, also = [], {}
    for f in cands:
        p = profiles[f]
        orig = by_title.get(norm(p.get("rerun_of")))
        if p.get("episode_kind") == "rerun" and orig and orig != f:
            also.setdefault(orig, []).append(f"{p.get('title')} ({p.get('date')})")
        else:
            keep.append(f)
    return keep, also


def keyword_heavy_episodes(question, min_chunks=2, limit=25):
    """Episodes where the question's key terms appear in several passages, even if
    the episode's profile never mentions the topic (e.g. a cannibalism tangent in an
    episode about bath salts). Reading a transcript costs well under a cent."""
    import sqlite3, search_v2
    if not Path(search_v2.FTS_DB).exists():
        search_v2.build_fts()
    db = sqlite3.connect(search_v2.FTS_DB)
    terms = [t for t in re.findall(r"\w+", question.lower())
             if t not in search_v2.STOP and len(t) > 3 and t not in {"josh", "chuck", "said", "talk", "talked"}]
    counts = {}
    for t in terms:
        # The index uses the Porter stemmer, so "cannibalism" also matches cannibal/cannibals.
        for (fn,) in db.execute("SELECT c.filename FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                                "WHERE chunks_fts MATCH ?", (f'text:"{t}"',)):
            counts[fn] = counts.get(fn, 0) + 1
    ranked = sorted((f for f, n in counts.items() if n >= min_chunks), key=counts.get, reverse=True)
    return ranked[:limit]


def run(question, max_episodes=80, use_dense=True, log=print, hs=None):
    """Run the full pipeline. Pass an existing search_v2.HybridSearch as hs to reuse its
    loaded model and connections (the web app does this)."""
    global USAGE
    USAGE = Usage()  # per-run cost tracking
    t0 = time.time()
    cat = er.catalog()
    log(f"1/5 Scanning {len(cat)} episode profiles...")
    scanned = scan_catalog(question, cat)
    log(f"    catalog scan flagged {len(scanned)} episodes "
        f"({sum(v == 'main' for v in scanned.values())} main)")

    hits = []
    log("2/5 Hybrid-search safety net...")
    try:
        import search_v2
        hs = hs or search_v2.HybridSearch(use_dense=use_dense)
        hits = [r["filename"] for r in hs.search(question, n_episodes=20)]
    except Exception as e:
        log(f"    (skipped: {e})")
    try:
        kw = keyword_heavy_episodes(question)
        hits += [k for k in kw if k not in hits]
    except Exception as e:
        log(f"    (keyword net skipped: {e})")
    extra = [h for h in hits if h not in scanned]
    log(f"    added {len(extra)} episodes the scan missed")

    order = ([f for f, k in scanned.items() if k == "main"] + extra +
             [f for f, k in scanned.items() if k == "substantial"])
    cands, also = fold_reruns(list(dict.fromkeys(order)))
    if len(cands) > max_episodes:
        log(f"    capping {len(cands)} candidates at --max-episodes {max_episodes}")
        cands = cands[:max_episodes]

    log(f"3/5 Reading {len(cands)} transcripts...")
    episodes, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(read_episode, question, f): f for f in cands}
        for fut in cf.as_completed(futs):
            done += 1
            try:
                ep = fut.result()
                ep["also_aired"] = also.get(ep["filename"], [])
                episodes.append(ep)
            except BaseException as e:  # keep going if one episode fails
                log(f"    ! {futs[fut]}: {e}")
            if done % 10 == 0:
                log(f"    {done}/{len(cands)} read")
    episodes = sorted([e for e in episodes if e["passages"]], key=lambda e: e["date"])
    n_pass = sum(len(e["passages"]) for e in episodes)
    log(f"4/5 Found {n_pass} passages in {len(episodes)} episodes "
        f"({sum(not p['verified'] for e in episodes for p in e['passages'])} unverified quotes)")

    log("5/5 Writing summary...")
    report, _ = write_report(question, episodes)
    md = (f"# {question}\n\n*SYSK Deep Research, {dt.date.today():%B %d, %Y}. {len(episodes)} episodes, "
          f"{n_pass} passages. AI-generated from transcripts; verify quotes against the audio.*\n\n"
          f"{report}\n\n{evidence_table(episodes)}")
    elapsed = time.time() - t0
    log(f"Done in {elapsed:.0f}s, {USAGE.calls} API calls, about ${USAGE.cost:.2f}")
    return md, {"question": question, "date": dt.datetime.now().isoformat(timespec="seconds"),
                "cost_usd": round(USAGE.cost, 4), "seconds": round(elapsed), "candidates": cands,
                "episodes": episodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--max-episodes", type=int, default=80)
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    md, data = run(a.question, a.max_episodes, not a.no_dense)
    OUT_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", a.question.lower()).strip("_")[:50]
    stem = OUT_DIR / f"{slug}_{dt.datetime.now():%Y%m%dT%H%M}"
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport: {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
