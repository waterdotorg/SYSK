#!/usr/bin/env python3
"""
SYSK v2, Phase 1: build an "episode profile" for each transcript with Claude Haiku.

Each profile (enrichment/<transcript-stem>.json) holds:
  * summary         - 3 to 5 sentence overview
  * episode_kind    - full_episode | short_stuff | rerun | selects | listener_mail | other
  * rerun_of        - original episode title when this is a rerun/playlist repost, else null
  * topics          - specific, searchable topic phrases (with common synonyms)
  * entities        - people, places, organizations, works, events
  * segments        - timestamped map of the whole episode (start, end, title, summary,
                      topics, kind), with ad breaks flagged so search can skip them

These profiles power Deep Research: Claude scans every profile to find all episodes
on a topic, then reads only the matching segments.

Usage
-----
  # Sync mode (small tests): profile specific files, or the N newest without a profile
  python3 enrich_episodes.py --files transcripts/A.txt transcripts/B.txt
  python3 enrich_episodes.py --limit 10

  # Batch mode (whole archive, 50% cheaper, results within ~24h, usually much sooner)
  python3 enrich_episodes.py --batch-submit            # writes enrichment/_batch.json
  python3 enrich_episodes.py --batch-collect           # downloads finished results

The API key comes from ANTHROPIC_API_KEY or .streamlit/secrets.toml.
Only the Python standard library is required.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = os.getenv("SYSK_ENRICH_MODEL", "claude-haiku-5-5")
API = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
TRANSCRIPTS = Path("transcripts")
OUT_DIR = Path("enrichment")
BATCH_STATE = OUT_DIR / "_batch.json"
PROFILE_VERSION = 2

# USD per million tokens (Haiku 5.5, prompts <=100K tokens, Oct 2026). Verify before budgeting.
PRICE_IN, PRICE_OUT = 0.10, 0.50

SYSTEM = """You index episodes of the podcast "Stuff You Should Know" (hosts Josh Clark and Chuck Bryant) so listeners can later find every place a topic is discussed.

You receive one transcript. Lines look like "[HH:MM:SS] Speaker: text". Call the save_profile tool exactly once.

Rules:
- segments must cover the whole episode in order, without gaps, from the first to the last timestamp. Use real timestamps from the transcript for start and end.
- Make a new segment whenever the subject changes; typical content segments run 2 to 6 minutes.
- Mark sponsor reads and ad breaks as kind "ad" (summary may be one short line). Mark the opening banter "intro", listener mail "listener_mail", sign-off "outro"; everything else "content".
- topics: specific and searchable (e.g. "pirate radio", "FCC enforcement", "1987 Chicago TV hijacking"), including the common synonyms someone might search for. Episode-level topics: 10 to 30. Segment topics: 2 to 8.
- If the hosts say this is a rerun, a "Selects" episode, or part of a playlist of past episodes, set episode_kind to "rerun" and rerun_of to the original episode title if stated or obvious.
- Summaries must be factual and only describe what is actually said in the episode. Keep distinct facts distinct (do not merge two teams, people or events into one).
- entities: the episode's subjects only. Do not list the hosts (Josh, Chuck), producers (Jerry, Dave), or the show's companies (iHeartRadio, HowStuffWorks).
- Spell names exactly as the hosts spell or settle on them in the transcript; when the transcript gives two spellings, use the one the hosts repeat or correct to."""

TOOL = {
    "name": "save_profile",
    "description": "Save the structured profile of this episode.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "episode_kind": {"type": "string", "enum": [
                "full_episode", "short_stuff", "rerun", "selects", "listener_mail", "other"]},
            "rerun_of": {"type": ["string", "null"]},
            "topics": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": [
                        "person", "place", "organization", "work", "event", "other"]},
                },
                "required": ["name", "type"],
            }},
            "segments": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "HH:MM:SS"},
                    "end": {"type": "string", "description": "HH:MM:SS"},
                    "kind": {"type": "string", "enum": [
                        "intro", "content", "ad", "listener_mail", "outro"]},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end", "kind", "title", "summary", "topics"],
            }},
        },
        "required": ["summary", "episode_kind", "rerun_of", "topics", "entities", "segments"],
    },
}

HEADER_KEYS = {"Episode": "title", "Date": "date", "Duration": "duration",
               "Episode URL": "episode_url", "Audio URL": "audio_url",
               "Transcript URL": "transcript_url"}
TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        secrets = Path(".streamlit/secrets.toml")
        if secrets.exists():
            m = re.search(r'^ANTHROPIC_API_KEY\s*=\s*"([^"]+)"', secrets.read_text(), re.M)
            key = m.group(1) if m else ""
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found (env or .streamlit/secrets.toml).")
    return key


def parse_transcript(path: Path):
    """Return (metadata dict, compact '[HH:MM:SS] Speaker: text' transcript)."""
    raw = path.read_text(encoding="utf-8")
    head, _, body = raw.partition("=" * 80)
    meta = {"filename": path.name}
    for line in head.splitlines():
        k, sep, v = line.partition(": ")
        if sep and k in HEADER_KEYS:
            meta[HEADER_KEYS[k]] = v.strip()
    out, ts = [], None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if TS_RE.match(line):
            ts = line
        else:
            out.append(f"[{ts}] {line}" if ts else line)
            ts = None
    return meta, "\n".join(out)


def build_request(meta: dict, text: str) -> dict:
    user = (f"Title: {meta.get('title', '')}\nPublished: {meta.get('date', '')}\n"
            f"Length: {meta.get('duration', '')}\n\nTranscript:\n{text}")
    return {
        "model": MODEL,
        "max_tokens": 8000,
        "system": SYSTEM,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "save_profile"},
        "messages": [{"role": "user", "content": user}],
    }


def post(path: str, payload=None, method="POST"):
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"x-api-key": api_key(), "anthropic-version": API_VERSION,
                 "content-type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            sys.exit(f"API error {e.code}: {body[:500]}")


def save_profile(meta: dict, message: dict) -> dict:
    block = next((b for b in message.get("content", []) if b.get("type") == "tool_use"), None)
    if not block:
        raise ValueError(f"no tool_use block (stop_reason={message.get('stop_reason')})")
    usage = message.get("usage", {})
    data = dict(block["input"])
    for k in ("segments", "topics", "entities"):  # model occasionally returns a list as a JSON string
        if isinstance(data.get(k), str):
            try:
                data[k] = json.loads(data[k])
            except json.JSONDecodeError:
                raise ValueError(f"field {k} is not valid JSON")
    missing = [k for k in TOOL["input_schema"]["required"] if k not in data]
    if missing or not isinstance(data.get("segments"), list) or not data["segments"]:
        raise ValueError(f"incomplete profile (missing {missing or 'segments'})")
    profile = {"profile_version": PROFILE_VERSION, "model": message.get("model", MODEL),
               **meta, **data,
               "usage": {"input_tokens": usage.get("input_tokens", 0),
                         "output_tokens": usage.get("output_tokens", 0)}}
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / (Path(meta["filename"]).stem + ".json")
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def pending_files(limit: int):
    done = {p.stem for p in OUT_DIR.glob("*.json")} if OUT_DIR.exists() else set()
    files = sorted(TRANSCRIPTS.glob("*.txt"), key=lambda p: p.name, reverse=True)
    todo = [p for p in files if p.stem not in done]
    return todo[:limit] if limit else todo


def cost(i, o, batch=False):
    c = (i * PRICE_IN + o * PRICE_OUT) / 1e6
    return c / 2 if batch else c


def run_sync(files):
    tin = tout = 0
    for n, path in enumerate(files, 1):
        meta, text = parse_transcript(path)
        msg = json.loads(post("/messages", build_request(meta, text)))
        try:
            p = save_profile(meta, msg)
        except ValueError as e:
            print(f"[{n}/{len(files)}] {path.name}: FAILED ({e})")
            continue
        u = p["usage"]
        tin += u["input_tokens"]; tout += u["output_tokens"]
        print(f"[{n}/{len(files)}] {path.name}: {len(p['segments'])} segments, "
              f"{len(p['topics'])} topics, kind={p['episode_kind']}, "
              f"{u['input_tokens']:,} in / {u['output_tokens']:,} out")
    print(f"\nTotal: {tin:,} input + {tout:,} output tokens, about ${cost(tin, tout):.4f}")


BATCH_CHUNK = 400  # episodes per batch request (keeps each upload well under API size limits)


def batch_submit(files):
    """Submit all files as one or more batches; state is appended to enrichment/_batch.json."""
    OUT_DIR.mkdir(exist_ok=True)
    state = json.loads(BATCH_STATE.read_text(encoding="utf-8")) if BATCH_STATE.exists() else {"batches": []}
    queued = {m["filename"] for bt in state["batches"] for m in bt["episodes"].values()}
    files = [f for f in files if f.name not in queued]
    if not files:
        print("Everything is already queued; run --batch-collect.")
        return
    for start in range(0, len(files), BATCH_CHUNK):
        chunk, reqs, ids = files[start:start + BATCH_CHUNK], [], {}
        for i, path in enumerate(chunk):
            meta, text = parse_transcript(path)
            cid = f"ep{start + i:05d}"
            ids[cid] = meta
            reqs.append({"custom_id": cid, "params": build_request(meta, text)})
        resp = json.loads(post("/messages/batches", {"requests": reqs}))
        state["batches"].append({"batch_id": resp["id"], "episodes": ids})
        BATCH_STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        print(f"Submitted batch {resp['id']} ({len(reqs)} episodes).")
    print("Run --batch-collect later (safe to re-run until everything is saved).")


def batch_collect():
    if not BATCH_STATE.exists():
        print("No batches pending.")
        return
    state = json.loads(BATCH_STATE.read_text(encoding="utf-8"))
    remaining, ok, bad, tin, tout = [], 0, 0, 0, 0
    for bt in state["batches"]:
        info = json.loads(post(f"/messages/batches/{bt['batch_id']}", method="GET"))
        if info.get("processing_status") != "ended":
            print(f"{bt['batch_id']}: {info.get('processing_status')} {info.get('request_counts')}")
            remaining.append(bt)
            continue
        results = post(f"/messages/batches/{bt['batch_id']}/results", method="GET")
        for line in results.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            meta = bt["episodes"][r["custom_id"]]
            if r["result"]["type"] != "succeeded":
                bad += 1
                print(f"FAILED {meta['filename']}: {r['result']['type']}")
                continue
            try:
                p = save_profile(meta, r["result"]["message"])
                ok += 1
                tin += p["usage"]["input_tokens"]; tout += p["usage"]["output_tokens"]
            except ValueError as e:
                bad += 1
                print(f"FAILED {meta['filename']}: {e}")
    state["batches"] = remaining
    if remaining:
        BATCH_STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    else:
        BATCH_STATE.unlink()
    print(f"Saved {ok} profiles, {bad} failed, {len(remaining)} batch(es) still running. "
          f"~${cost(tin, tout, batch=True):.2f} at batch rates for this collection.")
    if bad:
        print("Failed episodes have no profile yet; re-run --batch-submit to retry them.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", type=Path, help="Specific transcript files to profile.")
    ap.add_argument("--limit", type=int, default=0, help="Max episodes (newest first) without a profile.")
    ap.add_argument("--batch-submit", action="store_true")
    ap.add_argument("--batch-collect", action="store_true")
    a = ap.parse_args()
    if a.batch_collect:
        return batch_collect()
    files = a.files or pending_files(a.limit)
    if not files:
        print("Every transcript already has a profile.")
        return
    if a.batch_submit:
        return batch_submit(files)
    run_sync(files)


if __name__ == "__main__":
    main()
