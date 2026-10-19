#!/usr/bin/env python3
"""
Remove "SYSK Selects" re-runs whose original episode is already in the archive.
Selects that are the ONLY copy of an episode (the original has dropped out of the
public feed) are kept. Removal covers everywhere an episode lives:

  1. Pinecone vectors in sysk-v2 (new app) and sysk-transcripts (old app / MCP server)
  2. transcripts/<file>.txt and enrichment/<file>.json (via `git rm`, so the
     removal is committed and Streamlit Cloud drops them on redeploy)
  3. research_cache/ answers that cited a Selects episode
  4. the local keyword index (sysk_fts.db), rebuilt without them

The downloader uses the same test (original_in_archive), so removed Selects
won't come back on the weekly update.

Usage (from the repo folder, venv active, PINECONE_API_KEY in env or secrets.toml):
  python3 remove_selects.py               # dry run: what would be removed / kept
  python3 remove_selects.py --apply       # do it
  python3 remove_selects.py --apply --skip-pinecone   # files/cache/FTS only
"""
import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path

import chunker_v2
from sysk_rss_transcript_downloader import archive_title_keys, is_selects, original_in_archive

PINECONE_INDEXES = ("sysk-v2", "sysk-transcripts")


def selects_files():
    """Returns (duplicates to remove, only-copy Selects to keep), as (path, title) lists."""
    keys = archive_title_keys(chunker_v2.TRANSCRIPTS)
    remove, keep = [], []
    for path in sorted(chunker_v2.TRANSCRIPTS.glob("*.txt")):
        meta, _ = chunker_v2.parse_transcript(path)
        title = meta.get("title", "")
        if not is_selects(title):
            continue
        rerun_of = (chunker_v2.load_profile(path.name) or {}).get("rerun_of")
        (remove if original_in_archive(title, keys, rerun_of) else keep).append((path, title))
    return remove, keep


def purge_pinecone(files, apply):
    import index_v2, pinecone_state
    from pinecone import Pinecone
    pc = Pinecone(api_key=index_v2.api_key())
    existing = {i.name for i in pc.list_indexes()}
    for name in PINECONE_INDEXES:
        if name not in existing:
            print(f"  {name}: not found, skipped")
            continue
        index = pc.Index(name)
        total = 0
        for path in files:
            # Both ID schemes start with the url-quoted filename:
            #   v2 "<quoted>#0001", v1 "<quoted>_3". Filenames end in .txt, so no false prefixes.
            ids = []
            for variant in {path.name, pinecone_state._nfc(path.name)}:
                ids += list(index_v2.indexed_ids(index, prefix=urllib.parse.quote(variant, safe="")))
            ids = sorted(set(ids))
            total += len(ids)
            if apply:
                for i in range(0, len(ids), 1000):
                    index.delete(ids=ids[i:i + 1000])
        print(f"  {name}: {total:,} vectors {'deleted' if apply else 'would be deleted'}")


def git_rm(paths, apply):
    tracked = set(subprocess.run(["git", "ls-files", "--", *map(str, paths)], capture_output=True,
                                 text=True, check=True).stdout.split("\n")) - {""}
    untracked = [p for p in paths if str(p) not in tracked]
    print(f"  {len(tracked)} tracked files to git rm, {len(untracked)} untracked to delete")
    if apply:
        tl = sorted(tracked)
        for i in range(0, len(tl), 200):
            subprocess.run(["git", "rm", "-q", "--", *tl[i:i + 200]], check=True)
        for p in untracked:
            p.unlink()


def purge_cache(files, apply):
    names = {p.name for p in files}
    hit = []
    for f in Path("research_cache").glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        cited = {e.get("filename") for e in d.get("episodes", [])} | set(d.get("candidates", []))
        if cited & names:
            hit.append(f)
    print(f"  {len(hit)} cached Deep Research answers cite a Selects episode")
    if apply:
        for f in hit:
            f.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually remove (default is a dry run).")
    ap.add_argument("--skip-pinecone", action="store_true")
    ap.add_argument("--list", action="store_true", help="List every episode being removed.")
    a = ap.parse_args()

    remove, keep = selects_files()
    print(f"Keeping {len(keep)} Selects that are the only copy of their episode:")
    for _, t in keep:
        print(f"  = {t}")
    files = [p for p, _ in remove]
    profiles = [chunker_v2.ENRICHMENT / (p.stem + ".json") for p in files]
    profiles = [p for p in profiles if p.exists()]
    print(f"\nRemoving {len(files)} Selects whose original is already here ({len(profiles)} profiles):")
    for _, t in remove[:8]:
        print(f"  - {t}")
    if len(remove) > 8:
        print(f"  ... and {len(remove) - 8} more (--list shows all)")
    if a.list:
        for _, t in remove[8:]:
            print(f"  - {t}")
    if not files:
        return

    print("\n1. Pinecone")
    if a.skip_pinecone:
        print("  skipped (--skip-pinecone)")
    else:
        purge_pinecone(files, a.apply)
    print("\n2. Repo files")
    git_rm(files + profiles, a.apply)
    print("\n3. Research cache")
    purge_cache(files, a.apply)
    print("\n4. Keyword index")
    if a.apply:
        import search_v2
        print("  rebuilt: %d chunks, %d episode profiles" % search_v2.build_fts())
    else:
        print("  would rebuild sysk_fts.db")
    print("\nDone." if a.apply else "\nDry run only. Re-run with --apply to remove.")


if __name__ == "__main__":
    main()
