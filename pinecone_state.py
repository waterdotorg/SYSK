"""
Pinecone as the single source of truth for "what is indexed".

Background (see COWORK.md, Priority #1): the project used to track indexed
episodes in a local ``indexing_progress.json`` ledger that drifted out of sync
with the actual Pinecone state more than once. This module derives indexed-state
directly from Pinecone instead, so there is nothing to drift.

How it works
------------
Every chunk is upserted with a vector ID of the form ``f"{filename}_{chunk_id}"``
where ``filename`` is the transcript file's basename (e.g.
``SYSK_2026-06-11_The_NY_Subway_Vigilante_48min.txt``) and ``chunk_id`` is an
integer. Because transcript filenames always end in ``.txt`` and never contain
the substring ``.txt_``, we can recover the originating filename from any vector
ID by splitting on ``.txt_``. We enumerate all IDs with the paginated
``index.list()`` API, which is cheap (IDs only, no metadata fetch).

This module has no Streamlit dependency and can be imported from the app, the
offline indexer, or the MCP server.
"""

import re
from typing import Iterable, List, Optional, Set

# Vector IDs look like "<basename>.txt_<int>"; this splits off the chunk suffix.
_ID_FILENAME_RE = re.compile(r"^(?P<filename>.+\.txt)_\d+$")

# Transcript filenames embed the publish date: SYSK_YYYY-MM-DD_Title_Nmin.txt
_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def filename_from_vector_id(vector_id: str) -> Optional[str]:
    """Recover the transcript filename from a Pinecone vector ID.

    Returns the ``*.txt`` basename, or ``None`` if the ID does not match the
    expected ``<filename>.txt_<int>`` pattern.
    """
    if not vector_id:
        return None
    match = _ID_FILENAME_RE.match(vector_id)
    if match:
        return match.group("filename")
    return None


def get_indexed_filenames(index, namespace: str = "") -> Set[str]:
    """Return the set of transcript filenames that are present in Pinecone.

    This is the authoritative answer to "what is indexed". It paginates over all
    vector IDs (IDs only — no metadata fetch) and recovers the filename from each
    ID prefix.

    Args:
        index: a Pinecone Index handle (``pc.Index(name)``).
        namespace: Pinecone namespace (default index uses the empty namespace).
    """
    indexed: Set[str] = set()
    for id_batch in index.list(namespace=namespace):
        # index.list() yields lists of IDs (one page at a time).
        for vector_id in id_batch:
            filename = filename_from_vector_id(vector_id)
            if filename:
                indexed.add(filename)
    return indexed


def _sort_key_newest_first(filename: str):
    """Sort key that orders by embedded publish date, newest first.

    Files without a parseable date sort oldest (so genuinely new, well-named
    episodes are always processed first).
    """
    match = _DATE_RE.search(filename)
    date_str = match.group(1) if match else "0000-00-00"
    # Newest first: invert by using the date string descending, then filename.
    return (date_str, filename)


def sort_newest_first(filenames: Iterable[str]) -> List[str]:
    """Return filenames ordered newest-first by their embedded ``YYYY-MM-DD``.

    This is Priority #2: a weekly update should touch the genuinely new episodes
    first instead of walking a positional cursor through the whole backlog.
    """
    return sorted(filenames, key=_sort_key_newest_first, reverse=True)


def diff_to_index(disk_filenames: Iterable[str], indexed_filenames: Iterable[str]) -> List[str]:
    """Compute the set of files that need indexing, newest-first.

    ``needs indexing = files on disk - episodes already in Pinecone``
    """
    to_index = set(disk_filenames) - set(indexed_filenames)
    return sort_newest_first(to_index)
