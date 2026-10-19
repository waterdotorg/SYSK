"""
SYSK Archive Explorer (v2) - Streamlit app

  Quick Search    hybrid search (meaning + keywords + episode profiles), results
                  grouped by episode, each passage playable from its timestamp, with a
                  short AI summary of the top 10 episodes (Claude Haiku, < $0.01).
  Deep Research   finds every episode on a topic, extracts the dialogue, and writes
                  a cited summary (research.py). ~1 minute, ~$0.15-0.50 per question.

Access: invite codes with a daily Deep Research allowance (Streamlit secrets).

Secrets (Streamlit Cloud > Settings > Secrets, or .streamlit/secrets.toml):
  ANTHROPIC_API_KEY = "sk-ant-api03-..."      # workspace-scoped key
  PINECONE_API_KEY  = "pcsk_..."
  DEEP_RESEARCH_GLOBAL_DAILY = 40             # all users combined, per day
  ADMIN_CODES = ["your-admin-code"]           # unlimited, can force fresh runs
  [invite_codes]                              # code = Deep Research runs per day
  "friends-2026" = 5

Run locally:  streamlit run app_v2.py
"""
import datetime as dt
import hashlib
import json
import os
import re
import threading
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SYSK Archive Explorer", page_icon="🎙️", layout="wide")

# Expose Streamlit secrets as env vars for the pipeline modules (they read env first).
for _k in ("ANTHROPIC_API_KEY", "PINECONE_API_KEY"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass

import research   # noqa: E402  (needs env vars set first)
import search_v2  # noqa: E402

CACHE_DIR = Path("research_cache")
CACHE_DAYS = 30


def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the archive (first visit after a restart takes ~30 s)...")
def get_search():
    return search_v2.HybridSearch(use_dense=True)


@st.cache_resource
def usage_counters():
    """In-memory daily counters, shared by all sessions. They reset when the app
    restarts; the Anthropic workspace spend limit is the hard backstop."""
    return {"lock": threading.Lock(), "day": None, "per_code": {}, "total": 0}


def check_and_count(code, is_admin):
    """Returns (allowed, message). Counts the run if allowed."""
    c = usage_counters()
    today = dt.date.today().isoformat()
    with c["lock"]:
        if c["day"] != today:
            c.update(day=today, per_code={}, total=0)
        if is_admin:
            c["total"] += 1
            return True, ""
        limit = int((secret("invite_codes", {}) or {}).get(code, 0))
        used = c["per_code"].get(code, 0)
        if used >= limit:
            return False, f"You've used today's {limit} Deep Research runs. They reset at midnight UTC."
        if c["total"] >= int(secret("DEEP_RESEARCH_GLOBAL_DAILY", 40)):
            return False, "The archive has hit today's overall Deep Research limit. Please try again tomorrow."
        c["per_code"][code] = used + 1
        c["total"] += 1
        return True, ""


def remaining_today(code, is_admin):
    if is_admin:
        return "unlimited"
    c = usage_counters()
    limit = int((secret("invite_codes", {}) or {}).get(code, 0))
    used = c["per_code"].get(code, 0) if c["day"] == dt.date.today().isoformat() else 0
    return f"{max(limit - used, 0)} of {limit}"


# ------------------------------------------------------------------ helpers
def mmss(sec, approx=False):
    return ("~" if approx else "") + research.mmss(int(sec))


def play_button(label, url, sec, key):
    if url and st.button(f"▶ {label}", key=key, help="Play from here"):
        st.session_state.player = {"url": url, "sec": int(sec), "label": label}


def player():
    p = st.session_state.get("player")
    if p:
        st.caption(f"Now playing: {p['label']}")
        st.audio(p["url"], start_time=p["sec"])


def cache_key(question):
    return hashlib.sha1(re.sub(r"\s+", " ", question.strip().lower()).encode()).hexdigest()[:16]


def load_cached(question):
    f = CACHE_DIR / f"{cache_key(question)}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        if dt.datetime.fromisoformat(d["date"]) > dt.datetime.now() - dt.timedelta(days=CACHE_DAYS):
            return d
    return None


def save_cached(question, md, data):
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{cache_key(question)}.json").write_text(
        json.dumps({**data, "markdown": md}, ensure_ascii=False), encoding="utf-8")


# -------------------------------------------------------------------- gate
def login():
    codes = secret("invite_codes", {}) or {}
    admins = secret("ADMIN_CODES", []) or []
    if st.session_state.get("code"):
        return st.session_state.code, st.session_state.code in admins
    st.title("🎙️ SYSK Archive Explorer")
    st.write("Search 2,400 episodes of *Stuff You Should Know* and research any topic across "
             "18 years of the show. Access is by invite code.")
    code = st.text_input("Invite code", type="password")
    if st.button("Enter"):
        if code in codes or code in admins:
            st.session_state.code = code
            st.rerun()
        st.error("That code isn't valid.")
    st.stop()


# --------------------------------------------------------------------- tabs
def quick_search_tab():
    q = st.text_input("Search the archive", placeholder="e.g. pirate radio, the Donner Party, why we yawn")
    n = st.slider("Episodes to show", 5, 50, 15)
    if q:
        with st.spinner("Searching..."):
            results = get_search().search(q, n_episodes=n, passages_per_episode=3)
        # Summary sits above the results but is filled in last, so results show at once.
        summary_box = st.container()
        st.caption(f"{len(results)} episodes, best matches first. ▶ plays the episode from that moment.")
        for i, r in enumerate(results):
            badge = " · rerun" if r["episode_kind"] == "rerun" else ""
            with st.expander(f"**{i + 1}. {r['title']}** ({r['date']}){badge}", expanded=i < 3):
                if r.get("episode_url"):
                    st.markdown(f"[Episode page]({r['episode_url']})")
                approx = not _is_timed(r["filename"])
                for j, p in enumerate(r["passages"]):
                    cols = st.columns([1, 6])
                    with cols[0]:
                        play_button(mmss(p["start_sec"], approx), _audio_url(r["filename"]),
                                    p["start_sec"], key=f"qs{i}_{j}")
                    with cols[1]:
                        if p.get("segment_title"):
                            st.markdown(f"**{p['segment_title']}**")
                        st.write(p["text"][:600] + ("..." if len(p["text"]) > 600 else ""))
        if results:
            with summary_box:
                st.markdown("#### Summary")
                with st.spinner("Summarizing the top episodes..."):
                    try:
                        summary = _quick_summary(q, results[:10])
                    except Exception as e:  # never let the summary break search
                        summary = None
                        st.caption(f"Summary unavailable right now: {e}")
                if summary:
                    st.markdown(summary)
                    st.caption(f"AI-generated from the top {min(len(results), 10)} episodes below; "
                               "[n] refers to the numbered results. For a full, every-episode answer, "
                               "use Deep Research.")
                st.divider()


@st.cache_data(ttl=7 * 24 * 3600, max_entries=2000, show_spinner=False)
def _quick_summary(query, results):
    """Cached per query + result set, so ▶ clicks and reruns don't call the API again."""
    return research.quick_summary(query, results)


@st.cache_data(max_entries=5000)
def _audio_url(filename):
    meta, _ = research.chunker_v2.parse_transcript(research.chunker_v2.TRANSCRIPTS / filename)
    return meta.get("audio_url", "")


@st.cache_data(max_entries=5000)
def _is_timed(filename):
    _, turns = research.chunker_v2.parse_transcript(research.chunker_v2.TRANSCRIPTS / filename)
    return len({s for s, _ in turns}) > 5


def deep_research_tab(code, is_admin):
    st.write("Ask about any topic. Deep Research reads every episode that discusses it and "
             "writes a cited summary with the actual dialogue. Takes about a minute.")
    q = st.text_input("Research question", placeholder="e.g. What have Josh and Chuck said about cannibalism?",
                      key="dr_q")
    st.caption(f"Deep Research runs left today: {remaining_today(code, is_admin)}")
    fresh = st.checkbox("Ignore cached answer and run fresh", value=False) if is_admin else False
    if st.button("Run Deep Research", type="primary", disabled=not q):
        cached = None if fresh else load_cached(q)
        if cached:
            st.session_state.dr = cached
            st.info(f"Showing a saved answer from {cached['date'][:10]} (no charge to your daily runs).")
        else:
            ok, msg = check_and_count(code, is_admin)
            if not ok:
                st.warning(msg)
                return
            with st.status("Researching...", expanded=True) as status:
                md, data = research.run(q, log=status.write, hs=get_search())
                status.update(label=f"Done: {len(data['episodes'])} episodes, "
                                    f"{sum(len(e['passages']) for e in data['episodes'])} passages", state="complete")
            save_cached(q, md, data)
            st.session_state.dr = {**data, "markdown": md}
    d = st.session_state.get("dr")
    if d:
        render_report(d)


def render_report(d):
    md = d["markdown"]
    report = md.split("\n## Evidence", 1)[0]
    st.markdown(report)
    st.download_button("Download report (Markdown)", md, file_name="sysk_research.md")
    st.subheader("Evidence")
    st.caption("Times marked ~ are estimated (older episodes have no timestamps). "
               "† = quote could not be matched word-for-word to the transcript.")
    for i, ep in enumerate(d["episodes"]):
        also = f" · also aired as {'; '.join(ep['also_aired'])}" if ep.get("also_aired") else ""
        with st.expander(f"**{ep['title']}** ({ep['date']}) · {len(ep['passages'])} passages{also}"):
            if ep.get("episode_url"):
                st.markdown(f"[Episode page]({ep['episode_url']})")
            for j, p in enumerate(ep["passages"]):
                cols = st.columns([1, 6])
                with cols[0]:
                    play_button(mmss(p["start_sec"], p["approx_time"]), ep.get("audio_url"),
                                p["start_sec"], key=f"dr{i}_{j}")
                with cols[1]:
                    who = f"**{p['speaker']}:** " if p.get("speaker") else ""
                    st.markdown(f"{p['point']}  \n> {who}“{p['quote']}”{'' if p['verified'] else ' †'}")


def about_tab():
    st.markdown("""
**What's in here:** transcripts of every *Stuff You Should Know* episode in the public feed
(2008 to today), refreshed weekly. "SYSK Selects" re-runs are left out when the original
episode is already here (about 30 whose originals left the feed are kept). Episodes before mid-2023 have no timestamps or speaker
names in their transcripts, so their play positions are estimated (marked ~).

**Quick Search summaries:** Claude Haiku reads the passages shown for the top 10 results
and writes a short overview citing them by number. It only sees those passages, so use
Deep Research when you want everything the show has said on a topic.

**How Deep Research works:** Claude Haiku reads a profile of every episode to find all that
touch your topic, then reads each of those transcripts and pulls out the relevant dialogue.
Claude Sonnet writes the summary from those passages only, citing each one.

**Please note:** summaries are AI-generated. Check quotes against the audio before relying on
them. Audio and transcripts belong to iHeartMedia; this tool links to the official episodes.
""")


def main():
    code, is_admin = login()
    with st.sidebar:
        st.markdown("### 🎙️ SYSK Archive Explorer")
        st.caption(f"Deep Research today: {remaining_today(code, is_admin)}")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
    t1, t2, t3 = st.tabs(["Quick Search", "Deep Research", "About"])
    with t1:
        quick_search_tab()
    with t2:
        deep_research_tab(code, is_admin)
    with t3:
        about_tab()
    # Rendered last so a ▶ click in this run shows immediately; the sidebar stays
    # visible while scrolling long results.
    with st.sidebar:
        player()


main()
