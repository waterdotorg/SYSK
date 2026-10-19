#!/usr/bin/env python3
"""
Keep the Streamlit Community Cloud app awake by visiting it in a real (headless)
browser. A plain curl only fetches the outer page; a browser session opens the
app's live connection, which Streamlit counts as traffic. If the app is already
asleep, this clicks "Yes, get this app back up!" and waits for it to load.

Exit code 0 = app loaded; 1 = app never came up (GitHub emails you on failure).

Run locally:  pip install playwright && playwright install chromium
              SYSK_APP_URL=https://<your-app>.streamlit.app python3 keep_alive.py
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.getenv("SYSK_APP_URL", "").strip()
READY_TEXT = re.compile(r"SYSK Archive Explorer|Invite code", re.I)
WAKE_BUTTON = re.compile(r"get this app back up", re.I)
LOAD_TIMEOUT_S = 300  # a cold start (install-free) can take a few minutes


def app_ready(page):
    for frame in page.frames:  # the app renders inside an iframe on *.streamlit.app
        try:
            if frame.get_by_text(READY_TEXT).first.is_visible(timeout=1000):
                return True
        except Exception:
            pass
    return False


def main():
    if not APP_URL:
        print("SYSK_APP_URL is not set yet; nothing to do (set it after deploying the app).")
        return 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Visiting {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)

        deadline = time.time() + LOAD_TIMEOUT_S
        woke = False
        while time.time() < deadline:
            if app_ready(page):
                print("App is up" + (" (woken from sleep)." if woke else "."))
                time.sleep(20)  # stay connected briefly so the visit registers as traffic
                browser.close()
                return 0
            btn = page.get_by_role("button", name=WAKE_BUTTON)
            if not woke and btn.count() and btn.first.is_visible():
                print("App was asleep; clicking the wake-up button...")
                btn.first.click()
                woke = True
            time.sleep(5)

        print(f"App did not finish loading within {LOAD_TIMEOUT_S} s.")
        page.screenshot(path="keep_alive_failure.png", full_page=True)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
