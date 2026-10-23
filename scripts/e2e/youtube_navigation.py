"""The extension's toolbar badge across YouTube-style navigation, in real Chromium.

    python scripts/e2e/youtube_navigation.py --chromium /path/to/chrome

YouTube itself is NOT contacted. Requests to https://www.youtube.com/ are
answered inside the browser by stand-in pages (Playwright routing), and videos change the way
YouTube changes them: `history.pushState`, back/forward and reloads, without a page load. This
tests the extension's video identification and badge logic, not YouTube's markup or content
acquisition. A stub backend on 127.0.0.1 returns scores with controllable delays and failures.

Needs a built extension (`cd extension && npm run build`) and Playwright for Python.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
A, B, C, S = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC", "SSSSSSSSSSS"
D, E = "DDDDDDDDDDD", "EEEEEEEEEEE"  # first looked up mid-test
results = {"PASS": 0, "FAIL": 0}


def step(name: str, ok: bool, detail: str = "") -> None:
    results["PASS" if ok else "FAIL"] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


class Backend:
    def __init__(self) -> None:
        self.scores: dict[str, float] = {A: 4.6, B: 1.4, S: 2.2, D: 4.9}
        self.delay: dict[str, float] = {}
        self.status = 200
        backend = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                vid = self.path.rsplit("/", 1)[-1]
                time.sleep(backend.delay.get(vid, 0.0))
                if backend.status != 200:
                    self.send_response(backend.status)
                    self.end_headers()
                    return
                if vid not in backend.scores:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({"intrinsic_quality": {"overall": backend.scores[vid], "synthetic": False}})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *a: Any) -> None:
                pass

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


PAGE = "<!doctype html><title>stand-in</title><p>Stand-in YouTube page (no YouTube content).</p>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", required=True)
    ap.add_argument("--extension", default=str(ROOT / "extension" / "dist"))
    args = ap.parse_args()
    backend = Backend()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            "",
            executable_path=args.chromium,
            headless=True,
            args=[
                f"--disable-extensions-except={args.extension}",
                f"--load-extension={args.extension}",
                "--headless=new",
            ],
        )
        sw = (
            ctx.service_workers[0]
            if ctx.service_workers
            else ctx.wait_for_event("serviceworker", timeout=15000)
        )
        # Point the extension at the stub backend, from an extension page (storage API is ready there).
        setup = ctx.new_page()
        setup.goto(f"chrome-extension://{sw.url.split('/')[2]}/popup.html")
        setup.evaluate(f"chrome.storage.local.set({{'avd.apiBase': 'http://127.0.0.1:{backend.port}'}})")
        setup.close()
        ctx.route(
            "https://www.youtube.com/**", lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE)
        )
        # Not covered: youtu.be links (YouTube redirects them to /watch) and m./music. hosts. The
        # extension has host access to www.youtube.com only, so it never sees those URLs itself.

        def badge(page: Any, settle: float = 0.6) -> dict[str, str]:
            time.sleep(settle)
            return sw.evaluate(
                """async (u) => { const tabs = await chrome.tabs.query({});
                   const t = tabs.find((x) => x.url === u) ?? tabs.find((x) => (x.pendingUrl ?? x.url) === u);
                   return {text: await chrome.action.getBadgeText({tabId: t.id}),
                           title: await chrome.action.getTitle({tabId: t.id})}; }""",
                page.url,
            )

        def push(page: Any, path: str) -> None:
            page.evaluate(f"history.pushState({{}}, '', {json.dumps(path)})")

        page = ctx.new_page()
        page.goto(f"https://www.youtube.com/watch?v={A}")
        step("direct open of a watch page", badge(page)["text"] == "4.6")

        push(page, f"/watch?v={B}")
        step("in-page navigation to another video", badge(page)["text"] == "1.4")
        page.go_back()
        step("browser back", badge(page)["text"] == "4.6")
        page.go_forward()
        step("browser forward", badge(page)["text"] == "1.4")

        backend.delay[D] = 2.0  # D's lookup is slow; the user moves on before it answers
        push(page, f"/watch?v={D}")
        b = badge(page, 0.3)
        step(
            "while a lookup runs, the previous video's score is not shown",
            b["text"] == "" and "checking" in b["title"],
            json.dumps(b),
        )
        push(page, f"/watch?v={B}")
        b = badge(page, 2.5)
        step(
            "a late answer for the previous video never replaces the current one",
            b["text"] == "1.4",
            json.dumps(b),
        )

        page.reload()
        step("reload keeps the right score", badge(page)["text"] == "1.4")

        push(page, f"/watch?v={B}&list=PL0123456789&index=2")
        step("playlist URL resolves to its video", badge(page)["text"] == "1.4")
        push(page, f"/shorts/{S}")
        step("Shorts URL resolves to its video", badge(page)["text"] == "2.2")
        push(page, f"/live/{A}")
        step("live URL resolves to its video", badge(page)["text"] == "4.6")

        push(page, f"/watch?v={C}")
        b = badge(page)
        step(
            "a video not in the catalog shows nothing, says so",
            b["text"] == "" and "not analyzed" in b["title"],
            b["title"],
        )
        push(page, "/feed/subscriptions")
        b = badge(page)
        step(
            "a page that is not a video resets badge and title",
            b == {"text": "", "title": "AI Video Discovery"},
            json.dumps(b),
        )

        backend.status = 503
        push(page, f"/watch?v={E}")
        b = badge(page)
        step("a failing backend is shown as an error, not as 'not analyzed'", b["text"] == "!", b["title"])
        backend.status = 200
        backend.scores[E] = 2.3
        push(page, "/")
        push(page, f"/watch?v={E}")
        step(
            "the badge recovers as soon as the backend does (failures are not cached)",
            badge(page)["text"] == "2.3",
        )

        second = ctx.new_page()
        second.goto(f"https://www.youtube.com/watch?v={B}")
        step("a second tab gets its own score", badge(second)["text"] == "1.4")
        page.bring_to_front()
        step("switching back keeps the first tab's score", badge(page)["text"] == "2.3")
        second.goto(f"https://www.youtube.com/watch?v={A}")
        step("a second tab navigating does not change the first tab", badge(page)["text"] == "2.3")
        second.close()
        ctx.close()
    backend.server.shutdown()
    print(json.dumps(results))
    sys.exit(1 if results["FAIL"] else 0)


if __name__ == "__main__":
    main()
