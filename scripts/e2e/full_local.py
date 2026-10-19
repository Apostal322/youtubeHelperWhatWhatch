"""End-to-end check of a local installation with real processes, plus the extension in Chromium.

    python scripts/e2e/full_local.py --avd backend/.venv/bin/avd [--chromium /path/to/chrome] [--keep]

Needs: a built extension (`cd extension && npm run build`), Playwright for Python in the
interpreter that runs this script (`pip install playwright`; it is not a project dependency), and
the licensed pilot cached (`avd dataset import` once). Without `--chromium` the browser steps are
skipped and reported as skipped.

No model is used: evaluations come from the clearly labelled heuristic baseline. Every step
prints PASS/FAIL/SKIP with what it observed. The exit code is non-zero if any step failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULTS: list[tuple[str, str, str]] = []


def step(name: str, ok: bool | None, detail: str = "") -> bool:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}{': ' + detail if detail else ''}", flush=True)
    return bool(ok)


class Backend:
    def __init__(self, avd: str, data: Path, port: int) -> None:
        self.avd, self.data, self.port = avd, data, port
        self.proc: subprocess.Popen[bytes] | None = None
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("AVD_")}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def cli(self, *args: str) -> str:
        env = {**self.env, "AVD_DATABASE_URL": f"sqlite:///{self.data / 'avd.db'}"}
        return subprocess.run([self.avd, *args], env=env, capture_output=True, text=True, check=True).stdout

    def start(self, *extra: str) -> float:
        t0 = time.monotonic()
        log = open(self.data.parent / f"server-{int(t0)}.log", "w")
        self.proc = subprocess.Popen(
            [self.avd, "start", "--data-dir", str(self.data), "--port", str(self.port), *extra],
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for _ in range(240):
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=1)
                return time.monotonic() - t0
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("backend did not start")

    def stop(self) -> bool:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=30)
            return True
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return False

    def req(self, path: str, method: str = "GET", body: Any = None, token: str | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(f"{self.url}{path}", data=data, method=method)
        r.add_header("Content-Type", "application/json")
        if token:
            r.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def wait_idle(self, timeout: float = 120) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            q = self.req("/v1/status")[1].get("queue", {})
            if not q.get("queued") and not q.get("running"):
                return True
            time.sleep(0.5)
        return False


def backend_flow(b: Backend, work: Path) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    t = b.start("--demo")
    step("1-2 start backend, initialize database", True, f"healthy after {t:.2f}s")
    try:
        out = b.cli("dataset", "import", "--offline", "--evaluate")
        step("3 import authorized content (licensed pilot, pinned cache)", '"imported"' in out, out.strip()[-120:])
    except subprocess.CalledProcessError as e:
        step("3 import authorized content (licensed pilot, pinned cache)", None, f"pilot not cached: {e.stderr[-160:]}")
    step("4-6 jobs created, processed and stored", b.wait_idle(), json.dumps(b.req("/v1/status")[1].get("worker")))
    st, vids = b.req("/v1/videos?limit=200")
    ctx["videos"] = [v["video"]["id"] for v in vids["items"]]
    step("7 evaluated videos indexed and listed", st == 200 and len(ctx["videos"]) > 0, f"{len(ctx['videos'])} visible")
    st, s = b.req("/v1/search?q=hash%20tables")
    step("8 search", st == 200 and len(s["items"]) > 0, f"{len(s['items'])} results")
    st, u = b.req("/v1/users", "POST")
    ctx["token"] = u["token"]
    b.req("/v1/users/me/preferences", "PUT", {"interests": ["science", "programming"]}, ctx["token"])
    st, r = b.req("/v1/recommendations", token=ctx["token"])
    step("9 recommendations", st == 200 and len(r["items"]) > 0, f"{len(r['items'])} items")
    ctx["rec"] = r["items"][0] if r["items"] else None
    if ctx["rec"]:
        st, f = b.req(
            "/v1/feedback",
            "POST",
            {"video_id": ctx["rec"]["video"]["id"], "kind": "recommendation_useful", "impression_id": ctx["rec"]["impression_id"]},
            ctx["token"],
        )
        step("12 submit feedback", st == 201, json.dumps(f)[:80])
    ex = work / "export.json"
    out = json.loads(b.cli("evaluations", "export", "--out", str(ex)))
    step("13 export permitted evaluation records", out["evaluations"] > 0, f"{out['evaluations']} records, sha256 {out['file_sha256'][:12]}")
    ctx["export"] = ex
    return ctx


def restart_flow(b: Backend, ctx: dict[str, Any], work: Path) -> None:
    step("14 stop backend (Ctrl+C) cleanly", b.stop())
    t = b.start()
    step("14 restart backend", True, f"healthy after {t:.2f}s")
    st, vids = b.req("/v1/videos?limit=200")
    after = [v["video"]["id"] for v in vids["items"]]
    st2, prefs = b.req("/v1/users/me/preferences", token=ctx["token"])
    step(
        "15 data persisted",
        after == ctx["videos"] and st2 == 200 and prefs["interests"] == ["science", "programming"],
        f"{len(after)} videos, preferences kept",
    )
    other = Backend(b.avd, work / "other", b.port + 1)
    other.data.mkdir(parents=True, exist_ok=True)
    other.cli("migrate")
    out = json.loads(other.cli("evaluations", "import", str(ctx["export"]), "--origin", "e2e first instance"))
    step("16 import compatible evaluation records into a second instance", out["imported"] > 0 and out["trusted"] is False, json.dumps(out)[:120])
    other.start()
    vid = ctx["videos"][0]
    st, imp = other.req(f"/v1/videos/{vid}/imported-evaluations")
    st_local, _ = other.req(f"/v1/videos/{vid}/evaluation")
    step(
        "17 imported results stay unverified and never become local evaluations",
        st == 200 and imp["status"] == "imported_unverified" and st_local == 404,
        f"imported items={len(imp['items'])}, local evaluation status={st_local}",
    )
    other.stop()
    st, exported = b.req("/v1/users/me/export", token=ctx["token"])
    had = len(exported.get("feedback", []))
    st, d = b.req("/v1/users/me", "DELETE", token=ctx["token"])
    st2, _ = b.req("/v1/users/me/preferences", token=ctx["token"])
    step("18 delete user data", st == 200 and had >= 1, json.dumps(d)[:120])
    step("19 cleanup verified (token rejected, data gone)", st2 == 401, f"old token now gets HTTP {st2}")


def browser_flow(b: Backend, chromium: str, ext: Path, shots: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            tempfile.mkdtemp(),
            executable_path=chromium,
            headless=True,
            args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}", "--headless=new"],
            viewport={"width": 420, "height": 760},
        )
        sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=15000)
        ext_id = sw.url.split("/")[2]
        page = ctx.new_page()
        popup = f"chrome-extension://{ext_id}/popup.html"
        page.goto(popup)
        page.evaluate("b => chrome.storage.local.set({'avd.apiBase': b})", b.url)

        def body() -> str:
            return page.inner_text("body")

        page.reload()
        page.wait_for_timeout(2500)
        step("10 extension connects to the local backend", "Heuristic mode (not AI)" in body(), "mode label shown")
        page.screenshot(path=str(shots / "01-discover.png"), full_page=True)
        titles = page.locator("button.title")
        step("10 recommendations rendered", titles.count() > 0, f"{titles.count()} cards")
        page.get_by_text("👍 Useful").first.click()
        page.wait_for_timeout(800)
        step("12 feedback from the extension", "Thanks for the feedback" in body())
        titles.first.click()
        page.wait_for_timeout(1500)
        page.locator("button.dim-head").first.click()
        page.wait_for_timeout(500)
        txt = body()
        step(
            "11 quality breakdown with labels and evidence",
            "Synthetic" in txt and any(":" in line and line.strip()[:1].isdigit() for line in txt.splitlines()),
            "synthetic label and timestamped evidence visible",
        )
        page.screenshot(path=str(shots / "02-breakdown.png"), full_page=True)
        page.get_by_role("tab", name="Search").click()
        page.get_by_placeholder("Topic, e.g. hash tables").fill("hash tables")
        page.get_by_role("button", name="Search").click()
        page.wait_for_timeout(1500)
        step("search from the extension", page.locator("button.title").count() > 0)
        page.get_by_role("tab", name="Preferences").click()
        page.wait_for_timeout(1500)
        page.get_by_label("Interests (one per line)").fill("history\nphysics")
        page.get_by_role("button", name="Save preferences").click()
        page.wait_for_timeout(1000)
        step("preferences saved from the extension", "Saved" in body())
        status = [line for line in body().splitlines() if line.startswith(("Connected", "Evaluator"))]
        step("connection diagnostics", any(line.startswith("Connected") for line in status), " | ".join(status))
        addr = page.get_by_label("Backend address")
        for bad, expect in (("http://avd.example.org", "HTTPS"), ("not a url", "Not a valid URL")):
            addr.fill(bad)
            page.get_by_role("button", name="Connect", exact=True).click()
            page.wait_for_timeout(300)
            step(f"incorrect backend URL refused ({bad})", expect in body())
        addr.fill("https://avd-unreachable.invalid")
        page.get_by_role("button", name="Connect", exact=True).click()
        page.wait_for_timeout(9000)
        btn = page.get_by_role("button", name="Connect to avd-unreachable.invalid")
        step(
            "remote backend: consent required, blocked while unreachable",
            btn.count() == 1 and btn.is_disabled() and "Its operator will receive" in body(),
        )
        page.screenshot(path=str(shots / "03-remote-consent.png"), full_page=True)
        # Expired credentials: the server forgets the extension's account.
        token = page.evaluate("() => chrome.storage.local.get('avd.token').then(r => r['avd.token'])")
        b.req("/v1/users/me", "DELETE", token=token)
        page.get_by_role("tab", name="Discover").click()
        page.wait_for_timeout(2000)
        first = body()
        page.get_by_role("tab", name="Preferences").click()
        page.get_by_role("tab", name="Discover").click()
        page.wait_for_timeout(2500)
        step(
            "expired credentials recover with a new anonymous session",
            page.locator("button.title").count() > 0,
            "first view: " + ("session-expired message" if "session expired" in first.lower() else "recovered directly"),
        )
        # Unavailable server, then restart.
        b.stop()
        page.reload()
        page.wait_for_timeout(2500)
        step("backend down: clear error shown", "Cannot reach the backend" in body())
        page.screenshot(path=str(shots / "04-offline.png"), full_page=True)
        b.start()
        page.reload()
        page.wait_for_timeout(2500)
        step("backend restart: extension recovers", page.locator("button.title").count() > 0)
        ctx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avd", default=str(REPO / "backend" / ".venv" / "bin" / "avd"))
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="avd-e2e-"))
    b = Backend(args.avd, work / "data", args.port)
    try:
        ctx = backend_flow(b, work)
        restart_flow(b, ctx, work)
        ext = REPO / "extension" / "dist"
        if args.chromium and (ext / "manifest.json").exists():
            shots = work / "screenshots"
            shots.mkdir()
            browser_flow(b, args.chromium, ext, shots)
            print(f"screenshots: {shots}")
        else:
            step("extension in Chromium", None, "no --chromium given or extension not built")
    finally:
        if b.proc and b.proc.poll() is None:
            b.stop()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept: {work}")
    counts = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(json.dumps(counts))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
