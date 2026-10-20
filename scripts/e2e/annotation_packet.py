"""End-to-end check of the offline rating packet in a real browser.

Builds a throw-away database from the licensed pilot (from the local cache; nothing is
downloaded), issues a calibration packet with the real CLI, fills one item in headless Chromium
the way a rater would (status, score, a citation made by selecting transcript text), downloads the
ratings file, and imports it with the CLI. It also confirms that the packet makes no network
request.

The ratings entered here are a scripted test input, not a human judgement. The database is
deleted afterwards; nothing is exported as a label.

    python scripts/e2e/annotation_packet.py --avd backend/.venv/bin/avd --chromium <path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results: dict[str, int] = {"PASS": 0, "FAIL": 0}


def step(name: str, ok: bool, detail: str = "") -> None:
    results["PASS" if ok else "FAIL"] += 1
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avd", default="avd")
    ap.add_argument("--chromium", required=True)
    args = ap.parse_args()
    if os.sep in args.avd:  # a relative path must survive the working-directory change below
        args.avd = str(Path(args.avd).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="avd-packet-"))
    env = {
        **os.environ,
        "AVD_DATABASE_URL": f"sqlite:///{tmp / 'coord.db'}",
        "AVD_LLM_PROVIDER": "heuristic",
    }
    env.pop("AVD_REDIS_URL", None)

    def avd(*a: str) -> str:
        return subprocess.run(
            [args.avd, *a],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT / "backend",
        ).stdout

    subprocess.run(
        [args.avd, "migrate"],
        env=env,
        capture_output=True,
        check=False,
        cwd=ROOT / "backend",
    )
    manifest = str(ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    imported = json.loads(avd("dataset", "import", manifest, "--offline").strip().splitlines()[-1])
    step(
        "coordinator imports the licensed pilot from the cache",
        bool(imported),
        json.dumps(imported)[:120],
    )
    avd("annotate", "create-rater", "volunteer-a")
    # The repository template is a draft awaiting owner approval; test with the approval line removed.
    template = (ROOT / "docs" / "templates" / "rating-consent.txt").read_text(encoding="utf-8")
    consent = tmp / "consent-approved-for-test.txt"
    consent.write_text(
        "\n".join(line for line in template.splitlines() if "DRAFT FOR PROJECT-OWNER APPROVAL" not in line),
        encoding="utf-8",
    )
    issued = json.loads(
        avd(
            "annotate",
            "packet",
            "--rater",
            "volunteer-a",
            "--out",
            str(tmp / "packet.html"),
            "--consent-file",
            str(consent),
        )
    )
    step(
        "onboarding packet holds the calibration items",
        issued["calibration"] and len(issued["items"]) == 2,
        str(issued["items"]),
    )
    html = (tmp / "packet.html").read_text(encoding="utf-8")
    step("packet forbids network access", "default-src 'none'" in html)
    step(
        "packet carries no model output or popularity",
        '"overall_score"' not in html and "view_count" not in html,
    )

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(accept_downloads=True)
        requests: list[str] = []
        page.on("request", lambda r: requests.append(r.url))
        page.goto((tmp / "packet.html").as_uri())
        page.wait_for_selector(".seg")
        dims = page.locator(".dim select[aria-label^='Status for']")
        n = dims.count()
        step("all ten rubric dimensions are shown", n == 10, f"{n} dimensions")
        first = page.locator(".dim").first
        first.locator("select[aria-label^='Status for']").select_option("scored")
        page.locator(".dim").first.locator("input[type=radio][value='3']").check()
        # Select the first words of segment 2, as a rater would with the mouse, then cite them.
        page.evaluate(
            """() => {
              const seg = document.querySelector('.seg[data-index="2"]');
              const text = [...seg.childNodes].find(n => n.nodeType === 3);
              const words = text.textContent.trim().split(/\\s+/).slice(0, 6).join(' ');
              const start = text.textContent.indexOf(words);
              const r = document.createRange();
              r.setStart(text, start); r.setEnd(text, start + words.length);
              const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
            }"""
        )
        page.locator(".dim").first.get_by_role("button", name="Cite selection").click()
        step(
            "a selection becomes a citation",
            page.locator(".cite").count() == 1,
            page.locator(".cite").first.inner_text()[:80],
        )
        for i in range(1, n):
            page.locator(".dim select[aria-label^='Status for']").nth(i).select_option(
                "insufficient_evidence"
            )
        page.get_by_role("button", name="Mark this item complete").click()
        step("item marked complete", "✓" in page.locator("nav button").first.inner_text())
        page.reload()  # drafts survive a reload
        step(
            "progress survives a reload",
            "✓" in page.locator("nav button").first.inner_text(),
        )
        alerts: list[str] = []
        page.once("dialog", lambda d: (alerts.append(d.message), d.dismiss()))
        page.get_by_role("button", name="Download ratings file").click()
        page.wait_for_timeout(500)
        step(
            "no ratings file before consent is given",
            any("consent" in m for m in alerts),
            alerts[0][:60] if alerts else "",
        )
        step("consent text is shown", "Pseudonymous, not anonymous" in page.inner_text("body"))
        page.locator("#consent-box").check()
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download ratings file").click()
        out = tmp / "ratings.json"
        dl.value.save_as(out)
        step(
            "no network request was made",
            all(u.startswith(("file:", "blob:")) for u in requests),
            f"{len(requests)} requests",
        )
        browser.close()

    back = json.loads(avd("annotate", "import-packet", str(out)))
    step(
        "coordinator imports the ratings file",
        len(back["accepted"]) == 1 and not back["rejected"] and len(back["still_open"]) == 1,
        json.dumps({k: back[k] for k in ("accepted", "rejected", "still_open", "rater_status")})[:200],
    )
    again = json.loads(avd("annotate", "import-packet", str(out)))
    step(
        "importing the same file twice records nothing new",
        again["already_imported"] == back["accepted"] and not again["accepted"],
    )
    tampered = json.loads(out.read_text())
    tampered["secret"] = "pkt_forged"
    (tmp / "forged.json").write_text(json.dumps(tampered))
    r = subprocess.run(
        [args.avd, "annotate", "import-packet", str(tmp / "forged.json")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT / "backend",
    )
    step(
        "a forged return is refused",
        r.returncode != 0 and "secret" in r.stderr,
        r.stderr.strip()[:80],
    )

    # Usability pilot: a labelled excerpt to test the packet itself; it can never become a rating.
    pilot = json.loads(
        avd(
            "annotate",
            "usability-packet",
            "--label",
            "pilot-a",
            "--out",
            str(tmp / "pilot.html"),
            "--consent-file",
            str(consent),
        )
    )
    step(
        "usability pilot: an excerpt of the shortest calibration episode",
        pilot["excerpt_words"] < pilot["episode_words"] and pilot["excerpt_words"] >= 1500,
        f"{pilot['item']}: {pilot['excerpt_words']} of {pilot['episode_words']} words",
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(accept_downloads=True)
        requests = []
        page.on("request", lambda r: requests.append(r.url))
        page.goto((tmp / "pilot.html").as_uri())
        page.wait_for_selector(".seg")
        body = page.inner_text("body")
        step(
            "usability pilot: labelled as an excerpt, never a rating, on screen",
            "Usability pilot (excerpt, not a rating)" in body and "never a rating of the episode" in body,
        )
        page.locator(".dim").first.locator("select[aria-label^='Status for']").select_option("scored")
        page.locator(".dim").first.locator("input[type=radio][value='2']").check()
        page.evaluate(
            """() => {
              const seg = document.querySelector('.seg[data-index="1"]');
              const text = [...seg.childNodes].find(n => n.nodeType === 3);
              const words = text.textContent.trim().split(/\\s+/).slice(0, 5).join(' ');
              const start = text.textContent.indexOf(words);
              const r = document.createRange();
              r.setStart(text, start); r.setEnd(text, start + words.length);
              const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
            }"""
        )
        page.locator(".dim").first.get_by_role("button", name="Cite selection").click()
        dims = page.locator(".dim select[aria-label^='Status for']")
        for i in range(1, dims.count()):
            dims.nth(i).select_option("insufficient_evidence")
        page.locator("textarea[placeholder^='What was unclear']").fill("Test note: statuses were clear.")
        page.get_by_role("button", name="Mark this item complete").click()
        page.locator("#consent-box").check()
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download ratings file").click()
        pilot_out = tmp / "pilot-return.json"
        dl.value.save_as(pilot_out)
        step("usability pilot: no network request", all(u.startswith(("file:", "blob:")) for u in requests))
        browser.close()
    r = subprocess.run(
        [args.avd, "annotate", "import-packet", str(pilot_out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT / "backend",
    )
    step(
        "usability pilot: the return is refused as a rating",
        r.returncode != 0 and "never a rating" in r.stderr,
        r.stderr.strip()[:90],
    )
    rep = json.loads(avd("annotate", "usability-report", str(pilot_out), "--consent-file", str(consent)))
    it = rep["items"][0]
    step(
        "usability pilot: report reads timing, answers, citations and notes; stores nothing",
        rep["is_rating"] is False
        and it["dimensions_answered"] == 10
        and it["citations"]["found_in_excerpt"] == 1
        and it["notes"] == "Test note: statuses were clear.",
        json.dumps({k: it[k] for k in ("minutes_reported", "dimensions_answered", "citations")})[:160],
    )
    print(json.dumps(results))
    sys.exit(1 if results["FAIL"] else 0)


if __name__ == "__main__":
    main()
