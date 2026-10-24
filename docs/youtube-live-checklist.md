# Real-YouTube browser checklist

The extension has never been run on youtube.com. Everything below marked **simulated** passed in
headless Chromium against a stand-in page served at `https://www.youtube.com/...` by the test
itself (`scripts/e2e/youtube_navigation.py`, 17 steps) and against a local backend
(`scripts/e2e/full_local.py`, 34 steps). A simulated pass shows the extension's own logic is
right; it does not show that YouTube behaves the way the stand-in page does.

## What is covered where

| Behaviour | Simulated E2E | Live youtube.com |
|---|---|---|
| Badge on a directly opened watch page | ✅ | ☐ not tested |
| In-page navigation (YouTube changes the URL without reloading) | ✅ via `history.pushState` | ☐ not tested: YouTube's real navigation may differ |
| Back, forward, reload | ✅ | ☐ |
| A slow answer for the previous video never shows on the current one | ✅ (delayed stub backend) | ☐ |
| Playlist (`&list=`), Shorts (`/shorts/`), live (`/live/`) URLs | ✅ | ☐ |
| Extra URL parameters YouTube adds (`&t=`, `&pp=`, `&si=`, `&index=`) | ☐ not simulated | ☐ |
| Autoplay moving to the next video | ☐ not simulated | ☐ |
| Miniplayer, theatre mode, full screen | ☐ not simulated | ☐ |
| Ads before or during the video | ☐ not simulated | ☐ |
| Cookie-consent page before YouTube (`consent.youtube.com`, EU) | ☐ not simulated | ☐ |
| Signed in vs signed out | ☐ not simulated | ☐ |
| Video not in the catalog / non-video page / backend down and recovery | ✅ | ☐ |
| Two tabs keep their own scores | ✅ | ☐ |
| Submitting a transcript from the popup | ✅ (local backend) | ☐ |
| Score, label ("AI" or "heuristic") and evidence quotes in the popup | ✅ heuristic only | ☐ with a real model: never |
| `youtu.be`, `m.youtube.com`, `music.youtube.com` | ✖ outside the extension's permissions by design | ✖ same |

## Before you start (about 10 minutes)

* One video whose transcript **you have the right to submit**: your own upload, or a video whose
  creator agreed. Nothing is downloaded from YouTube; you paste the transcript yourself. Keep the
  transcript as text, VTT or SRT.
* In `backend/`: `avd start` (heuristic evaluator, fast) or, after
  `docs/first-experiment.md` Part A, `avd start --provider ollama` (real model, slow on a CPU).
* In `extension/`: `npm ci && npm run build`, then `chrome://extensions` → *Developer mode* →
  *Load unpacked* → `extension/dist/`.
* Write down: Chrome version, OS, signed in or not, country (for the consent page), extension
  commit (`git rev-parse --short HEAD`).

## Checklist (about 20 minutes, plus model time)

Mark each **pass**, **fail** (what you saw) or **n/a**. A screenshot for every fail helps.

1. ☐ Open the video's watch page. Before submitting anything the badge is empty and its tooltip
   (hover the extension icon) reads "AI Video Discovery — not analyzed yet": no score, no "!".
2. ☐ Open the popup, submit the transcript with your attestation. The popup says it is queued.
3. ☐ When the evaluation finishes, the badge shows a score. With `--provider ollama` this can
   take many minutes: note how long.
4. ☐ The popup header reads "AI evaluation · not human-validated" (real model) or "Heuristic mode
   (not AI)", and the score's source names the model.
5. ☐ Every evidence quote shown is really in your transcript, at a sensible place.
6. ☐ Click another video in the sidebar (in-page navigation). The tooltip briefly reads
   "checking…", then "not analyzed yet"; the badge never keeps your video's score.
7. ☐ Browser Back: your video's score returns. Forward: the other video again.
8. ☐ Reload the page: the right score is still shown.
9. ☐ Open your video with extra parameters, e.g. `&t=30s` and, if you have one, from a playlist
   (`&list=...`). Same score.
10. ☐ If your video is a Short, open it at `/shorts/<id>`. Same score.
11. ☐ Let autoplay move to the next video. The badge follows it.
12. ☐ Miniplayer and theatre mode: the badge stays correct.
13. ☐ With an ad playing before the video: the badge is for the video, not blank or wrong.
14. ☐ Open a second tab on a different video; switch between tabs. Each keeps its own badge.
15. ☐ Stop `avd` (Ctrl+C) and open another video. The tooltip reads "backend unreachable" (not
    "not analyzed yet"). Start `avd` again and revisit your video; its score returns without
    reloading the extension. (A "!" badge appears only if the backend answers with a server error.)
16. ☐ Open the YouTube home page and a search results page. The badge is cleared and the tooltip
    is just "AI Video Discovery".

## Report

Open an issue (**Bug report** for failures, otherwise **Setup help** or a comment on the
tracking issue) with: the environment notes above, the pass/fail list, and screenshots. Do not
attach your transcript if it is not public.

Until someone reports this checklist, the release checklist keeps "real YouTube: not tested".
