# Personal installation (local mode)

One process, one SQLite file, no accounts, no Redis, no PostgreSQL, no Docker, no API key. Your
data stays in a directory on your computer.

**Tested on:**

* Linux (Ubuntu 24.04, Python 3.12): the commands below were run end to end.
* **macOS and Windows: not tested.** The code has no platform-specific dependency beyond
  Python 3.11+ (ffmpeg is needed only for submitting your own audio/video). The commands are
  given in good faith; please report problems.

## 1. Install the application

**Linux / macOS**

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch\backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e .
```

## 2. Start it

```bash
avd start --demo        # first time: adds 20 clearly labelled SYNTHETIC demo videos
avd start               # afterwards
```

It prints where your data lives and which evaluator is active, then serves the API on
`http://127.0.0.1:8000`, reachable from this computer only.

| Data directory | |
|---|---|
| Linux | `~/.local/share/ai-video-discovery` |
| macOS | `~/Library/Application Support/ai-video-discovery` |
| Windows | `%LOCALAPPDATA%\ai-video-discovery` |

Override it with `--data-dir` or `AVD_DATA_DIR`.

**Stopping:** press Ctrl+C. The worker finishes its current step. A job cut off mid-way is
re-queued on the next start, so nothing is lost or counted twice.

**Measured** on a 4-vCPU Linux VM without a GPU, heuristic evaluator:

* start to first answer: 1.3–1.9 s;
* API process: about 120 MB RSS idle;
* data directory: 11 MB with 49 evaluated items (20 synthetic + 29 licensed transcripts);
* search: 25–45 ms; recommendations: 10–65 ms.

Local AI models need far more; see [`local-ai.md`](local-ai.md).

## 3. Connect the browser extension

Build it once, then load it:

```bash
cd extension && npm ci && npm run build
```

Open `chrome://extensions`, enable Developer mode, choose *Load unpacked* and select
`extension/dist`. The extension talks to `http://localhost:8000` by default. Any port on
`localhost`/`127.0.0.1` works without extra permissions. *Preferences → Backend* shows the
connection status and what evaluator the backend uses.

## 4. Choose an evaluator

| Evaluator | How | What you get |
|---|---|---|
| Heuristic baseline (default) | nothing to configure | keyword/structure rules, labelled "Heuristic mode (not AI)" everywhere |
| Local AI | install Ollama and a model (`avd local doctor`, `avd local pull …`), then `avd start --provider ollama` | open-weight model on your hardware; nothing leaves your computer |
| Hosted AI | set `AVD_LLM_PROVIDER=anthropic` (or `openai_compatible`) and its key | **sends transcripts to that provider**, billed to your account |

Model installation is separate from application installation. Nothing large is downloaded
unless you approve it (`avd local pull … --approve-gb N`). There is no automatic fallback from
one provider to another.

## 5. Add content you are allowed to analyze

A new installation is empty: there is no central catalog. Options:

* `avd start --demo`: synthetic examples written for this project, always labelled as such.
* `avd dataset import`: the licensed pilot (29 podcast transcripts at pinned commits). Its
  assessments stay hidden in the API while its blind human annotation is open, so that raters
  are not influenced. This is intended.
* On a video's page in the extension, submit a transcript you have the right to share (you
  attest to it).
* With a YouTube Data API key of your own (`AVD_YOUTUBE_API_KEY`), `avd discover` adds catalog
  entries with metadata only. Videos are evaluated only when authorized content exists. Being
  publicly viewable is not permission to download.

## 6. Everyday commands

```bash
avd doctor                         # configuration, database, queue, AI provider, hardware
avd jobs stats                     # queue overview
avd jobs cancel --type stage_b     # cancel queued work (running jobs finish their attempt)
avd evaluations export --out e.json
avd data backup --out ~/avd-backup.db
avd data clear --model-cache --yes
```

Data ownership and privacy: [`privacy.md`](privacy.md). Running it for other people:
[`self-hosting.md`](self-hosting.md).
