# First experiment: one real model evaluation and three human calibration ratings

The goal is the project's first real evidence. Nobody pays for anything, and there is no
central server.

* **Part A (one volunteer with a computer):** a real local model evaluates one authorized
  development transcript. The result records the model digest, latency, hardware and evidence
  validity, and is submitted to GitHub for review.
* **Part B (three volunteers, any computer with a browser):** independent offline ratings of
  the two calibration episodes.

What was tested where (2026-10-20):

| Step | Linux | Windows |
|---|---|---|
| Clone, install, dataset import | **tested** on a fresh clone (Ubuntu 24.04, Python 3.12) | not tested |
| `avd community smoke` → `pack` → `verify` | **tested** against a *protocol mock* of the Ollama API (no real model was available): `verify` correctly refused the mock run | not tested |
| A real model call | **not done**: model downloads are blocked in the development environment | not done |
| Offline rating packet | **tested** in headless Chromium on the real transcripts (consent, citation, download, import) | the packet is a plain HTML file; any modern browser |

## Part A: the model run (about 1–3 hours, mostly unattended)

**You need** 16 GB of RAM (8 GB minimum), 10 GB of free disk, internet for the downloads, and a
free GitHub account to submit.

The commands use branch `claude/phase3-4-evidence`, which contains everything until PRs
#2–#7 are merged. Use `main` after that.

### Windows (PowerShell), untested: please report anything that differs

Open **PowerShell** and install the tools. You can skip any you already have. Close
PowerShell afterwards and open a new window.

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id Ollama.Ollama -e
```

If `winget` is not available, install Git from git-scm.com, Python 3.12 from python.org (tick
"Add python.exe to PATH") and Ollama from https://ollama.com/download.

In the **new** PowerShell window:

```powershell
cd $HOME
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch
git checkout claude/phase3-4-evidence
cd backend
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
avd dataset import ..\benchmarks\datasets\real_pilot_v1\manifest.json
avd local doctor
ollama pull qwen3.5:4b
ollama list
avd community smoke --model qwen3.5:4b --out smoke-run
```

Then continue with **Check and submit** below.

### Linux, tested up to the model call

Install Ollama with the script from https://ollama.com/download/linux and make sure the server
is running (`ollama serve`, or the system service). Then:

```bash
cd ~
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch
git checkout claude/phase3-4-evidence
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
avd dataset import ../benchmarks/datasets/real_pilot_v1/manifest.json
avd local doctor
ollama pull qwen3.5:4b
ollama list
avd community smoke --model qwen3.5:4b --out smoke-run
```

macOS is the same as Linux, with Ollama from https://ollama.com/download/mac (untested).

### Notes on these commands

* **Dataset import.** `avd dataset import` downloads the 29 pilot transcripts (about 2.3 MB)
  from the publishers' public GitHub repositories, at pinned commits, and checks each file's
  SHA-256. The run uses only one of them: the shortest development episode, *Practical AI
  #178* (9,062 words, Changelog Media, CC BY-SA 4.0). The sealed hold-out is never downloaded.
* **Model tag.** `qwen3.5:4b` is the tag `avd local doctor` recommends. The project could not
  check it against the Ollama library. If `ollama pull` says it does not exist, pull a small
  instruction model that does (for example `qwen3:4b` or `llama3.2:3b`) and pass that tag to
  `--model`.
* **Slow computers.** On a CPU, one model call over a long transcript can take many minutes.
  The smoke test allows 30 minutes per call. On a slow laptop, add `--timeout 3600`.
* **Keep Ollama local.** Do not set `OLLAMA_HOST=0.0.0.0` or open port 11434: the run needs only
  your own computer.

### What `smoke` prints

A JSON summary of the one real evaluation:

| Field | Meaning |
|---|---|
| `transport` | `real` means an HTTP Ollama server answered |
| `model`, `model_digest`, `quantization`, `runtime_version` | exactly which weights and which Ollama version produced it |
| `hardware` | OS, CPU model, logical cores, RAM, GPUs, accelerator. Nothing personal: no host name, user name or path |
| `model_calls`, `errors` | how many real calls were made, and failures |
| `structured_validity` | share of calls that returned valid structured output |
| `evidence_claimed`, `evidence_verified`, `citation_correctness` | quotes the model cited, and how many were found at the cited place in the transcript |
| `latency_ms_p50`, `latency_ms_p95`, `wall_seconds`, `output_tokens_per_second` | speed on your machine |
| `warnings` | anything that would make the result unusable (for example a mock server, or uncommitted changes) |
| `pack_dry_run_ok` | whether the result can be shared safely |

`warnings` must be empty. Real model calls produce real, imperfect numbers. That is the point:
do not rerun until the numbers look good.

### Check and submit (both systems)

```bash
avd community pack smoke-run --out smoke.zip
avd community verify smoke.zip
```

`pack` writes only summaries and reports. It refuses to write anything if a file contains
transcript text, a credential, an e-mail address, a non-loopback address, a symbolic link, or an
item whose publisher has not confirmed sharing. Local paths are replaced with
`<local path removed>`. `verify` must print `"accepted": true` and
`"status": "contributor_reported"`.

To submit, open https://github.com/Apostal322/youtubeHelperWhatWhatch/issues/new/choose, choose
**Benchmark result (local model)**, attach `smoke.zip`, and paste the output of `verify`.
Use a pseudonym if you prefer.

Do not attach anything else from the `smoke-run` folder. The run database there contains
transcript text.

### What the maintainer does with it

```bash
avd community verify smoke.zip
avd community import smoke.zip --contributor <pseudonym>
```

The import repeats every check and records the result under `benchmarks/community/` as
**contributor-reported**. Status, limits and what it does not prove:

* A checksum shows the file was not changed after packing. It does not show who ran it, or
  that the model really ran.
* A second, independent run with the same model digest is what makes it credible.
* It is operational evidence (valid output, grounded citations, speed). It says nothing yet
  about whether the scores are *right*: that needs Part B.

## Part B: three calibration ratings (2.5–3 hours each, estimated)

The coordinator (the project owner) has the prepared round: three packets, the coordinator
database and a README. The steps are:

1. **Approve the consent text** (`rating-consent.txt`; template in
   `docs/templates/rating-consent.txt`). If you change it, reissue the packets as the round's
   README describes.
2. **Recruit three volunteers.** Use the invitation template in
   [`annotation-pilot.md`](annotation-pilot.md). Give each one packet and the participant
   instructions from the same page.
3. **Each volunteer:**
   * opens their `packet-volunteer-N.html` in any modern browser. It works offline and cannot
     send anything;
   * reads the consent text and ticks the box;
   * rates both practice episodes;
   * presses **Download ratings file** and sends that file back.
4. **The coordinator imports each file** with `avd annotate import-packet` (see the round's
   README). Files without consent, with the wrong secret, or forged are refused.

Calibration ratings train and calibrate the raters, and are never exported as labels. Once all
three are back, the raters' development packets (`avd annotate packet --rater volunteer-N
--items 2 --consent-file rating-consent.txt`) produce the first human labels and relevance
judgments.

## Nothing here needs an official server or a paid API

| Workflow | Network use | Paid service |
|---|---|---|
| Install | PyPI, GitHub (free) | none |
| Dataset import | the publishers' GitHub repositories (raw files at pinned commits) | none |
| Model | the Ollama library, once, for the weights; inference runs on `localhost` | none |
| Smoke test, pack, verify | none (the dataset cache is used offline; the model is local) | none |
| Submission | a GitHub issue (free account) | none |
| Rating packets | none: the file blocks every network request | none |
| Coordinator import | none | none |

No API key is needed anywhere. The optional hosted-model provider is never used by these
commands.
