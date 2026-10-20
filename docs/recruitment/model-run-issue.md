# GitHub issue, ready to publish (owner approval required)

> **Not posted.** The owner decides whether to open it. Suggested labels: `help wanted`,
> `benchmark-result`. Everything below the line is the issue: the first line is the title, the
> rest the body. Commands were checked on Linux up to the model call; the Windows block is
> untested and says so.

---

**Title:** Help wanted: the first real AI evaluation, one local-model run on your own computer (about 1–3 hours, mostly unattended)

**Body:**

AI Video Discovery assesses long-form talks and podcasts by their content, not their
popularity. **No real AI model has evaluated anything yet.** Every score so far comes from a
labelled heuristic baseline, because the development environment cannot download models. One
run on your machine would be the project's first real model evaluation.

**What it costs you:** nothing but time. The model runs on your computer through
[Ollama](https://ollama.com). There is no API key, no paid service, no account with us and no
project server.

**What you need**
- 16 GB RAM recommended (8 GB works, slowly), 10 GB free disk, no GPU required
- Python 3.11+ and git, about 1–3 hours (mostly waiting), a GitHub account to submit
- Ollama 0.5 or newer (structured output). Keep it on `127.0.0.1`: do not set `OLLAMA_HOST=0.0.0.0`

**Speed is unknown.** On a CPU one call over a 9,000-word transcript may take many minutes;
measuring that is part of the point.

**Linux / macOS** (tested on Linux up to the model call; macOS untested)

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch && git checkout main && cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
avd dataset import ../benchmarks/datasets/real_pilot_v1/manifest.json
avd local doctor
ollama pull qwen3.5:4b
avd community smoke --model qwen3.5:4b --out smoke-run      # slow laptop: add --timeout 3600
avd community review smoke-run
avd community pack smoke-run --out smoke.zip
avd community verify smoke.zip
```

**Windows (PowerShell): untested.** Please tell us what differs.

```powershell
winget install --id Git.Git -e; winget install --id Python.Python.3.12 -e; winget install --id Ollama.Ollama -e
# open a NEW PowerShell window, then:
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch; git checkout main; cd backend
py -3.12 -m venv .venv; Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
avd dataset import ..\benchmarks\datasets\real_pilot_v1\manifest.json
avd local doctor
ollama pull qwen3.5:4b
avd community smoke --model qwen3.5:4b --out smoke-run
avd community review smoke-run
avd community pack smoke-run --out smoke.zip
avd community verify smoke.zip
```

**If `ollama pull qwen3.5:4b` says the model does not exist:** we could not check that tag
against the Ollama library. Pull a small instruction model that does exist (for example
`qwen3:4b` or `llama3.2:3b`) and use that tag with `--model`. Say which one you used.

**What the run does.** `smoke` evaluates **one** transcript, *Practical AI #178* (Changelog
Media, CC BY-SA 4.0, 9,062 words). It downloads nothing else and never touches held-out data.
Its summary must show:
- `transport: "real"` and `model_calls` of at least 1;
- an **empty** `warnings` list. A mock server, or uncommitted changes in your checkout, makes
  the run unusable.

Real models produce imperfect numbers. Please don't rerun to make them look better.

**What to submit.** Open an issue with the **Benchmark result (local model)** form. Attach
**only `smoke.zip`**, and paste the output of `avd community verify smoke.zip`, which must show
`"accepted": true` and `"status": "contributor_reported"`.

`smoke.zip` contains:
- scores, counts and timings;
- the model's per-dimension output for this one openly licensed transcript;
- the model name and digest, the Ollama version, the prompt and rubric versions and the git
  commit;
- a hardware summary (OS, CPU model, cores, RAM, GPU model).

`pack` refuses to write it if it finds transcript text beyond cited quotes, credentials, e-mail
addresses, network addresses, symbolic links or local paths. Local paths are redacted.

**Never upload** the `smoke-run` folder (its database holds transcript text), API keys or
tokens, or anything else from your computer.

**What it will mean.** The result is recorded as *contributor-reported*. It shows whether the
model produced valid, schema-conforming output, whether its cited quotes exist in the
transcript, and how fast it ran on your hardware. It does **not** show that the model's quality
judgements are right: that needs independent human ratings, collected separately. A second
independent run with the same model digest is what makes a result credible.

Stuck? Open a **Setup help** issue and paste `avd local doctor`. It contains no secrets: no host
name, user name or path.
