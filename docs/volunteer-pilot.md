# Volunteer local-AI benchmark kit

**Status (2026-10-20): no real model has evaluated anything for this project yet.** Its
development environment cannot download Ollama or model weights (the registries are blocked by
its network policy), and it has no GPU. If your computer can run a small model, your run would be
the project's **first real AI evidence**.

* Everything runs on your machine, with the local SQLite mode. You do not need PostgreSQL, Redis,
  Docker, an API key or an account anywhere.
* You share one small results file, and only if you choose to. It is checked for transcript
  text, credentials and local paths before it is written.
* Nobody pays for anything, including you: local models cost nothing per call.

> **Tested platforms.** The commands below were run by the maintainers on **Linux only**, and
> never with a real model (the protocol is exercised against a mock of the Ollama API). The
> Windows PowerShell and macOS commands are written in good faith but **untested**. If one fails,
> please report it with the setup-help issue form.

## 1. Hardware

| | Minimum | Comfortable |
|---|---|---|
| RAM | 8 GB, CPU only | 16 GB, or a GPU with 6 GB or more of memory |
| Disk | 10 GB free | 20 GB free |
| OS | Windows 10/11, macOS 13+, or Linux (x86-64 or ARM64) | |
| Software | Python 3.11 or newer, git, Ollama | |

`avd local doctor` (step 7) checks your machine and recommends a model. It never downloads
anything.

**How long a run takes: unknown.** Nobody has measured it yet; the smoke test (step 8) measures
it on your machine. The pilot transcripts are long (9,000 to 14,000 words, 49 to 68 minutes of
speech), so on a CPU expect minutes per transcript, not seconds.

## 2. Recommended starter model

`qwen3.5:4b` (Q4_K_M). `avd local doctor` estimates about 2.4 GB to download and 3.5 GB of
memory. These are estimates; the real figures are recorded once the model is installed. If you
have 16 GB of RAM or a GPU, `qwen3.5:9b` is the second model worth benchmarking.

Each model has its own license, shown on its Ollama library page. You accept it when you pull the
model. This project does not redistribute weights.

## 3. Install Ollama

Download it from the official site, <https://ollama.com/download>, and install it.

* **Windows / macOS:** the installer starts Ollama in the background (a tray or menu-bar icon).
* **Linux:** use the install script from that page, then run `ollama serve` if the service is not
  started automatically.

**Keep it local.** Ollama listens on `127.0.0.1:11434` by default. Do not set `OLLAMA_HOST` to
`0.0.0.0`, and do not open that port in your firewall: the benchmark needs only your own machine.

## 4. Check out the repository

Use the commit or branch named in the issue that asks for runs. Results are compared per commit.

**Windows (PowerShell)**

```powershell
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch
git checkout <commit-or-branch-from-the-issue>
```

**Linux / macOS**

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch
cd youtubeHelperWhatWhatch
git checkout <commit-or-branch-from-the-issue>
```

## 5. Install the dependencies

**Windows (PowerShell)**

```powershell
cd backend
py -3.12 -m venv .venv                # or: py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
avd --help
```

**Linux / macOS**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
avd --help
```

Every later command runs in `backend/`, with the virtual environment active.

## 6. Prepare the authorized benchmark material

```powershell
avd dataset import
avd dataset rights
```

`avd dataset import` downloads the pilot transcripts from their publishers' public repositories,
at pinned commits, and checks each file's SHA-256 against the manifest. The files are stored in
`backend/.dataset_cache`.

* The **benchmark uses only the 10 development episodes**. All are from Changelog Media's
  transcripts, licensed CC BY-SA 4.0, with every use confirmed (`avd dataset rights`).
* **Talk Python transcripts are never part of a shared package.** Their license scope is
  unconfirmed. They appear only in the test split, which the benchmark refuses to run.
* The sealed hold-out set is never downloaded or used by any of these commands.

## 7. Check that the model is available

```powershell
avd local doctor                          # hardware, installed models, recommendation
avd local pull qwen3.5:4b --approve-gb 3  # you approve the download size; or: ollama pull qwen3.5:4b
ollama list                               # the model should be listed
```

Optional: a single live structured-output call.

**Windows (PowerShell)**

```powershell
$env:AVD_TEST_OLLAMA_URL = "http://localhost:11434"; $env:AVD_TEST_OLLAMA_MODEL = "qwen3.5:4b"
pytest -m live -q
```

**Linux / macOS**

```bash
AVD_TEST_OLLAMA_URL=http://localhost:11434 AVD_TEST_OLLAMA_MODEL=qwen3.5:4b pytest -m live -q
```

## 8. Smoke test: one transcript first

```powershell
avd community smoke --model qwen3.5:4b --out smoke-run
```

This evaluates the shortest development transcript once, with a real model call, and then runs the
package dry run. It prints:

* the model fingerprint and digest, the Ollama version and your accelerator;
* the number of model calls;
* structured-output validity, citation correctness and abstention rate;
* the wall time, output tokens per second and peak memory;
* whether the result could be packed safely.

Check that `transport` says `real` and `model_calls` is at least 1. A mock is labelled
`injected` and can never be submitted. If the wall time is acceptable, continue.

## 9. The benchmark

**Windows (PowerShell)**

```powershell
avd bench-local --models qwen3.5:4b --num-ctx 8192,16384 --repeats 2 --offline --bench-id pilot-$(Get-Date -Format yyyyMMdd)
```

**Linux / macOS**

```bash
avd bench-local --models qwen3.5:4b --num-ctx 8192,16384 --repeats 2 --offline --bench-id pilot-$(date +%Y%m%d)
```

This runs 10 transcripts × 2 context windows × 2 repeats, plus an injection probe and an
abstention probe. Stop with Ctrl+C at any time. Run the same command with the same `--bench-id`
to continue: finished calls are cached and not repeated.

To keep the run shorter, use one window (`--num-ctx 8192`) or `--repeats 1`. Repeats measure
consistency, so two are better when you can afford them.

## 10. Check the result

```powershell
avd community inspect ..\benchmarks\local\pilot-YYYYMMDD     # Linux/macOS: ../benchmarks/local/pilot-YYYYMMDD
```

`inspect` is a dry run and writes nothing. It lists:

* every file that would be shared, and how many local paths in it would be redacted;
* every file that stays on your machine, and why: run databases and per-item evaluations
  contain transcript text and quotes;
* anything that blocks sharing.

## 11. Export it safely

```powershell
avd community pack ..\benchmarks\local\pilot-YYYYMMDD --out pilot.zip
avd community verify pilot.zip
```

`pack` refuses to write anything if it finds:

| Finding | Meaning |
|---|---|
| transcript text | an 8-word run from any cached transcript appears in a shared file |
| transcript/quote field | a `quote`, `transcript` or `excerpt` field has content |
| a credential, API key or token | something that looks like a secret |
| an e-mail address | personal data |
| a network address that is not loopback | your LAN or a remote server |
| a symbolic link | it could point at a private file |
| unconfirmed uses | an item whose source has not confirmed external use |
| human labels / trust status | a contributed run cannot carry human ratings or a "verified" status |

Absolute paths, such as `C:\Users\you\…` or `/home/you/…`, are replaced with
`<local path removed>`. The file then holds scores, counts, timings, the model digest and
quantization, the Ollama version, a hardware summary (CPU model, RAM, GPU model, OS), the prompt
and rubric versions, and the git commit.

`verify` runs the checks a maintainer will run. It must say `"accepted": true`, with status
`contributor_reported`.

## 12. Submit

Open an issue with the **Benchmark result** form and attach `pilot.zip`. Use a pseudonym if you
prefer.

A maintainer then runs `avd community import pilot.zip --contributor <pseudonym>`. That repeats
every check and records the result under `benchmarks/community/<id>/` with a `VERIFICATION.json`.

What the record means:

* **Status `contributor_reported`.** A checksum proves the file was not changed after packing. It
  does not prove who made it, or that the model ran as stated.
* **Never authoritative on its own.** Independent runs that agree (same commit and model digest,
  similar numbers) are what builds confidence.
* **Operational evidence only:** validity of the output, grounding of citations, abstention,
  consistency and speed. Whether the model's quality judgments are right needs independent human
  ratings ([`annotation-pilot.md`](annotation-pilot.md)).
* **Never overwritten.** An existing record is never replaced. A second run is a second record.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `avd` is not recognized | activate the environment (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`) |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then open a new PowerShell |
| `py` is not recognized | install Python from python.org and tick "Add python.exe to PATH" |
| `Ollama unreachable at http://localhost:11434` | start the Ollama app, or run `ollama serve` |
| `model … not installed` | `ollama pull <tag>`, then restart the command |
| `context window too small` / `cut off` | use a larger `--num-ctx` (16384); output that is cut off is counted as invalid, not repaired |
| The process is killed, or the PC becomes unresponsive | not enough memory: use the 4B model and `--num-ctx 8192`, close other programs |
| `not in cache and offline mode is on` | run `avd dataset import` (step 6) |
| `pack` reports "cannot check for transcript text" | pass the cache directory: `--cache-dir .dataset_cache` |
| `pack` reports a finding | the table in step 11 explains each one. Do not edit the files to get around a finding; open a setup-help issue |
| Memory figures are empty | expected on Windows and macOS: CPU/RSS sampling is Linux-only, while GPU memory comes from Ollama on every OS |
| Long-path errors on Windows | clone into a short directory, e.g. `C:\src` |
