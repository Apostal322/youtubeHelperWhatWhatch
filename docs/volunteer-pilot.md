# Run the first real local-model pilot (volunteer guide)

The project has no GPU and no inference budget. Its development environment cannot download
Ollama or model weights, so **no real model has evaluated anything yet**. If you have a computer
that can run a small model, you can produce the first real results in about an hour of mostly
unattended time. Everything stays on your machine except a small results file you choose to
share.

## What you need

* Linux, macOS or Windows with Python 3.11+, git and about 10 GB of free disk.
* At least 8 GB of RAM for a 4B model on CPU (16 GB is comfortable). A GPU helps but is not
  required.
* [Ollama](https://ollama.com) installed from its official site.

## Steps

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch && cd youtubeHelperWhatWhatch
git checkout <the commit or branch named in the issue asking for runs>
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -e .   # Windows: see docs/local-setup.md

avd local doctor                          # what your hardware can run; downloads nothing
ollama serve                              # in another terminal (or the Ollama app)
avd local pull qwen3.5:4b --approve-gb 3  # or the model doctor recommends; you approve the size

avd dataset import --offline || avd dataset import    # licensed pilot transcripts at pinned commits
pytest -m live                             # one real structured-output call: must pass first
avd bench-local --models qwen3.5:4b --num-ctx 8192,16384 --repeats 2 --bench-id pilot-$(date +%Y%m%d)
avd community pack ../benchmarks/local/pilot-YYYYMMDD --out pilot.zip
avd community verify pilot.zip             # check it before sharing
```

`pytest -m live` needs `AVD_TEST_OLLAMA_URL=http://localhost:11434` and
`AVD_TEST_OLLAMA_MODEL=qwen3.5:4b`.

Expect a CPU-only run of 10 items × 2 windows × 2 repeats to take a while; nobody has measured it
yet. You can stop with Ctrl+C. The runs are resumable with the same `--bench-id`.

## What you share, and what you don't

`pilot.zip` contains only `report.json`/`report.md` and per-run summaries. These hold scores,
counts, timings, the model digest, your hardware summary and the git commit. `avd community pack`
**excludes** the run databases and media, because those contain transcript text. Attach
`pilot.zip` to a GitHub issue or pull request. Choose a pseudonym if you prefer; nothing personal
is in the file beyond your CPU/GPU model and OS.

## What happens next

A maintainer runs `avd community import pilot.zip --contributor <your pseudonym>`. It re-runs the
checks and records the result under `benchmarks/community/` as **contributor-reported**. It is
not proof that the model ran: the file is unsigned and the numbers are self-reported.
Independent runs that agree are what builds confidence.

What these numbers mean:

* structured-output validity, citation correctness, abstention, consistency and speed of a real
  model on real transcripts;
* **not** whether its quality judgements are right. That needs independent human ratings
  ([`annotation.md`](annotation.md)).

The sealed hold-out is never used here; `bench-local` refuses it.
