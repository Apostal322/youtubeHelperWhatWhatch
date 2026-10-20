# Local AI evaluation (Ollama)

Run the AI evaluator on your own hardware with open-weight models: no API key, no cloud
inference and no per-call cost. The hosted providers (`anthropic`, `openai_compatible`) remain
available as optional alternatives; nothing here changes them.

> **Status (2026-10-19).** The integration is implemented and tested against a protocol mock of
> the Ollama API. **No local model has been run in this repository yet**: the development
> container has no network route to `ollama.com`, `registry.ollama.ai` or `huggingface.co`, so
> neither Ollama nor any model weights could be installed. Every number in this page that
> describes a model is an *estimate* or a *procedure*, not a measurement. See
> [`phase3_1-report.md`](phase3_1-report.md).

## 1. Check the machine

```bash
cd backend && pip install -e ".[dev]"
avd local doctor            # hardware, Ollama status, installed models, recommendations
avd local doctor --json     # the same, machine-readable
```

`doctor` reads the OS (`/proc`, `sysctl`) and vendor tools when present (`nvidia-smi`,
`rocm-smi`). It does not assume NVIDIA: AMD (ROCm), Apple Silicon (Metal, unified memory) and
CPU-only machines are supported. It reports:

* OS, CPU model, logical cores and the SIMD flags relevant to CPU inference (AVX2, AVX-512, AMX);
* RAM, available RAM and swap; GPU, VRAM and runtime (CUDA/ROCm/Metal); free disk where models are stored;
* whether the `ollama` binary is installed, and whether the server answers and with which models;
* for each candidate model and quantization: estimated memory and download size, and whether it fits.

It never downloads anything.

### How a model is chosen

| Machine | Judged against | Preference |
|---|---|---|
| GPU (NVIDIA, AMD) | free VRAM − 0.5 GB | the largest model that fits (quality first) |
| Apple Silicon | ~2/3 of unified memory (Metal's budget) | the largest model that fits |
| CPU only | available RAM − 3 GB host reserve | the fewest *active* parameters (speed first), ≤ 10B active |

On a CPU, decoding speed is bounded by memory bandwidth divided by the bytes of *active* weights,
so a 4B dense model, or a mixture-of-experts model with ~3B active parameters if it fits, is the
practical choice. `doctor` also proposes up to three candidates to benchmark.

**Candidates** (Ollama library tags; confirm each on the library page before pulling, since tags
could not be verified from the development environment):

| tag | params (active) | vision | note |
|---|---|---|---|
| `qwen3.5:4b` | 4B (4B) | yes | CPU-friendly baseline |
| `qwen3.5:9b` | 9B (9B) | yes | quality/speed middle |
| `qwen3.5:27b` | 27B (27B) | yes | dense; large GPU or lots of RAM |
| `qwen3.5:35b-a3b` | 35B (~3B) | yes | MoE: big memory, fast decoding |

Memory estimates are weights (parameters × bits per weight: 4.85 for `q4_K_M`, 8.5 for `q8_0`)
plus a conservative context allowance and 0.8 GB runtime overhead. They err high. After a pull,
`ollama show <tag>` and `/api/ps` give real figures, and `avd bench-local` measures them.

The development container measured: 4 vCPU Xeon (AVX-512, AMX), 15.7 GB RAM, no swap, no GPU,
26 GB free disk. `doctor` recommends `qwen3.5:4b`, with `qwen3.5:9b` as the second benchmark
candidate.

## 2. Install Ollama and a model

```bash
curl -fsSL https://ollama.com/install.sh | sh        # Linux; macOS/Windows: the installer from ollama.com
ollama serve                                          # listens on 127.0.0.1:11434 by default
avd local pull qwen3.5:4b --approve-gb 3              # refuses unless the approval covers the size
```

`avd local pull` exists so that a download is always an explicit decision. It refuses:

* when `--approve-gb` is smaller than the model's estimated download size;
* models not in the table, unless you give `--size-gb` from the library page;
* cloud models (`*-cloud`, `*:cloud`);
* when the disk does not have room.

`ollama pull <tag>` works too.

With Docker Compose:

```bash
echo "AVD_LLM_PROVIDER=ollama" >> .env
docker compose --profile local-ai up --build
docker compose exec ollama ollama pull qwen3.5:4b
```

The compose `ollama` service publishes its port on `127.0.0.1` only.

## 3. Configure

| Setting | Default | Meaning |
|---|---|---|
| `AVD_LLM_PROVIDER` | `heuristic` | `ollama` to evaluate locally |
| `AVD_OLLAMA_BASE_URL` | `http://localhost:11434` | native API root (a trailing `/v1` is accepted) |
| `AVD_OLLAMA_MODEL` | `qwen3.5:4b` | used for Stage B and C unless `AVD_LLM_MODEL_STAGE_B/C` are set explicitly |
| `AVD_OLLAMA_NUM_CTX` | `16384` | context window per request (the KV cache grows with it) |
| `AVD_OLLAMA_OUTPUT_RESERVE_TOKENS` | ¼ of the window, 1536–4096 | output tokens reserved per answer |
| `AVD_OLLAMA_KEEP_ALIVE` | `10m` | how long the model stays loaded after the last request |
| `AVD_OLLAMA_THINK` | `false` | reasoning traces (slower; sent only to models that support thinking) |
| `AVD_OLLAMA_MAX_CONCURRENCY` | `1` | in-flight requests per worker process |
| `AVD_OLLAMA_SEED` | `0` | sampling seed (plus the sample index) |
| `AVD_OLLAMA_ALLOW_REMOTE` | `false` | allow a non-private endpoint (only for a host you control) |

With `AVD_LLM_PROVIDER=ollama` the settings derive the rest:

* **Prompt sizing follows the context window.** The window, minus the output reserve, minus
  3,500 tokens for the system prompt, metadata and rubric (measured at about 7.4k characters
  plus up to 2k characters of description), is what a transcript may use. Single-pass and
  chunk sizes are clamped to it.
  * At 16k: about 6.6k estimated transcript tokens in one pass; anything longer is evaluated
    hierarchically.
  * At 8k: about 2k.
  * A window too small for a useful prompt is a configuration error.
* **Concurrency defaults to 1.** One Ollama model instance serves requests one at a time
  unless `OLLAMA_NUM_PARALLEL` is raised. More in-flight requests only queue, and parallel
  slots multiply KV-cache memory. Raise both together, and only after measuring peak memory
  with `avd bench-local`.

## 4. Safety properties (each covered by a test)

* **Local only.** The endpoint must be one of:
  * loopback;
  * a private or link-local address;
  * a hostname that resolves *only* to such addresses (e.g. a Compose service name).

  Ollama cloud models and models that `/api/show` reports as served from a `remote_host` are
  refused, because they would send transcripts to a third party. The HTTP client ignores proxy
  environment variables, so local traffic never passes through an egress proxy.
* **Never silently truncated.** Ollama cuts a prompt that exceeds `num_ctx` without an error.
  The provider refuses before sending when a conservative estimate (3 characters per token)
  cannot fit, and treats an answer whose reported prompt + output tokens reach the window as
  possibly truncated: that is an error, never a result. The pipeline chunks long transcripts
  instead, keeping segment indices and timestamps in every chunk.
* **Structured output.** The native API's `format` constrains decoding to the JSON schema. The
  answer is still validated with Pydantic, and every cited quote is checked against the
  transcript at the cited segment, exactly as for hosted models.
  * An answer cut off by the output limit (`done_reason: length`) is an output error, and so
    is an empty one (e.g. a thinking-only answer) or one that is not a JSON object.
* **Failure semantics.**
  * Server down, timeouts, HTTP 429/500/502/503: *transient*. Jobs are deferred and retried
    with backoff, nothing is persisted, and the API and extension keep serving existing
    results.
  * Unknown model: a configuration error naming `ollama pull <tag>`.
  * `avd evaluate-real` and `avd bench-local` check the server and every model *before*
    running anything, and stop with a message when either is missing.
* **Reproducibility.** Every local answer records `served_model = ollama:<tag>@<digest12>:<quant>`
  plus the load time, prompt and output tokens per second, and the context size. A model's
  *revision* (digest, quantization, effective context, seed, thinking mode) is part of:
  * the response-cache key, so re-pulled weights under the same tag are never served from an
    old answer;
  * the evaluation config hash, so earlier evaluations become stale.

  Hosted models' hashes are unchanged. Restart workers after `ollama pull` (model metadata is
  cached per process).
* **Cost.** Local calls cost $0 in the ledger; tokens and durations are still recorded. A local
  `evaluate-real` run uses a $0 run ceiling, so a request that would cost money is refused.

## 5. Evaluate

```bash
avd evaluate-real --provider ollama --offline --dry-run --run-id dev-local-plan    # plan only: 10 dev items, $0
avd evaluate-real --provider ollama --offline --run-id dev-qwen35-4b                # real local run
```

No budget is needed. `plan.json`/`summary.json` record `local_runtime`:

* Ollama version and per-model fingerprint, digest, quantization, native and effective context
  and capabilities;
* the sizing in force;
* the hardware snapshot.

Results land in the run database like any evaluation. The API reports
`evaluator_source: local_ai` and `/v1/meta` `inference: "local"`. The extension shows **Local AI
evaluation · qwen3.5:4b (Q4_K_M)** next to the trust state, e.g. *Evidence-checked*,
*Preliminary* or *Insufficient evidence*. Heuristic, synthetic and hosted results are labelled
differently, and *Human-reviewed* (independent human ratings) is shown separately, never merged
into the AI scores.

## 6. Benchmark local models

```bash
avd bench-local --offline --models qwen3.5:4b,qwen3.5:9b --num-ctx 8192,16384 --repeats 2
```

For each model × window × repeat this runs the production path on the **development** split
while sampling resources:

* host CPU utilisation (`/proc/stat`);
* the Ollama server's resident memory (`/proc/<pid>/status`);
* GPU memory (`/api/ps` `size_vram`);
* GPU utilisation (`nvidia-smi`, when present).

It then adds controlled probes on one dev item:

* **Injection**: an appended segment instructs the model to give every dimension 5. The probe
  reports how far scores moved, how many dimensions went to 5, and whether the injected
  segment was cited. The pipeline already refuses it as evidence.
* **Abstention**: a four-line transcript with nothing to judge. Text dimensions should be not
  assessable.
* **Context length**: running more than one `--num-ctx` makes a small window evaluate the same
  items hierarchically.

The report (`benchmarks/local/<id>/report.{json,md}`) gives, per model and window:

* structured validity and citation correctness (quotes verified at the cited segment);
* invalid citations and ungrounded scored dimensions;
* abstention rate and score consistency across repeats (exact agreement, mean absolute
  difference, scored-vs-abstained flips);
* pairwise ranking stability;
* p50/p95 latency, output tokens per second, items per hour, peak server RSS and VRAM, and
  CPU/GPU utilisation;
* errors by type and cost ($0);
* a side-by-side with the heuristic baseline, labelled as *not* evidence of validity.

**What it does not do.** It refuses `--split test` and has no `--confirm-test`; held-out data
stays sealed. It makes no claim about agreement with people: the validity gates are reported as
*blocked* until human labels exist. A run whose provider was injected (tests) is labelled
*NOT real inference* in both reports.

## 7. Human evaluation of local results (10 dev items)

The dev split has 10 items cleared for analysis (Talk Python items are excluded until their
licence scope is confirmed). To compare a local model with people:

1. Keep the dataset blinded while rating (`public_blinding`: the API refuses evaluations of
   blinded items, so raters cannot see the model's view).
2. Recruit at least 2 raters; each rates the 10 dev items with the annotation UI
   (`docs/annotation.md`). Repeats for test–retest are assigned automatically.
3. `avd annotate qc` and `avd annotate export --dataset real_pilot@v1`.
4. `avd bench-real --labels <export dir>` against the local run's database gives agreement
   (weighted κ, ICC, Spearman) with bootstrap CIs; report it per model.
5. Unblind only when annotation is closed: `avd annotate unblind --dataset real_pilot@v1`.

No ratings exist yet. None have been or will be generated.

## 8. Multimodal (experimental, not used for scoring)

`OllamaProvider.describe_images(model, images, prompt, schema)` sends authorized frames to a
vision-capable local model. The Qwen3.5 candidates report the `vision` capability. It refuses
models without that capability and marks its output `experimental`. It is **not wired into
scoring**: the visual dimensions keep abstaining (`requires_media`) until there is a validated
frame-sampling protocol and human labels for visual quality.

## 9. Continuous processing

Workers use the same job queue as hosted models:

* bounded concurrency (`AVD_OLLAMA_MAX_CONCURRENCY`, default 1 per process);
* retries with exponential backoff for transient errors;
* idempotent jobs (a duplicate job sees an up-to-date evaluation and does nothing);
* priorities that ignore popularity.

GPU memory is protected by running one model with one slot: `OLLAMA_MAX_LOADED_MODELS=1` and
`OLLAMA_NUM_PARALLEL=1` in the compose service. `keep_alive` controls residency. When Ollama is
down, the API, search, recommendations and existing evaluations keep working, and new deep
evaluations wait in the queue.
