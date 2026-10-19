# Phase 3.1 engineering report: fully local AI evaluation

Date: 2026-10-19. Branch `claude/phase3-1-local-ollama`, stacked on Phase 3
(`claude/eager-galileo-cn31qb`, PR #2, not yet merged).

## Bottom line

* **Built and tested:**
  * a local-only Ollama backend for the AI evaluator;
  * hardware detection and model selection;
  * context-aware prompt sizing (chunk, never truncate);
  * model-revision pinning;
  * a local benchmark;
  * extension labelling;
  * an optional Compose service.

  The backend suite has 395 tests passing, including 60 new ones for local inference, all run
  against a protocol mock of the Ollama API.
* **Not done: real local inference.** No model was loaded and no transcript was evaluated by a
  local model. The development container's egress policy rejects:
  * `ollama.com`, `registry.ollama.ai` and `huggingface.co` (connection refused by the proxy);
  * GitHub release downloads (HTTP 403).

  So neither Ollama nor any model weights could be installed. This is a network-policy limit,
  not a hardware limit: `qwen3.5:4b` at q4 (~3.5 GB estimated) fits this machine's 15.7 GB RAM.
* **No result here is a model result.** There are no local benchmark numbers, no quality
  judgements, and no human ratings. None were simulated. The single live test is skipped and
  reported as skipped.

### To produce the first real local results

Allow `ollama.com`, `registry.ollama.ai` and `huggingface.co` in the environment's network
settings, or use any machine with Ollama. Then:

```bash
cd backend && avd local doctor
curl -fsSL https://ollama.com/install.sh | sh && ollama serve &
avd local pull qwen3.5:4b --approve-gb 3
AVD_TEST_OLLAMA_URL=http://localhost:11434 AVD_TEST_OLLAMA_MODEL=qwen3.5:4b pytest -m live
avd evaluate-real --provider ollama --offline --run-id dev-qwen35-4b
avd bench-local --offline --models qwen3.5:4b,qwen3.5:9b --num-ctx 8192,16384 --repeats 2
```

## 1. Inspection

* The repository was at Phase 3 (PR #2 open, CI green, not merged). This work branches from its
  head (49ba400) so PR #2's review history is untouched, and targets that branch.
* The baseline before changes was 335 backend tests passing.

### Hardware (measured by `avd local doctor`)

| | |
|---|---|
| OS | Ubuntu 24.04.5 LTS, x86_64 |
| CPU | Intel Xeon @ 2.10 GHz, 4 logical cores; AVX2, AVX-512 (F, VNNI, BF16), AMX |
| RAM | 15.7 GB, no swap |
| GPU | none (no `/dev/nvidia*`, `/dev/kfd`, `/dev/dri`; no `nvidia-smi`/`rocm-smi`) |
| Disk | 26 GB free |
| Ollama | not installed; server unreachable |

`doctor` recommends `qwen3.5:4b` (CPU: fewest active parameters first) and proposes
`qwen3.5:4b` and `qwen3.5:9b` for benchmarking. The 27B and 35B-A3B candidates do not fit.

## 2. What was built

| Area | Implementation |
|---|---|
| Provider | `avd/llm/ollama.py`: native `/api/chat` with `format` = JSON schema, `num_ctx`, `num_predict`, `keep_alive`, `think` (only for models reporting the capability), `seed` + sample index; token counts and durations; `served_model = ollama:<tag>@<digest12>:<quant>`; per-call runtime (load s, prompt/output tokens/s, window) |
| Local only | Loopback, private or link-local endpoints, or names resolving only to them; cloud models and `remote_host` models refused; proxy env ignored |
| Failure semantics | Server down, timeouts and 429/5xx are transient (jobs deferred); a missing model is a configuration error naming `ollama pull`; cut-off, empty or non-object output is an output error; context overflow is refused before sending, and server-reported exhaustion is an error |
| Configuration | `AVD_LLM_PROVIDER=ollama` and `AVD_OLLAMA_*`. Stage models follow `ollama_model` unless set explicitly; concurrency defaults to 1; single-pass and chunk sizes are clamped to the window minus the output reserve and the measured prompt overhead |
| Cost | Local models are priced at $0; a local `evaluate-real` run has a $0 ceiling, so any priced call is refused |
| Revision pinning | Digest, quantization, effective context, seed and thinking mode enter the cache key and the evaluator config hash, so re-pulled weights re-evaluate; hosted hashes are unchanged |
| Runner | `avd evaluate-real --provider ollama`: no budget needed; pre-flight checks the server and model; `local_runtime` (version, fingerprints, sizing, hardware) recorded in `plan.json`/`summary.json` |
| Hardware | `avd/llm/local.py`: Linux/macOS/Windows detection, NVIDIA/AMD/Apple/CPU; candidate table with memory/download estimates; recommendation; `ResourceSampler` (CPU %, Ollama RSS, VRAM via `/api/ps`, GPU % via `nvidia-smi`) |
| Downloads | `avd local pull` requires `--approve-gb` ≥ the estimated size, and refuses unknown sizes, cloud models and full disks |
| Benchmark | `avd bench-local` (`avd/bench/local.py`): model × window × repeat on dev only; validity, citation correctness, abstention, consistency, pairwise stability, latency, throughput, resources; injection and abstention probes; heuristic side by side; validity gates reported blocked; an injected transport is labelled *NOT real inference* |
| API / extension | `evaluator_source` (`local_ai` / `hosted_ai` / `heuristic` / `synthetic`) on evaluations and summaries; `human_reviews` count; `/v1/meta.inference`. The extension shows *Local AI evaluation · qwen3.5:4b (Q4_K_M)* beside the trust state, and *Human-reviewed* separately |
| Multimodal | `describe_images` (experimental, vision-capable models only); **not used for scoring**; visual dimensions still abstain |
| Deployment | Compose `ollama` service (profile `local-ai`) on `127.0.0.1:11434`, one model, one slot |

## 3. Verification

All of the following run in CI without a GPU, a model or a server:

* backend: `ruff`, `mypy avd`, `pytest` (395 passed, 1 skipped = the live test), on SQLite
  and on PostgreSQL + pgvector + Redis;
* extension: `tsc`, `vitest` (32 passed), `vite build`.

New tests (`tests/test_local_ollama.py`, against `tests/ollama_mock.py`):

* **Client:**
  * the native request shape: schema `format`, `num_ctx`, seed, temperature, `think`
    gating;
  * the fingerprint and effective context;
  * no proxy;
  * `/v1` stripping.
* **Failures:**
  * an unavailable server and server 5xx are transient;
  * a missing model names the pull command;
  * invalid, non-object or empty JSON is an output error, and so is cut-off output;
  * context overflow is refused before sending, with no request made; server-reported
    exhaustion is an error;
  * a timeout is transient and recovers through the retry wrapper.
* **Local only:**
  * cloud and remotely served models are refused;
  * loopback, private and IPv6 addresses are accepted, and so are service names that
    resolve to private addresses;
  * public hosts need an explicit opt-in.
* **Settings:**
  * stage models follow `ollama_model`, and an explicit one is kept;
  * concurrency is 1 and the output reserve adapts to the window;
  * sizing is clamped to the window, and a window that is too small is rejected;
  * hosted providers are unaffected.
* **Pricing:** local models cost $0 and hosted ones still cost.
* **Hardware:**
  * detection on this machine;
  * recommendations for CPU, NVIDIA, AMD, Apple, low disk, and nothing fitting;
  * estimates scale with quantization.
* **Pull:** refused without size approval, for unknown sizes and for cloud models; allowed
  with approval.
* **Evaluation path:**
  * end-to-end `evaluate-real` through the mock (3 items, $0, all quotes verified, fingerprint
    and hardware recorded, every request fits its window);
  * results labelled `local_ai` for the API;
  * resume makes no duplicate calls;
  * long transcripts (~35k tokens) are chunked to fit 8k, 16k and 32k windows: map + reduce,
    never truncated;
  * a down server or a missing model refuses before anything runs;
  * `--provider` re-validates settings;
  * popularity fields never reach the model;
  * a Stage B job is deferred (not failed) while Ollama is down.
* **Revision:** new weights under the same tag bypass the cache and change the config hash; a
  down server bypasses the cache; hosted hashes are unchanged.
* **Benchmark:**
  * the report is labelled injected;
  * held-out data is refused;
  * a missing model runs nothing and writes no report;
  * the consistency metrics detect disagreement.
* **Vision:** images go only to vision models; visual dimensions still abstain.
* **API:** `/v1/meta` reports `inference: local`.
* **Extension:** the local label with model and quantization; human ratings shown separately;
  heuristic and synthetic never called AI; `modelLabel` formatting.
* **Live (skipped here):** structured output from a real local model, when
  `AVD_TEST_OLLAMA_URL` and `AVD_TEST_OLLAMA_MODEL` point at a running server.

The mock's answers score 3 everywhere and cite the first segment shown. They test plumbing,
not quality, and nothing derived from them is reported as a model result.

### CLI checks on this machine (no server)

* `avd local doctor` printed the hardware table above.
* `avd evaluate-real --provider ollama --dry-run` planned 10 dev items at $0 expected and $0
  worst case.
* The live `evaluate-real` and `bench-local` refused with "Ollama unreachable … No local
  inference was run".
* `avd local pull qwen3.5:4b --approve-gb 1` refused ("~2.4 GB … nothing downloaded").

## 4. Status against the Phase 3.1 brief

| § | Item | Status |
|---|---|---|
| 1 | Inspect repo, tests; branch preserving PR #2 | done |
| 2 | Hardware detection, model choice, approval before downloads | done; model *benchmarking* blocked (no model) |
| 3 | Ollama first-class backend; local-only; validation | done (mock-verified); live behaviour unverified |
| 4 | Real evaluation of the authorized dataset | **blocked**: no model can be installed; the runner is ready (10 dev items, $0) |
| 5 | Long transcripts: chunking, timestamps, no truncation, context lengths | done (mock-verified); measurement across windows awaits a model |
| 6 | Local model benchmark | implemented; **no measurements** |
| 7 | Human evaluation workflow; 10-dev experiment | workflow verified in Phase 3; protocol written; no ratings exist |
| 8 | Multimodal preparation | interface only, experimental, not scoring |
| 9 | Continuous processing | done: bounded concurrency, retries, deferral when down, keep-alive, fingerprints, revision-aware caching |
| 10 | Extension labels | done |
| 11 | Scientific benchmarking vs heuristic; gates | implemented; validity gates blocked (no labels); no results |
| 12 | Tests; CI without GPU/LLM; optional live tests | done; the live test is skipped here, not passed |
| 13 | Deliverables, PR (not merged) | done |

## 5. Known limitations

* **Candidate tags and sizes** (`qwen3.5:4b`, `:9b`, `:27b`, `:35b-a3b`) could not be checked
  against the Ollama library from here. Sizes are estimates until a model is installed.
* **Prompt fit.** The rubric prompt was written and checked for hosted models; whether a
  4B/9B model follows it well is unknown. At an 8k window the 1.5–2k output reserve is tight
  for 10 dimensions. Overruns fail visibly as "cut off" and are reported by the benchmark.
* **Stale model metadata.** Model metadata is cached per worker process, so restart workers
  after `ollama pull`. Otherwise new weights are noticed only after the restart.
* **Unpinned Compose image.** The Compose image tag defaults to `latest` because no release
  could be verified from here. Pin `OLLAMA_IMAGE_TAG`; every run records the server version
  anyway.
* **Resource sampling** reads `/proc` (Linux). On macOS and Windows only `/api/ps` VRAM is
  sampled.
