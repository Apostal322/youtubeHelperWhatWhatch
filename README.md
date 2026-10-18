# AI Video Discovery

An open-source system that discovers, evaluates and recommends YouTube videos by the **quality of
their content, not their popularity**.

> A video with 100 views can be better than a video with 10 million views.

Views, likes, subscriber counts and channel identity **never** enter the intrinsic quality
evaluation — not as features, not in prompts, not through priorities. This is enforced in code
(the evaluator's input type has no such fields) and verified by tests and by the benchmark
(inflating popularity ×1000 must change nothing).

| Component | What it does |
|---|---|
| **Backend** (`backend/`, Python/FastAPI) | Discovery → content-availability → genre → multidimensional evaluation → validation → persistence → recommendation. Durable job queue + Celery workers, budgets, quotas, versioning. |
| **Evaluation engine** (`backend/avd/evaluation/`) | 14 independent dimensions in 3 groups (content / presentation / technical), explicit abstention, evidence grounded in transcript segments, genre-specific transparent aggregation. |
| **Recommender** (`backend/avd/recsys/`) | Vector retrieval → transparent linear ranker → MMR diversification. Intrinsic quality, relevance and rank score are stored and returned separately. LTR interface + pairwise trainer. |
| **Browser extension** (`extension/`, React/TS/MV3) | Discover feed, per-video quality breakdown with evidence, topic search, preferences, feedback and incorrect-evaluation reports. Reads precomputed evaluations; never runs inference. |
| **Benchmark harness** (`backend/avd/bench/`, `benchmarks/`) | Agreement with labels, pairwise accuracy, inter-rater agreement, calibration (ECE), popularity invariance, recommendation nDCG/diversity/exposure, cost per video. |

## Status — read this first

* **No AI evaluator has been run against a live model in this repository.** The LLM path
  (`AVD_LLM_PROVIDER=anthropic`) is implemented and tested with scripted responses. Without
  credentials, scores come from a keyword/structure *heuristic baseline*: the API reports
  `evaluation_mode: heuristic_baseline` and the extension header says *Heuristic mode (not AI)*.
* **Real, licensed content is in the repo; real human labels are not.** `real_pilot_v1` holds 29
  podcast episodes (Changelog CC BY-SA 4.0, Talk Python GPL-3.0) fetched at pinned commits with
  checksums. A blind annotation workflow exists, but nobody has annotated yet. **There is no
  evidence yet about accuracy on real content.**
* **Synthetic fixtures** (20 videos written for this project) still back the zero-key demo. They are
  labelled `synthetic_demo` everywhere and do not exist on YouTube. Their labels are the fixture
  author's design intent, so benchmark numbers on them test the harness only.
* Phase 2's audit, acceptance criteria and measured results are in
  [`docs/phase2.md`](docs/phase2.md).

## Quick start

### 1. Zero-infrastructure demo (SQLite, no keys, ~1 minute)

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
avd demo                       # migrate + seed synthetic fixtures + run Stage A/B + summary table
AVD_ADMIN_TOKEN=dev avd serve  # API on http://localhost:8000 (docs at /docs)
```

### 2. Full stack (PostgreSQL + pgvector, Redis, API, Celery worker + beat)

```bash
cp .env.example .env           # optional: add API keys here
docker compose up -d --build
docker compose --profile init run --rm init   # migrate + seed demo fixtures; workers process them
curl localhost:8000/v1/search?q=hash%20tables
```

### 3. Browser extension

```bash
cd extension && npm ci && npm run build
```
Chrome → `chrome://extensions` → enable *Developer mode* → *Load unpacked* → select `extension/dist`.
The extension talks to `http://localhost:8000` by default (configurable under *Preferences*).

### 4. Licensed pilot data and annotation

```bash
cd backend
avd dataset import --evaluate && avd worker --once   # fetch at pinned commits, verify, evaluate
avd annotate create-rater r1                         # token printed once; rater UI at /annotate
avd bench-real                                       # dev split; human-grounded metrics need labels
```
See [`docs/annotation.md`](docs/annotation.md) and the dataset datasheet in
[`benchmarks/datasets/real_pilot_v1/`](benchmarks/datasets/real_pilot_v1/).

### 5. Real evaluations

```bash
export AVD_LLM_PROVIDER=anthropic AVD_ANTHROPIC_API_KEY=sk-ant-...   # model: claude-opus-5-5 by default
export AVD_YOUTUBE_API_KEY=AIza...                                    # discovery + metadata
avd discover --n 4 && avd worker --once
```
Videos are only *evaluated* when authorized content exists (creator-authorized captions,
user-submitted transcripts with an attestation, licensed datasets). Discovery alone yields catalog
entries with metadata and an explicit "content unavailable" status — see
[`docs/permissions.md`](docs/permissions.md).

## Checks

```bash
cd backend && ruff check avd tests && mypy avd && pytest -q          # SQLite only
AVD_TEST_DATABASE_URL=postgresql+psycopg://... AVD_TEST_REDIS_URL=redis://... pytest -q   # + pgvector/Redis
avd bench                                                             # benchmark report → benchmarks/reports/
cd extension && npm run typecheck && npm test && npm run build
```

## Documentation

* [Architecture](docs/architecture.md) — components, pipeline stages, data model, job system, scaling.
* [Evaluation framework & research](docs/evaluation.md) — dimensions, rubric, aggregation, confidence,
  validation, benchmark design, metrics, results, **biases and limitations**.
* [Permissions & content access](docs/permissions.md) — what uses which API, what needs which grant.
* [Annotation protocol](docs/annotation.md) — blind human rating, QC, adjudication, export.
* [Security review](docs/security.md) — trust boundaries, controls, audits, known gaps.
* [Phase 2 record](docs/phase2.md) — audit, acceptance criteria, measured results, blockers.
* [API reference](docs/api.md)
* [Roadmap](docs/roadmap.md) — the path to large-scale continuous analysis.

## Repository layout

```
backend/            FastAPI app, evaluation engine, workers, recommender, benchmark harness
  avd/migrations/   Alembic migrations (shipped inside the package)
  tests/            pytest suite (SQLite always; PostgreSQL/pgvector + Redis when configured)
extension/          Chrome MV3 extension (React + TypeScript + Vite)
data/fixtures/      Synthetic, CC0 demo videos with original transcripts
benchmarks/         Datasets (synthetic_v1) and committed baseline reports
docs/               Design and research documentation
```

## License

Code: Apache-2.0. Synthetic fixtures and benchmark data: CC0-1.0.
