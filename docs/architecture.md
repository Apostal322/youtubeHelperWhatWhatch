# Architecture

## 1. Overview

```mermaid
flowchart LR
  subgraph Discovery & ingestion
    YT[YouTube Data API v3<br/>search.list · videos.list · channels.list] --> D[discover job]
    D --> I[ingest_metadata job<br/>batched 50 ids/call]
  end
  subgraph Content sources [Authorized content (pluggable)]
    CC[Creator OAuth captions]
    UT[User transcripts + attestation]
    LD[Licensed datasets]
    FX[Synthetic fixtures]
  end
  I --> A[Stage A<br/>genre · topics · metadata embedding<br/>content availability]
  CC & UT & LD & FX --> A
  A -->|transcript available| B[Stage B<br/>deep evaluation]
  B -->|low confidence / fact-critical<br/>and budget allows| C[Stage C<br/>stronger model · self-consistency<br/>retrieval fact check]
  B & C --> DB[(PostgreSQL + pgvector)]
  DB --> R[Recommender<br/>retrieve · rank · diversify]
  DB --> S[Search]
  R & S --> API[FastAPI]
  API --> EXT[Chrome extension]
```

Two processes share one code base: the **API** (FastAPI) and **workers** (inline worker or Celery).
State lives in PostgreSQL; Redis is the Celery broker and the shared rate-limiter store.

## 2. Pipeline stages

| Stage | Job type | Work | External cost |
|---|---|---|---|
| Discovery | `discover` | Stratified seed queries (genre × language), `order=date`, page-token cursors | 100 quota units / search |
| Metadata | `ingest_metadata` | `videos.list` batched by 50, `channels.list` for stale channels; popularity stored in `video_stats` only | 1 unit / 50 videos |
| **A — lightweight** | `stage_a` | Genre from metadata, topic keywords, metadata embedding, content-availability check across providers | none |
| **B — deep** | `stage_b` | Acquire content, genre refined with transcript, evaluation pipeline, validation, persistence, transcript-keyword embedding, escalation decision | 1 LLM call (or $0 heuristic) |
| **C — selective** | `stage_c` | Stage-C model with *n*-sample self-consistency, retrieval fact checking if a licensed corpus exists | *n* LLM calls + fact check |
| Maintenance | `reevaluate_stale` | Re-queue videos whose evaluation predates `FRAMEWORK_VERSION` | none |

Stage C runs only when Stage B confidence is low, or the genre is fact-critical *and* a reference
corpus is configured, **and** at least `AVD_STAGE_C_MIN_REMAINING_BUDGET_FRACTION` of today's budget
remains. The decision and reason are stored in the job result.

## 3. Evaluation pipeline (pure, testable)

`EvaluationInput` (popularity-blind, fingerprinted) → evaluators (each judges a subset of
dimensions; crashes become `failed`, transient errors propagate for retry) → **merge** (genre
profile marks non-applicable dimensions; missing modalities → `requires_media`) → **validation**
(evidence grounding, degenerate-output checks) → **confidence adjustment** → **aggregation** →
`EvaluationResult`. See [evaluation.md](evaluation.md) for the details.

Evaluators shipped:

| Evaluator | Dimensions | Mode |
|---|---|---|
| `HeuristicTextEvaluator` | 10 transcript dimensions (abstains on originality, factual accuracy) | synthetic demo baseline |
| `LLMRubricEvaluator` | 10 transcript dimensions in one structured call; `samples>1` for self-consistency | LLM |
| `AudioSignalEvaluator` | `audio_quality` from authorized PCM WAV (level, noise floor/SNR, clipping) | signal processing |
| `FactualVerificationEvaluator` | `factual_accuracy` via claim extraction → licensed-corpus retrieval → grounded verdicts | LLM + retrieval (Stage C) |

Visual dimensions have no evaluator yet and are reported as `requires_media`.

## 4. Data model

| Table | Purpose |
|---|---|
| `channels`, `videos` | Catalog. `channels.size_bucket` is used **only** for coverage monitoring/fair prioritisation. |
| `video_stats` | Popularity snapshots — isolated; never read by evaluation code. |
| `content_sources` | Authorized content with `source_type`, `authorization_basis`, `license`, content hash, revocable `status`. |
| `evaluations` | Immutable, versioned (`framework_version`, `stage`, `mode`, `models`, `input_fingerprint`); `is_current` flag; history kept. |
| `dimension_scores`, `evidence` | Per-dimension status/score/confidence/justification/method and verified evidence with transcript timestamps. |
| `users`, `user_preferences` | Anonymous users (hashed bearer tokens), interests (+ embedding), genre/dimension priorities. |
| `recommendation_impressions` | What was shown, with **intrinsic quality, relevance and rank score stored separately** plus ranking features and ranker version (LTR training data). |
| `feedback` | Usefulness ratings, not-interested, incorrect-evaluation reports (dimension, suggested score, comment). |
| `pipeline_jobs` | Durable job state (see §5). |
| `cost_ledger` | Every LLM call (tokens, USD, cached?) and YouTube API call (quota units). |
| `llm_cache` | Content-addressed cache of LLM responses (reproducibility, zero-cost re-runs). |
| `discovery_queries` | Seed-query cursors for incremental discovery. |

Migrations: `backend/avd/migrations` (Alembic; the pgvector extension and HNSW index are created on
PostgreSQL only; `alembic check` runs in CI to detect model/migration drift).

## 5. Job system

`pipeline_jobs` is the source of truth; executors are interchangeable:

* **Inline** (`avd worker`): claims and executes jobs in-process. No Redis needed.
* **Celery**: `avd.dispatch` (beat, every 5 s) claims due jobs and fans them out to
  `avd.execute_job` (`light` queue) / `avd.execute_deep_job` (`deep` queue, LLM work) so the two
  pools scale independently. `acks_late` + leases → at-least-once; handlers are idempotent.

| Concern | Mechanism |
|---|---|
| Deduplication | Unique `dedup_key`, e.g. `stage_b:<video>:<framework_version>:<content-hash>` |
| Incremental processing | Skip when `input_fingerprint` + framework version + stage already evaluated |
| Priority | Coverage-aware, popularity-blind (see `workers/prioritizer.py`) |
| Concurrency | `SELECT … FOR UPDATE SKIP LOCKED` on PostgreSQL |
| Failure recovery | Lease expiry → job re-claimed; final-attempt expiry → `dead` |
| Retries | Exponential backoff + jitter; non-retryable errors go straight to `dead` |
| Budget | Jobs with an estimated cost are claimed only if today's remaining LLM budget covers them; otherwise deferred (not failed) |
| YouTube quota | Check-then-record guard before every call; exhaustion defers the job to the next Pacific-time midnight |
| Rate limiting | Token bucket per provider/model — in-process or shared in Redis (Lua, atomic) |
| Batching | `videos.list`/`channels.list` 50 ids per call; one LLM call per video for all text dimensions |
| Caching | LLM response cache; content dedup by hash; metadata TTL for channel stats |
| Cost tracking | `cost_ledger`, `evaluations.cost_usd`, `/v1/admin/costs` |
| Versioning | `FRAMEWORK_VERSION` on every evaluation; `reevaluate_stale` backfills |

## 6. Recommendation

1. **Retrieval** — per-interest vector search (pgvector HNSW cosine; exact numpy scan on SQLite)
   plus top-quality candidates in preferred genres; filters for language, duration, synthetic
   content, "not interested".
2. **Features** — calibrated relevance, *personal* quality (genre-profile weights × the user's
   dimension priorities), confidence, genre preference, depth match, novelty, feedback affinity,
   optional small-creator boost (user opt-in, default 0).
3. **Eligibility** — with stated interests, items below the relevance floor are not topical; they
   may only fill a labelled exploration quota (≤ 20 %).
4. **Ranking** — `LinearRanker` (weights in `recsys/ranker.py`) with per-feature contributions in
   every response. `train_pairwise_ranker` learns weights from impressions + feedback
   (`avd train-ranker`) behind the same `Ranker` protocol.
5. **Diversification** — MMR over embeddings + max 2 items per channel.

## 7. Deliberate MVP simplifications and upgrade paths

| Simplification | Why | Upgrade path |
|---|---|---|
| Hashing embedder by default | Works offline, deterministic, no model download | `AVD_EMBEDDING_PROVIDER=sentence_transformers` (same 384-dim column); re-embed via `reevaluate_stale` |
| Single LLM call for all text dimensions | Cost | Split per group / per-dimension specialist evaluators; the `Evaluator` protocol is unchanged |
| No visual evaluator | Requires authorized video + multimodal model | Add an evaluator requiring `Modality.VIDEO`; dimensions already defined |
| Keyword genre classifier | Transparent, free | Embedding or LLM classifier behind `GenreClassifier` |
| Anonymous bearer tokens | Extension-only auth | OAuth / account linking; the `current_user` dependency is the only touch point |
| Chunked evaluation of very long transcripts not implemented | Avoid silent truncation | Transcripts above the limit raise and the job fails visibly; map-reduce evaluation is a roadmap item |
| Inline worker for local runs | No Redis needed for demos | Celery is already implemented for multi-process deployments |
