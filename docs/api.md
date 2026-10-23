# API reference

Interactive OpenAPI docs are served at `/docs`. All responses are JSON. Authenticated endpoints use
`Authorization: Bearer <token>` (obtained from `POST /v1/users`); admin endpoints use
`X-Admin-Token` (`AVD_ADMIN_TOKEN`; disabled when unset).

Popularity statistics are never returned by public endpoints.

### Conventions

* **Request ids.** Every response carries `X-Request-ID`. A safe inbound value is reused, otherwise
  one is generated. The same id is the correlation id in logs, on jobs enqueued by the request and
  on model calls they make.
* **Errors.** Every error has one shape:
  `{"detail": "...", "error": {"code": "not_found", "message": "...", "request_id": "..."}}`.
  Codes include `bad_request`, `unauthenticated`, `forbidden`, `not_found`, `conflict`,
  `payload_too_large`, `unsupported_media_type`, `validation_error`, `rate_limited`,
  `unavailable` and `internal_error`. A 500 never includes internals.
* **Rate limits.** Token bucket per caller (hashed bearer/admin token, else client IP):
  `AVD_API_RATE_LIMIT_PER_MINUTE` (300) with burst `AVD_API_RATE_LIMIT_BURST` (60). Exceeding it
  returns 429 with `Retry-After`. The limiter is per process (see `docs/security.md`).
* **Security headers.** `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`; `/annotate` has a strict CSP.

### Phase 3 behaviours

* **Blind annotation.** While a dataset's annotation is open, the public detail, evaluation,
  history and claims endpoints return **403** (`forbidden`) for its items, and search, listings
  and recommendations skip them.
* **Rights.** For sources whose publisher has not confirmed public quoting, evaluation and
  claims responses carry `quotes_withheld: true`. Quote text is replaced, and positions and
  verification status are kept.
* **Feedback.** An exact duplicate returns **200** `{"id", "status": "duplicate"}` instead of a
  new event. At most 500 events per user per day (429).
* **Rate limits.** Every request draws from its caller's bucket and its client IP's bucket
  (`AVD_API_RATE_LIMIT_PER_IP_PER_MINUTE`, `AVD_API_RATE_LIMIT_IP_BURST`). The buckets are
  shared through Redis when configured.
* **Evaluation object.** `coverage_report.support_audit` lists evidence-support flags, and any
  flag keeps an evaluation out of `grounded`. Fact-check claims add claim type `hypothesis` and
  verdict `outdated`. The video card adds `genre_status`.
* **`/v1/status`** adds `media_processing.ffmpeg_available`,
  `media_processing.os_resource_limits` and `rate_limit_backend`.

## Public

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + demo-mode flag |
| GET | `/v1/meta` | Framework version, `evaluation_mode` (`heuristic_baseline` when no AI evaluator is configured, else `llm`) + disclaimer, rubric (questions, anchors, required modalities), genre profiles (weights) |
| GET | `/v1/status` | Operational status without secrets: database reachability, queue counts, catalog counts (`videos`, `synthetic`, `evaluated`, `hidden_for_blind_annotation`), evaluation mode, provider, whether LLM credentials are configured (yes/no only), versions |
| GET | `/metrics` | Prometheus text (HTTP counters, rate-limited requests, jobs by status, today's LLM spend). Admin token required unless `AVD_METRICS_PUBLIC=true` |
| GET | `/v1/videos?genre=&evaluated=true&limit=&offset=` | Catalog listing with quality summaries |
| GET | `/v1/videos/{id}` | Video card, available modalities, content sources (type, authorization basis, license, attribution, provenance URL), current quality summary |
| GET | `/v1/videos/{id}/evaluation` | Full current evaluation (below) — 404 with the video status if not evaluated |
| GET | `/v1/videos/{id}/evaluations` | Evaluation history (all framework versions/stages) |
| GET | `/v1/videos/{id}/claims` | Claims from the current evaluation(s): quote, claim, `claim_type`, `verdict` (`supported`, `contradicted`, `mixed`, `unverifiable`, `insufficient_evidence`, `not_applicable`, or `unchecked` = extracted, not verified), `domain_risk`, cited reference passages with licences, explanation |
| GET | `/v1/search?q=&genre=&language=&min_quality=&include_synthetic=` | Topic search over analyzed videos; `match` = semantic similarity, title match, score. Videos below the embedder's relevance floor without a title match are left out, so an unmatched query returns an empty list. The default `hashing` embedder is a keyword-hash fallback whose collisions can still surface unrelated videos; `AVD_EMBEDDING_PROVIDER=onnx_minilm` is much more precise |

### Evaluation object (abridged)

```json
{
  "evaluation_id": "…", "video_id": "syn_edu_001", "stage": "B",
  "framework_version": "2026.10.0", "mode": "synthetic_demo", "synthetic": true,
  "disclaimer": "Synthetic demo evaluation … NOT an AI assessment …",
  "models": ["heuristic-baseline", "audio_signal_metrics_v1"],
  "genre": "education", "genre_confidence": 0.62, "status": "complete",
  "overall": {"score": 3.18, "confidence": 0.37, "coverage": 0.72, "abstain_reason": null,
              "method": "genre_weighted_mean_renormalized_v1", "notes": ["Group scores diverge strongly; …"]},
  "groups": {"content": {"score": 3.17, "confidence": 0.41, "coverage": 0.65, "scored_dimensions": ["…"]},
             "presentation": {…}, "technical": {"score": 2.0, …}},
  "aggregation_profile": {"genre": "education", "dimension_weights": {…}, "group_weights": {…}},
  "contributions": [{"dimension": "depth", "effective_weight": 0.12, "score": 2.3, "contribution": 0.28}],
  "dimensions": [
    {"dimension": "depth", "group": "content", "status": "scored", "score": 2.34, "confidence": 0.37,
     "justification": "…", "method": "heuristic_baseline_v1", "model": "heuristic-baseline",
     "missing_inputs": [],
     "evidence": [{"segment_index": 2, "start_seconds": 11.4, "end_seconds": 17.1,
                   "quote": "…", "verified": true, "content_source_id": "…"}]},
    {"dimension": "originality", "status": "insufficient_evidence", "score": null, …},
    {"dimension": "visual_quality", "status": "requires_media", "missing_inputs": ["video"], …}
  ],
  "validation": {"passed": true, "warnings": [], "evidence_claimed": 0, "evidence_verified": 0},
  "missing_inputs": ["video"], "cost_usd": 0.0
}
```

`status` per dimension: `scored`, `not_applicable` (genre profile weight 0), `insufficient_evidence`,
`requires_media` (an authorized modality is missing), `failed` (evaluator error or failed
validation). Only `scored` dimensions carry a score; an unavailable dimension is never 0.

Phase 2 additions to the evaluation object:

* `mode`: `llm`, `heuristic_baseline` (heuristic on real content) or `synthetic_demo`.
* `assessment_state` + `assessment_state_reasons`: `grounded` (model judgements with verified
  evidence, validation passed, no review flags — still not human-validated), `preliminary`,
  `insufficient_evidence`, `failed`, `heuristic_baseline`, `synthetic_demo`.
* Evidence items carry `kind` (`supporting` or `counter`).
* `needs_review` / `review_reasons` (e.g. a contradicted high-stakes claim, self-consistency
  disagreement), `versions` (framework, rubric, prompt, aggregation, preprocessing, injection
  detector), `config_hash`, and a public `coverage_report` (chunking, map-reduce, per-evaluator
  summaries).
* `aggregation_profile` is the profile actually used: for a `mixed` genre, a blend of the two
  leading genres; for `ambiguous` or `unknown`, the neutral `other` profile. The video card's
  `genre_status` says which case applies (`confident`, `mixed`, `ambiguous`, `unknown`,
  `overridden`).

## User

| Method | Path | Description |
|---|---|---|
| POST | `/v1/users` | Create an anonymous user → `{user_id, token}` (token shown once; stored hashed) |
| GET / PUT | `/v1/users/me/preferences` | Interests, intent (`any`, `learn`, `stay_informed`, `be_entertained`, `deep_dive`; supplies genre priorities when no genre weights are set), genre weights (0–1; v2 genres inherit their parent's weight), dimension priority multipliers (0–3), depth (`any/intro/intermediate/advanced`), languages, min/max duration (minutes), diversity, min quality, small-creator boost (0–0.3), include synthetic |
| GET | `/v1/recommendations?limit=` | Personalised feed (below) |
| POST | `/v1/feedback` | `{video_id, kind, impression_id?, evaluation_id?, dimension?, suggested_score?, comment?}`; `kind` ∈ `recommendation_useful`, `recommendation_not_useful`, `not_interested`, `evaluation_incorrect` (requires `evaluation_id`) |
| POST | `/v1/videos/{id}/transcripts` | `{format: vtt|srt|text|json, content, language?, attestation}` → stored as `user_provided_transcript` and queued (202) |
| PUT | `/v1/videos/{id}/media` | Authorized audio/video as the raw body (`Content-Type` from the allowlist), `X-Content-Attestation` header stating the right to submit it. Streamed size cap (`AVD_MAX_MEDIA_UPLOAD_MB`, 413 beyond), ffprobe validation (422 if not media), content-addressed storage; queued (202). 503 when ffmpeg is not installed. Nothing is fetched from YouTube |
| DELETE | `/v1/users/me` | Delete the account: preferences, feedback, impressions; submitted transcripts are revoked and their text erased; derived evaluations stop being current. Returns counts |

### Recommendation item

```json
{
  "impression_id": "…", "position": 0,
  "video": {"id": "…", "title": "…", "channel_title": "…", "genre": "science", "is_synthetic": true, "url": null},
  "intrinsic_quality": {"overall": 3.35, "overall_confidence": 0.36, "groups": {…}, "synthetic": true},
  "relevance": {"similarity": 0.21, "normalized": 0.56, "matched_interest": "astronomy and space weather"},
  "ranking": {"score": 0.61, "ranker_version": "linear_v1", "personal_quality": 3.1,
              "features": {"relevance": 0.56, "quality": 0.53, …},
              "contributions": {"relevance": 0.224, "quality": 0.159, …},
              "diversified": false, "exploration": false},
  "explanations": ["Matches your interest “astronomy and space weather”.", "Strongest: …"]
}
```

The response wraps items as
`{request_id, ranker_version, selection_policy_version, items, eligibility, empty_reason?, notes}`.

* `selection_policy_version` (Phase 3.4) is a digest of everything static that decides which
  candidates can appear:
  * retrieval sizes;
  * the embedder and document recipe;
  * the relevance floor and ceiling;
  * the ranker's version, features and weights;
  * the per-channel cap;
  * diversification and the exploration quota.

  Each impression stores it, together with the per-request user settings and per-item flags
  (`selection_context`). Each distinct configuration is stored once in
  `recommendation_policies`. Impressions logged before this have `NULL`: their policy is unknown,
  not guessed.
* `empty_reason`, set only when there are no items, is one of:
  * `empty_catalog`: nothing has been imported and analyzed;
  * `hidden_for_blind_annotation`: imported items are withheld while blind human annotation is
    open; `avd annotate unblind` lifts it;
  * `no_match`: the user's filters exclude everything.
* `ranking.channel_cap_relaxed` marks an item shown although its channel already has two items,
  because no other matching creator was left.
`eligibility` = `{candidates, eligible, ineligible: {reason: count}}`. Candidates already pass
the hard filters (language, duration bounds, synthetic toggle, not-interested, relevance floor);
`ineligible` counts the candidates dropped after that (`no_current_evaluation`,
`below_min_quality`, `no_quality_estimate_with_min_quality_set`). Explanations are factual
statements about the features used (matched interest, strongest dimensions, genre weight and
where it came from, assessment state and confidence), never invented reasons. `video.url` is the
publisher's page for licensed items and a YouTube link only for YouTube videos.

## Admin (`X-Admin-Token`)

| Method | Path | Description |
|---|---|---|
| POST | `/v1/admin/seed-demo` | Load synthetic fixtures and queue Stage A |
| POST | `/v1/admin/discover` | `{n_queries}` — queue discovery for the least-recently-run seed queries (needs YouTube key) |
| POST | `/v1/admin/reevaluate-stale` | Queue re-evaluation of evaluations from older framework versions |
| POST | `/v1/admin/worker/run?max_jobs=` | Drain the queue synchronously (demos; use workers in production) |
| GET | `/v1/admin/jobs?status=` | Job stats by type/status + recent jobs with errors and results |
| GET | `/v1/admin/costs` | Today's LLM spend vs budget, YouTube quota use, ledger by provider/model/operation, mean cost per evaluation |
| GET | `/v1/admin/coverage` | Catalog vs evaluated counts by genre, channel-size bucket, language, status |
| PUT | `/v1/admin/videos/{id}/genre` | `{genre, by}` — manual genre correction (`null` clears it). The classifier output is kept for audit, and the video is re-queued so Stage B is recomputed with the corrected profile |
| GET | `/v1/admin/dead-letters?limit=` | Jobs that exhausted their retries, grouped by type and error class, with correlation ids |
| POST | `/v1/admin/jobs/{job_id}/requeue` | Return a dead job to the queue (attempts reset); 409 for jobs that are not dead |

## Annotation (`/v1/annotation`)

Raters authenticate with a bearer token issued once at creation (stored hashed). The rater UI is
served at `/annotate`. See `docs/annotation.md` for the protocol.

| Method | Path | Who | Description |
|---|---|---|---|
| POST | `/raters` | admin | `{pseudonym, expertise, role: rater|adjudicator}` → rater + token (shown once) |
| GET | `/me` | rater | Pseudonym, role, status (`onboarding` until calibration items are done) |
| GET | `/next?dataset=` | rater | The rater's open item, or the next blind assignment: transcript segments, metadata, attribution/licence, rubric. Never model scores, other ratings or popularity |
| POST | `/annotations/{id}/submit` | rater | Per-dimension `{status, score?, confidence?, evidence[{segment_index, quote?, kind}], note?}`; quotes are verified against the transcript; server-side timing and QC flags |
| GET | `/pairs/next?dataset=&criterion=` / POST `/pairs` | rater | Pairwise preference between two items the rater has rated |
| GET | `/adjudication-queue?dataset=` / POST `/adjudications` | adjudicator | Disagreements; adjudicators cannot adjudicate items they rated |
| GET | `/qc?dataset=` | admin | Throughput, timing, flags, agreement summaries |
| GET | `/disagreements?dataset=` | admin | Cells whose ratings diverge |
| POST | `/annotations/{id}/exclude` | admin | Exclude an annotation with a reason |
| GET | `/export?dataset=` | admin | Benchmark-ready labels (`human_annotation`, `human_adjudication`), repeats, pairs, items and source attributions. Calibration items are never exported as labels |
