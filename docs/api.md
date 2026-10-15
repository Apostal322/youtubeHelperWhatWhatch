# API reference

Interactive OpenAPI docs are served at `/docs`. All responses are JSON. Authenticated endpoints use
`Authorization: Bearer <token>` (obtained from `POST /v1/users`); admin endpoints use
`X-Admin-Token` (`AVD_ADMIN_TOKEN`; disabled when unset).

Popularity statistics are never returned by public endpoints.

## Public

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + demo-mode flag |
| GET | `/v1/meta` | Framework version, evaluation mode + disclaimer, rubric (questions, anchors, required modalities), genre profiles (weights) |
| GET | `/v1/videos?genre=&evaluated=true&limit=&offset=` | Catalog listing with quality summaries |
| GET | `/v1/videos/{id}` | Video card, available modalities, content sources (type, authorization basis, license), current quality summary |
| GET | `/v1/videos/{id}/evaluation` | Full current evaluation (below) — 404 with the video status if not evaluated |
| GET | `/v1/videos/{id}/evaluations` | Evaluation history (all framework versions/stages) |
| GET | `/v1/search?q=&genre=&language=&min_quality=&include_synthetic=` | Topic search over analyzed videos; `match` = semantic similarity, title match, score |

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
validation). Only `scored` dimensions carry a score.

## User

| Method | Path | Description |
|---|---|---|
| POST | `/v1/users` | Create an anonymous user → `{user_id, token}` (token shown once; stored hashed) |
| GET / PUT | `/v1/users/me/preferences` | Interests, genre weights (0–1), dimension priority multipliers (0–3), depth (`any/intro/intermediate/advanced`), languages, max duration, diversity, min quality, small-creator boost (0–0.3), include synthetic |
| GET | `/v1/recommendations?limit=` | Personalised feed (below) |
| POST | `/v1/feedback` | `{video_id, kind, impression_id?, evaluation_id?, dimension?, suggested_score?, comment?}`; `kind` ∈ `recommendation_useful`, `recommendation_not_useful`, `not_interested`, `evaluation_incorrect` (requires `evaluation_id`) |
| POST | `/v1/videos/{id}/transcripts` | `{format: vtt|srt|text|json, content, language?, attestation}` → stored as `user_provided_transcript` and queued (202) |

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
