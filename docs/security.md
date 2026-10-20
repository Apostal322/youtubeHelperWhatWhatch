# Security review (Phases 2–3)

Scope: backend API and workers, the annotation UI, dataset and media ingestion, LLM integration,
and the browser extension. Method: targeted code review of trust boundaries, dependency audits,
and tests for each control. This is a self-review by the implementer, not an independent audit.

## Trust boundaries and controls

| Boundary | Threat | Control | Test |
|---|---|---|---|
| Transcript / metadata → LLM | Prompt injection, score manipulation | Nonce-delimited untrusted framing, `<`/`>` neutralised; detector (`injection_v2`) flags segments; flagged segments cannot be evidence; every citation is verified against the transcript; strict output schemas | `test_phase2_llm.py` (14 adversarial variants, 8 benign cases), `test_factcheck_v2.py` |
| Reference passages → fact-check judge | Injection via corpus documents | Passages framed and neutralised like transcripts; verdicts must cite retrieved passage ids, otherwise downgraded | `test_factcheck_v2.py::test_transcript_and_passages_are_framed_as_untrusted` |
| Popularity → quality | Leakage through text | Popularity statements stripped from title, description and tags; quotes that are popularity statements are rejected as evidence; ×1000 popularity invariance check | `test_factcheck_v2.py::test_popularity_statements_are_never_evidence`, benchmark gate |
| API callers | Abuse, brute force | Two token buckets per request: the caller's (credential hash, else IP) **and** the client IP's, so rotating made-up tokens cannot multiply the allowance (**found and fixed in Phase 3**). Shared across API processes and hosts through Redis (atomic Lua on the Redis clock); per process if Redis is absent or failing, counted in `avd_rate_limiter_fallback_total`. 429 + Retry-After; bearer tokens stored as SHA-256 hashes; admin token compared in constant time; `/metrics` admin-only by default | `test_ops.py`, `test_ratelimit_shared.py` (3 processes share one burst; 2 API instances share one limit; token rotation; Redis down) |
| Feedback | Spam, duplicate events | Exact duplicates are idempotent (`status: duplicate`); 500 events per user per day; personal reports never edit global evaluations | `test_ops.py::test_feedback_is_idempotent_capped_and_never_rewrites_evaluations` |
| Blind annotation | Raters seeing the model's view first | While a dataset is blinded, public detail, evaluation, history and claims endpoints return 403 for its items; search, listings and recommendations skip them; the rater payload has no publisher link or model output | `test_e2e_scenario.py` (step 2), browser check |
| Licensed content rights | Using content beyond confirmed permission | Use-time rights review: pending uses fail closed (Talk Python: no external annotation, quotes withheld from public responses) until a written confirmation is recorded | `test_rights.py` |
| Error responses | Information leakage | One error shape; unhandled errors return a generic 500 (details only in server logs) | `test_ops.py::test_errors_share_one_shape` |
| Dataset manifests | Path traversal on fetch | Paths must be plain relative repo paths; resolved cache path must stay inside the cache. **Found and fixed in this phase**: `../` in a manifest path escaped the cache dir | `test_datasets.py::test_manifest_paths_cannot_escape_the_cache` |
| Dataset content | Tampering, upstream change | Raw SHA-256 and normalised content hash verified on import; pinned commit; fail closed | `test_datasets.py` |
| Media uploads | Malicious files, SSRF via playlists, DoS | Content-type allowlist; streamed size cap (`AVD_MAX_MEDIA_UPLOAD_MB`); ffprobe validation; ffmpeg/ffprobe with `-protocol_whitelist file`, no stdin, timeouts, list arguments (no shell); inputs confined to the media dir; content-addressed file names; bounded frame decoding. **Phase 3:** every process runs under `prlimit` (2 GiB address space, CPU seconds = timeout, 1 GiB max written file, 64 open files); media over 6 h is refused and extraction is capped. The Docker image now ships ffmpeg and prlimit (before, the path returned 503 in Docker). `/v1/status` reports whether limits are enforced | `test_media.py` (incl. a playlist pointing at a URL; a 16 MB address-space cap makes ffprobe fail closed; over-length media refused) |
| Annotation UI | XSS from transcript text | Untrusted text rendered with `textContent` only; strict CSP on `/annotate` (`default-src 'none'`, `connect-src 'self'`, `frame-ancestors 'none'`); rater tokens are hashed server-side and kept in `sessionStorage` client-side | `test_annotation.py`, `test_ops.py::test_security_headers_and_annotation_csp` |
| All responses | Clickjacking, MIME sniffing | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` | `test_ops.py` |
| Credentials | Exposure | Secrets only from env (`SecretStr`, never logged or returned); `/v1/status` reports only *whether* credentials are configured; the extension never holds LLM credentials | `test_ops.py::test_metrics_and_status` |
| Spend | Runaway LLM cost | Daily and per-job spend caps checked *before* each call (including retries and chunk calls); `avd llm-smoke` caps budgets explicitly | `test_phase2_llm.py` |
| User data | Retention | `DELETE /v1/users/me` removes preferences, feedback and impressions; revokes and erases submitted transcripts; withdraws derived evaluations | `test_ops.py::test_account_deletion_removes_user_data` |
| Licensed content | Takedown | `avd dataset revoke`: sources revoked, evaluations withdrawn, annotation assignment stops | `test_datasets.py` |

## Dependency audits (2026-10-16)

* `pip-audit` (PyPI/OSV advisories) over the 76 installed backend packages: **no known
  vulnerabilities**.
* `npm audit` for the extension (all and production-only): **0 vulnerabilities**.
* CI runs both audits on every push and pull request. Re-run on 2026-10-19 before Phase 3:
  still 0 known vulnerabilities (pip-audit 76 packages; npm audit).

## Known gaps (not fixed)

* **Rate limiting degrades to per process when Redis is unavailable.** It is still limiting,
  and the fallback is visible in metrics. Client IPs come from the socket, so behind a reverse
  proxy all callers share one IP bucket until trusted-proxy handling (`X-Forwarded-For` from
  known proxies only) is configured.
* **Media isolation is resource limits, not a sandbox.** ffmpeg runs as the API's user, with no
  seccomp, namespaces or separate container. A decoder exploit could reach what that user can.
  Running media workers in a separate, network-less container is the next step.
* **Revocation leaves derived data.** Revoking a source or deleting an account withdraws
  evaluations and erases submitted transcript text. Embeddings and evaluation rows are kept
  (not current) for audit. Cached LLM responses for that content remain in `llm_cache` until it
  is pruned. Retention of these derived records has no time limit yet.
* **CSP allows inline script/style on `/annotate`** (a single self-contained page). Moving the
  script to a static file would allow `script-src 'self'` only.
* **No authentication for the public read API** (`/v1/videos`, `/v1/search`). This is
  intended, but those endpoints are rate-limited only.
* **User tokens are bearer tokens without expiry or rotation.**
* **Media is stored unencrypted on disk.** There is no malware scanning beyond ffprobe
  validation, and no per-user storage quota.
* **The prompt-injection detector's recall on real attacks is unmeasured.** The grounding check
  is the backstop: obeying an injection while citing it is invalidated.
* **Logs** contain request routes, status codes and correlation ids, never bodies or tokens.
  Retention of logs is a deployment decision.
* **Not reviewed:** Docker images and their base-layer CVEs, TLS termination, database
  credentials management. These depend on deployment.
