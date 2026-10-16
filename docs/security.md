# Security review (Phase 2)

Scope: backend API and workers, the annotation UI, dataset and media ingestion, LLM integration,
and the browser extension. Method: targeted code review of trust boundaries, dependency audits,
and tests for each control. This is a self-review by the implementer, not an independent audit.

## Trust boundaries and controls

| Boundary | Threat | Control | Test |
|---|---|---|---|
| Transcript / metadata → LLM | Prompt injection, score manipulation | Nonce-delimited untrusted framing, `<`/`>` neutralised; detector (`injection_v2`) flags segments; flagged segments cannot be evidence; every citation is verified against the transcript; strict output schemas | `test_phase2_llm.py` (14 adversarial variants, 8 benign cases), `test_factcheck_v2.py` |
| Reference passages → fact-check judge | Injection via corpus documents | Passages framed and neutralised like transcripts; verdicts must cite retrieved passage ids, otherwise downgraded | `test_factcheck_v2.py::test_transcript_and_passages_are_framed_as_untrusted` |
| Popularity → quality | Leakage through text | Popularity statements stripped from title, description and tags; quotes that are popularity statements are rejected as evidence; ×1000 popularity invariance check | `test_factcheck_v2.py::test_popularity_statements_are_never_evidence`, benchmark gate |
| API callers | Abuse, brute force | Per-caller token bucket (429 + Retry-After); bearer tokens stored as SHA-256 hashes; admin token compared in constant time; `/metrics` is admin-only by default | `test_ops.py` |
| Error responses | Information leakage | One error shape; unhandled errors return a generic 500 (details only in server logs) | `test_ops.py::test_errors_share_one_shape` |
| Dataset manifests | Path traversal on fetch | Paths must be plain relative repo paths; resolved cache path must stay inside the cache. **Found and fixed in this phase**: `../` in a manifest path escaped the cache dir | `test_datasets.py::test_manifest_paths_cannot_escape_the_cache` |
| Dataset content | Tampering, upstream change | Raw SHA-256 and normalised content hash verified on import; pinned commit; fail closed | `test_datasets.py` |
| Media uploads | Malicious files, SSRF via playlists, DoS | Content-type allowlist; streamed size cap (`AVD_MAX_MEDIA_UPLOAD_MB`); ffprobe validation; ffmpeg/ffprobe with `-protocol_whitelist file`, no stdin, timeouts, list arguments (no shell); inputs confined to the media dir; content-addressed file names; bounded frame decoding | `test_media.py` (incl. a playlist pointing at a URL) |
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
* CI runs both audits on every push and pull request.

## Known gaps (not fixed)

* **Rate limiting is per process.** With N API workers, a caller gets N× the limit. A shared
  Redis limiter is needed before multi-worker deployment.
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
