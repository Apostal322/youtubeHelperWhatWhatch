# Security review (Phases 2–3.2)

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

## Phase 3.2: local-first and self-hosted deployment

### Deployment boundaries

| Mode | Who may connect | Authentication | Safe for |
|---|---|---|---|
| Local (`avd start`, default) | loopback peers only (plus a Docker bridge you declare in `AVD_LOCAL_PEER_NETWORKS`); `Host` must be a loopback name | none needed; anonymous token for preferences | one person on their own computer |
| Server (`AVD_DEPLOYMENT_MODE=server`) | anyone who reaches the reverse proxy; `Host` must be in `AVD_ALLOWED_HOSTS` | bearer tokens; registration `invite` (default) or `closed`/`open`; optional private reads; admin token ≥ 24 chars | a shared instance behind TLS |

**Unsupported:**

* a local instance port-forwarded or tunnelled to the internet: the guard refuses non-loopback
  peers, but a tunnel that terminates on loopback defeats that;
* any instance without TLS in front of it on a network;
* Ollama, PostgreSQL or Redis exposed beyond the host.

### Threat model and findings

| Threat | Before | Now | Test |
|---|---|---|---|
| **DNS rebinding**: a web page re-resolves its domain to 127.0.0.1 and reads the local API as same-origin | **Open**: no `Host` check | `Host` must be in `AVD_ALLOWED_HOSTS` (loopback names by default) → 400 | `test_local_first.py::test_dns_rebinding_host_is_rejected`; CI smoke step |
| **Cross-site requests to localhost**: a page submits work or creates accounts on your machine | **Open** for simple requests | Local mode refuses state-changing requests with a web `Origin` other than an extension or local dev page (403); no cookies are used anywhere | `test_cross_site_writes_are_refused`; CI smoke step |
| **Local instance accidentally exposed** (bound to 0.0.0.0, LAN) | served everyone | non-loopback peers get 403 in local mode; `avd serve`/`avd start` refuse a non-loopback bind unless a container bridge is declared | `test_other_machines_are_refused`, `test_local_instance_refuses_a_public_bind` |
| **Database and Redis published** by Docker Compose on all interfaces with default credentials | **Open**: `5432:5432`, `6379:6379` | not published; API and Ollama on `127.0.0.1` only; server mode refuses the default database password | `docker compose config`; `test_unsafe_configuration_refuses_to_start` |
| **Empty admin token**: `AVD_ADMIN_TOKEN=` made the empty string a valid credential | **Latent**: an empty header matched | empty secrets count as unset (admin API disabled) | `test_empty_admin_token_disables_admin` |
| **Unsafe shared server**: weak or placeholder admin token, no allowed hosts, open registration by accident, public metrics | possible | server mode **fails closed at startup** with the list of problems | `test_unsafe_configuration_refuses_to_start` (5 cases) |
| **Open account creation on a public server** | anyone could create accounts | default `invite` (codes compared in constant time) or `closed`; admin token can always create | `test_invite_registration`, `test_closed_registration_and_private_reads` |
| **Extension disclosed browsing**: the badge looked up every opened YouTube video on the configured backend | **Open** for remote backends | lookups only for local backends, or remote ones with consent *and* a separate opt-in | extension `lib.test.ts`, `background.ts` |
| **Silent connection to a remote server** | URL field + permission prompt | URL validation (origin only, HTTPS for non-loopback, no credentials in the URL), diagnostics shown first, explicit consent per origin; requests to an unconsented remote origin are refused client-side | `backend.test.tsx`, `lib.test.ts` |
| **Any extension could read the API via CORS** (`chrome-extension://.*`) | allowed | no wildcard; this extension uses host permissions, not CORS | `test_no_cors_wildcard_for_extensions` |
| **Malicious evaluation packages** (imports) | n/a (new) | strict schema (unknown fields refused, value ranges, size caps), integrity and per-quote hash checks; stored apart, never authoritative, never ranked | `test_portability.py` |
| **Copyrighted text in exports** | n/a (new) | no transcript text ever; quotes as hashes unless every source permits quotation | `test_no_transcript_and_no_quotes_by_default`, `test_quotes_only_where_every_source_permits` |
| **Third-party evaluators overriding the protocol** | n/a (new) | only explicitly configured; lowest precedence (cannot replace official scores); recorded in versions and config hash | `TestExperimentalEvaluators` |
| **Unbounded local load** (CPU-only machines) | queue unbounded | submissions refused (503) above `AVD_MAX_QUEUED_JOBS`; one embedded worker; Ollama concurrency 1; cancellation and pause | `test_backlog_limit_refuses_new_submissions`, `test_cancel_queued_jobs` |
| **SSRF through model endpoints** | (Phase 3.1) | Ollama URL must be local/private unless `AVD_OLLAMA_ALLOW_REMOTE`; no proxy for local traffic; the backend never fetches URLs supplied by API callers | `test_local_ollama.py::TestLocalOnly` |
| **Malicious model output / prompt injection** | (Phases 2–3) | unchanged: strict schemas, grounding verification, injection detector | as above |

### Remaining gaps (Phase 3.2)

* **Local mode trusts every local user and process.** On a shared multi-user computer, other
  accounts can reach `127.0.0.1:8000`. Use server mode (with tokens) there.
* **Experimental evaluators run in-process.** They receive no credentials, but Python code can
  read the environment. Configure only code you trust.
* **Export integrity is not authenticity.** No signatures yet.
* **TLS is not built in.** Server mode expects a reverse proxy (`docs/self-hosting.md`). Client
  IPs come from `--forwarded-allow-ips`, which the operator must set to the proxy's address.
* **Bearer tokens do not expire.** Revoking means deleting the account.
* **Container images are not scanned for CVEs in CI.** The Ollama image tag is unpinned by
  default.

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
* **The public read API is unauthenticated by default** (`/v1/videos`, `/v1/search`). This is
  intended for public instances; a private server sets `AVD_REQUIRE_AUTH_FOR_READ=true`.
* **User tokens are bearer tokens without expiry or rotation.**
* **Media is stored unencrypted on disk.** There is no malware scanning beyond ffprobe
  validation, and no per-user storage quota.
* **The prompt-injection detector's recall on real attacks is unmeasured.** The grounding check
  is the backstop: obeying an injection while citing it is invalidated.
* **Logs** contain request routes, status codes and correlation ids, never bodies or tokens.
  Retention of logs is a deployment decision.
* **Not reviewed:** Docker images and their base-layer CVEs, TLS termination, database
  credentials management. These depend on deployment.
