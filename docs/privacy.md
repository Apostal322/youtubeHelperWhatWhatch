# Data ownership and privacy

Every instance keeps its data in its own database and media directory, on hardware its operator
controls. The software sends nothing to the project. There is no telemetry, analytics, crash
reporting or update check.

## What an instance stores

| Data | Where | Personal? | Removed by |
|---|---|---|---|
| Video metadata (title, channel, duration, tags) | `videos`, `channels` | no | `avd dataset revoke`, database deletion |
| Authorized transcripts / media | `content_sources`, media directory | may contain personal data of speakers | revocation; account deletion for user submissions |
| Evaluations, dimension scores, evidence references | `evaluations`, `dimension_scores`, `evidence` | no | `avd data clear --evaluations` |
| Embeddings | `videos` (content vector), `user_preferences` (interest vector) | no / derived from interests | `avd embeddings reindex`; account deletion |
| Model response cache | `llm_cache` | contains model output about content | `avd data clear --model-cache` |
| Accounts (anonymous) | `users` (only a token **hash**) | pseudonymous | `DELETE /v1/users/me` |
| Preferences, interests | `user_preferences` | yes | `DELETE /v1/users/me`, `avd data clear --preferences` |
| Feedback, recommendation impressions | `feedback`, `recommendation_impressions` | yes (behavioural) | `DELETE /v1/users/me`, `avd data clear --feedback` |
| Human annotations | `annotations`, `raters` (pseudonyms) | pseudonymous | dataset tooling |
| Imported evaluations | `imported_evaluations` | no | `avd data clear --imported` |
| Logs | stdout | routes, status codes and correlation ids, never request bodies or tokens | the operator's log retention |

## Controls

| Need | Command or endpoint |
|---|---|
| A user's copy of their data | `GET /v1/users/me/export` |
| A user deletes their account | `DELETE /v1/users/me` (preferences, feedback and impressions deleted; submitted transcripts revoked and erased; derived evaluations withdrawn) |
| Operator backup / restore | `avd data backup --out f.db` / `avd data restore f.db` (SQLite); `pg_dump` / `pg_restore` (PostgreSQL) |
| Clear cached evaluations or model output | `avd data clear --evaluations` / `--model-cache` |
| Clear preferences or feedback for everyone | `avd data clear --preferences` / `--feedback` |
| Share results without content or personal data | `avd evaluations export` ([`evaluation-export.md`](evaluation-export.md)) |

## The browser extension

* It stores the backend address, your anonymous token, an invite code if you entered one, and the
  list of remote servers you agreed to use, in `chrome.storage.local`, on this device only.
* It talks only to the backend you chose. By default that is `http://localhost:8000`; there is
  no built-in server.
* Before it sends anything to a backend on another computer, it shows that server's diagnostics
  and asks for your explicit agreement. Until then, its requests are blocked in the extension.
* The toolbar badge looks up each YouTube video you open. It does so automatically only for a
  backend on your own computer. For a remote backend it needs both your connection consent and
  a separate opt-in, because it would tell that server what you watch.
* It never holds model API keys and never runs model inference.

## Outbound connections

The test suite enforces this: any DNS lookup or connection to a non-local address made during
the tests fails the run (`tests/conftest.py`).


Only those the operator configures. Full list:
[`infrastructure-costs.md`](infrastructure-costs.md#outbound-connections-an-instance-makes).

* A hosted model provider (Anthropic, an OpenAI-compatible URL) receives transcript text when
  it is configured. That is the operator's explicit choice and is reported by `/v1/status` and
  shown in the extension.
* There is no automatic fallback to a hosted provider.

## Content rights are separate from privacy

Owning an installation does not grant rights to third-party content. Transcripts are processed
only with an authorization basis (creator grant, license, or the submitter's attestation). They
are never exported as text ([`permissions.md`](permissions.md), [`licensing.md`](licensing.md)).
