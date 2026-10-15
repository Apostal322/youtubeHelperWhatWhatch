# Permissions, content access and compliance

The system separates **discovery** (finding videos and their public metadata) from **content
acquisition** (obtaining material whose processing is permitted). Only the second can produce a
quality evaluation.

## What uses what

| Capability | Source | Credential / permission | Status in this repo |
|---|---|---|---|
| Discovery | YouTube Data API `search.list` (`order=date`, never `viewCount`) | API key (`AVD_YOUTUBE_API_KEY`), Google Cloud project, quota | Implemented, tested with mocked HTTP |
| Metadata | `videos.list`, `channels.list` | API key | Implemented, tested with mocked HTTP |
| Creator captions | `captions.list` + `captions.download` | **OAuth grant from the channel owner** (`youtube.force-ssl`); requires a verified Google OAuth app | Client + provider implemented and tested with mocks; OAuth consent flow and token storage are **not** implemented (`CreatorTokenStore` protocol) |
| User-provided transcripts | `POST /v1/videos/{id}/transcripts` | User attestation of their right to submit (stored as the authorization basis) | Implemented |
| Licensed datasets | `avd import-dataset file.jsonl` | Each row must declare `license` and `authorization_basis`; rows without are rejected | Implemented |
| Audio quality | Authorized PCM WAV attached to a content source | Same as the content source | Implemented (demo uses synthetic audio) |
| Visual quality / editing / readability | Authorized video | Rights to process the video + multimodal model | **Not implemented** — always `requires_media` |
| LLM evaluation | Anthropic API or OpenAI-compatible server | API key; provider terms | Implemented, tested with scripted responses only |
| Retrieval fact checking | Licensed reference corpus (`AVD_REFERENCE_CORPUS_DIR`, documents need a `.meta.json` with `license`) | Corpus licence | Implemented, tested with a fake retriever |

## What the system deliberately does not do

* No downloading of video or audio from YouTube, no use of unofficial endpoints, no scraping of
  watch pages, no circumvention of DRM or access controls, no rotating proxies or evasion of rate
  limits.
* No assumption that third-party captions are available through the API — for videos the
  authenticated user cannot edit, `captions.download` is not permitted, so such videos stay
  `content_unavailable` unless another authorized source exists.
* No inference in the browser extension. It reads precomputed results from the backend; the badge
  on YouTube pages is a cached backend lookup.

## Quotas

The YouTube Data API default is 10,000 units/day. The quota guard records units *before* each call
(failed calls are charged by YouTube too) and keeps `AVD_YOUTUBE_QUOTA_RESERVE_UNITS` in reserve. With
`order=date` discovery at 100 units per search page and batched metadata at 1 unit per 50 videos,
the default quota supports roughly 90 search pages (≈4,500 discovered videos) per day. Large-scale
operation needs a quota extension request (Google's audit process).

## Before running against real data

1. Review the YouTube API Services Terms of Service and Developer Policies for your use case
   (storage duration of API data, display requirements, refresh obligations for statistics).
2. Obtain a quota appropriate for your discovery volume.
3. For creator captions: register and verify an OAuth client, implement consent + token storage
   (encrypted), honour revocation (set `content_sources.status = 'revoked'`; the evaluation is then
   stale and should be re-run or withdrawn).
4. For LLM providers: check data-handling terms for transcripts you send.
5. For user-submitted transcripts: define a takedown process; submissions record the submitting
   user and their attestation.
