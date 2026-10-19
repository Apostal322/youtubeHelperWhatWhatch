# Licensing and provenance

**Status (2026-10-19): the license is declared but the license file is missing.**

* `README.md` and `backend/pyproject.toml` declare **Apache-2.0** for code, and the README
  declares **CC0-1.0** for synthetic fixtures and benchmark data.
* The repository has **no `LICENSE` file**. Without the license text in the repository, people
  cannot rely on the grant with confidence. **Adding it is the owner's decision**, and so is
  choosing a different license. Nothing here changes the license.
* **Prepared for approval:** branch `claude/proposal-apache-license` (draft pull request
  against `main`) adds only `LICENSE`, the unmodified Apache License 2.0 text downloaded from
  `https://www.apache.org/licenses/LICENSE-2.0.txt` (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  Merging it is the owner's decision; closing it leaves everything as it is.

## Recommendation (Phase 3.4 review of draft PR #5)

**Merge PR #5 to adopt Apache-2.0.** The review found:

* **Contributor rights:** the history to date has 2 commits by the owner and 69 authored with an
  AI coding assistant on the owner's behalf, and no outside contributors. The owner alone can
  license the code.
* **Content of the change:** the PR adds only the unmodified license text (SHA-256 above), and
  it matches what `README.md` and `backend/pyproject.toml` already declare.
* **Dependencies:** all compatible (table below).
* **No `NOTICE` file needed:** no third-party code is vendored, so there are no attribution
  notices to carry.

After merging, two follow-ups:

* replace the "LICENSE file is not yet in the repository" sentences in `README.md`,
  `CONTRIBUTING.md` and this page;
* tick the license item in [`release-checklist.md`](release-checklist.md).

The license covers **only this project's code** (and the synthetic fixtures under CC0-1.0). It
does not cover:

| | Governed by |
|---|---|
| Project code | Apache-2.0 (once PR #5 is merged) |
| Synthetic fixtures and synthetic benchmark | CC0-1.0 (declared) |
| Models: MiniLM embedder, local LLMs | each model's own license, accepted by the operator when downloading; never redistributed here |
| Datasets: pilot transcripts | the publisher's terms: Changelog CC BY-SA 4.0; Talk Python unconfirmed (see below). Manifests contain hashes and references, not text |
| Content: YouTube videos, captions, audio | their creators; nothing here grants rights to them |

## Who holds copyright today

The commit history has two authors:

* the repository owner;
* commits authored with an AI coding assistant on the owner's behalf, attributed in each commit
  message.

There are no outside contributors yet, so relicensing now needs only the owner's decision. Once
outside contributions arrive under a stated license, changing it needs their agreement (or a
contributor agreement, which this project does not have).

## Options

| | MIT | Apache-2.0 (declared) | GPLv3 | AGPLv3 |
|---|---|---|---|---|
| Permissive / copyleft | permissive | permissive | copyleft (distribution) | copyleft (distribution **and network use**) |
| Explicit patent grant | no | **yes** | yes | yes |
| Others may build closed products on it | yes | yes | no, if distributed | no, if distributed or offered as a service |
| Someone running a modified hosted instance must publish changes | no | no | no | **yes** |
| Compatible with the current dependencies | yes | yes | yes | yes |
| Corporate contributors' comfort | high | high | lower | lowest |

How they fit this project:

* **Apache-2.0 (current declaration)** suits a research tool that others should reuse freely,
  including in products, with a patent grant protecting users. It does not stop someone from
  running a modified, closed community instance.
* **AGPLv3** is the option that addresses the network-service concern. If the owner wants every
  publicly hosted instance, including modified ones, to publish its source (so its evaluation
  method can be inspected), AGPL enforces that, at the cost of fewer corporate users and
  contributors.
* **MIT** is simpler than Apache-2.0 but has no explicit patent grant. There is little reason to
  prefer it here.
* **GPLv3** covers distributed copies but not hosted instances, which is how this software would
  be "distributed" to most users. It gives most of AGPL's friction for less of its benefit here.

**Recommendation for the owner's decision:** keep Apache-2.0 and add its standard text as
`LICENSE` (plus a `NOTICE` file if attributions are needed). Choose AGPLv3 instead only if
ensuring that modified public instances stay inspectable matters more than broad reuse. Decide
before accepting outside contributions.

## Dependencies (as installed, checked from package metadata)

| Component | License | Note |
|---|---|---|
| FastAPI, Pydantic, pydantic-settings, SQLAlchemy, Alembic, pgvector, redis-py, anthropic, onnxruntime | MIT | |
| Uvicorn, Starlette, Celery, Kombu, httpx | BSD-3-Clause | |
| NumPy | BSD-3-Clause and bundled permissive licenses | |
| tokenizers | Apache-2.0 | |
| psycopg 3 | **LGPL-3.0-only** | used as an unmodified library; compatible with Apache-2.0 and AGPL distribution |
| React, React DOM | MIT | extension |

All are compatible with each option above. Re-check after dependency upgrades; CI runs
`pip-audit` and `npm audit` for vulnerabilities, not licenses.

## Models

* **Embedding model** `sentence-transformers/all-MiniLM-L6-v2`: Apache-2.0 per its model card;
  the ONNX file is fetched by the operator (`avd embeddings fetch-model`), pinned by hash, and
  not committed.
* **Local LLMs** (e.g. Qwen3.5 via Ollama): each model has its own license, published on its
  model card. The operator accepts it when pulling. No weights are committed or redistributed by
  this project.
* **Hosted models**: the provider's terms apply to the operator's account.

## Data and content

| Data | License / basis | Redistributable by this repo |
|---|---|---|
| Synthetic fixtures (`data/fixtures`), synthetic benchmark (`benchmarks/datasets/synthetic_v1`) | written for this project; declared CC0-1.0 | yes |
| `real_pilot_v1`, `real_holdout_v2` manifests | metadata, hashes and pinned source references only | yes (no transcript text is committed) |
| Changelog transcripts | CC BY-SA 4.0 from the publisher's repository | fetched by the operator at pinned commits; quotes need attribution and share-alike |
| Talk Python transcripts | scope **unconfirmed** | not shown to raters or quoted publicly (`avd dataset rights`) |
| YouTube videos, captions, audio | owned by their creators | **never**. A software license grants nothing about third-party content |
| User-submitted transcripts/media | the submitter's attestation, for evaluation on that instance | no; never exported as text |

**A software license does not grant rights to YouTube content.** Every instance operator is
responsible for having the right to process the content they evaluate (`docs/permissions.md`).
Evaluation exports carry evidence quotes only where every source permits quotation, and never
transcript text ([`evaluation-export.md`](evaluation-export.md)).

## Third-party code

No third-party source code is vendored in this repository. If code is ever copied in, keep its
license header and list it in a `NOTICE` file.
