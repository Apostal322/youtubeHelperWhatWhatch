# Beginner-friendly tasks

Each item is small, has a clear definition of done and touches one area. Open an issue first, or
comment on an existing one, so two people do not do the same thing. The owner can turn these into
GitHub issues with the `good first issue` label; the label does not exist yet.

| # | Task | Area | Done when |
|---|---|---|---|
| 1 | **Try `avd start` on Windows or macOS** and fix or document what breaks | install | `docs/local-setup.md` says "tested on …" with your OS and version, or an issue lists the failure |
| 2 | **Add `avd data restore` for PostgreSQL** (wrap `pg_restore`, refuse to overwrite a non-empty database) | ops | a test with the PostgreSQL fixture |
| 3 | **Show imported evaluations in the extension**, read-only, labelled "Imported, unverified · <origin>" (`GET /v1/videos/{id}/imported-evaluations`) | extension | component test; never mixed into local scores |
| 4 | **A `--json` flag for `avd local doctor` output in `avd doctor`**, so setup issues can attach one file | CLI | test of the JSON shape |
| 5 | **Dark-mode check of the popup** (colours come from CSS variables) | extension | screenshots in the PR, contrast ≥ 4.5:1 for body text |
| 6 | **Use MiniLM for search when its model is already on disk**: the default `hashing` embedder's collisions put unrelated videos near the top (`docs/demo.md`, step 5). Detect a fetched model in local mode, or make `avd doctor` recommend `AVD_EMBEDDING_PROVIDER=onnx_minilm` | search | a test on the synthetic catalog; `avd embeddings reindex` documented for existing installs |
| 7 | **Translate the rubric anchors** into another language as a *separate* registered prompt version (never edit the English one in place) | research | registered with `languages_written_for`; `languages_verified_live` stays empty until tested |
| 8 | **Pin the Ollama image** in `docker-compose.yml` to a release you verified | deployment | `docker compose --profile local-ai up` works with it; release noted |
| 9 | **Add a CI job that runs `scripts/e2e/full_local.py` without the browser part** | CI | green job; under 3 minutes |
| 10 | **Report a real local-model run** ([`first-experiment.md`](first-experiment.md)) | research | `avd community verify` accepts it |
| 11 | **Try the rating usability pilot** (15–20 minutes, an excerpt; [`annotation-pilot.md`](annotation-pilot.md)) once the consent text is approved | research | the coordinator's `usability-report` and your notes on what was unclear |

Before you start: [`CONTRIBUTING.md`](../CONTRIBUTING.md) (setup takes five minutes; no GPU or
key needed) and its ground rules on honesty about simulated output, popularity and held-out
data.
