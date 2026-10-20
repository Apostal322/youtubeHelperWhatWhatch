# Reproducing the reports

Every committed report records its git commit, dataset hash, settings and seed. All commands run
from `backend/`; none makes a paid API call.

| Report (`benchmarks/reports/`) | Command | Data |
|---|---|---|
| `baseline-heuristic-synthetic_v1` | `avd bench --tag baseline-heuristic-synthetic_v1 --out ../benchmarks/reports` | synthetic fixtures |
| `baseline-heuristic-real_pilot_v1-dev` | `avd bench-real --offline --tag baseline-heuristic-real_pilot_v1-dev` | real pilot, dev split |
| `retrieval-talkpython-n100-s7` | `avd bench-retrieval --clone <talk-python-transcripts clone> --n 100 --seed 7` | 100 Talk Python transcripts (internal analysis) |
| `recs-real_pilot_v1-test` | `avd bench-recs-real --with-minilm` | test split (**exposed**; kept for the record) |
| `recs-diagnostic-real_pilot_v1-dev` | `avd bench-recs-diagnose` | dev split, exploratory |
| `throughput-real_pilot_v1` | `avd bench-throughput` | dry-run call plan, projections |
| gate classification | `avd gates status --report ../benchmarks/reports/baseline-heuristic-real_pilot_v1-dev.json --manifest ../benchmarks/datasets/real_pilot_v1/manifest.json` | any bench-real report |
| live evaluation run | `avd evaluate-real --dry-run --offline` (plan), then with `--budget-usd` | needs credentials and an approved budget |

## Data provenance

* **Fetching.** `avd dataset import [--offline]` fetches each transcript at the pinned upstream
  commit and verifies:
  * the raw SHA-256;
  * the normalised content hash (parser drift fails closed);
  * licence fields.

  CI re-fetches and verifies the pilot on every push.
* **Datasets and records.**
  * Datasheet: `benchmarks/datasets/real_pilot_v1/README.md`.
  * Sealed hold-out: `benchmarks/datasets/real_holdout_v2/`.
  * Rights review: `avd dataset rights`.
  * Exposure ledger: `benchmarks/datasets/real_pilot_v1/exposure.json`.
* **Sources and conditions.**
  * Changelog transcripts: CC BY-SA 4.0, stated in the publisher's README; `bigtent/` excluded.
  * Talk Python transcripts: GPL-3.0 repository licence; scope over the text unconfirmed. Used
    for internal analysis only until the publisher confirms more.
* **Not in the repository.** Transcripts are not redistributed: only paths, checksums and
  metadata are committed, and the text is fetched on import.

Determinism: heuristic runs, bootstrap seeds and dry-run plans are deterministic. Timing columns
vary with machine load (e.g. the dev baseline's p50 latency, 57 ms vs 85 ms between runs).
