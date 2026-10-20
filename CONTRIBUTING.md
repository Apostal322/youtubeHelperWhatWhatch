# Contributing

Thanks for helping. This project tries to answer a hard question: can we assess a video's
quality independently of its popularity, transparently and reproducibly? Contributions that keep
it honest about what it can and cannot do are as valuable as new features.

## Start in five minutes

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch && cd youtubeHelperWhatWhatch/backend
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
avd start --demo                                            # http://127.0.0.1:8000, synthetic demo data
```

* The extension: `cd extension && npm ci && npm run build`, then load `extension/dist` as an
  unpacked extension in Chrome.
* No GPU, model, API key, PostgreSQL or Redis is needed for development. Tests mock model
  servers and say so.

## Before you open a pull request

```bash
cd backend && ruff check avd tests && ruff format --check avd tests && mypy avd && pytest -q
cd extension && npm run typecheck && npm test && npm run build
```

* **PostgreSQL + pgvector and Redis tests** run when `AVD_TEST_DATABASE_URL` and
  `AVD_TEST_REDIS_URL` are set; CI sets them.
* **Live model tests** (`pytest -m live`) run only against a local Ollama you configure
  (`AVD_TEST_OLLAMA_URL`, `AVD_TEST_OLLAMA_MODEL`). They are skipped otherwise, and a skip is
  not a pass.
* **Database changes** need an Alembic migration (`backend/avd/migrations/versions/`).
  `alembic upgrade head && alembic check` must pass.

## Ground rules (they protect the research)

1. **Never present simulated output as real.** Synthetic fixtures, mocked models and heuristic
   scores are labelled as such everywhere: API, extension, reports, tests and docs.
2. **Popularity is not quality.** Views, likes and subscriber counts must never reach an
   evaluator or the intrinsic score. `tests/test_evaluation_core.py` and the popularity-invariance
   benchmark guard this.
3. **Abstain rather than guess.** A dimension without evidence is `insufficient_evidence` or
   `requires_media`, never a default score.
4. **Held-out data stays sealed.** Do not tune anything on `real_pilot_v1` test or
   `real_holdout_v2`, and do not weaken release gates (`benchmarks/gates.json`) to make
   something pass.
5. **Version what changes results.** Prompts, rubrics, evaluators and model configurations are
   versioned and enter the config hash. A change that alters scores bumps a version.
6. **Content rights.** Do not add code that downloads or scrapes video, audio or captions
   without authorization (see `docs/permissions.md`). Public availability is not permission.
7. **No personal data leaves an instance** except where its operator configured it
   (`docs/privacy.md`). No telemetry.
8. **No hidden costs.** Nothing may call a paid API unless the operator configured it, and
   nothing may fall back to one.

## Where to start

* Issues labelled `good first issue` (maintainers add the label; if none are open, ask in an issue).
  Ready-to-file candidates: [`docs/good-first-issues.md`](docs/good-first-issues.md).
* Research contributions (model runs, ratings, relevance judgments, reproducing a benchmark):
  [`docs/research-contributing.md`](docs/research-contributing.md) and
  [`docs/volunteer-pilot.md`](docs/volunteer-pilot.md).
* Rating transcripts as a volunteer (no installation needed):
  [`docs/annotation-pilot.md`](docs/annotation-pilot.md).
* Evaluators, content sources, embeddings, rankers and exporters have documented extension
  points: [`docs/extending.md`](docs/extending.md).
* Architecture: [`docs/architecture.md`](docs/architecture.md). Research method:
  [`docs/evaluation.md`](docs/evaluation.md).

## Pull requests

* Keep one change per PR, with tests. Explain what changed and how you verified it, including
  what you could *not* verify (e.g. "not tested on Windows").
* Fill in the PR template's checklist honestly. An unchecked box with a reason is fine.
* Commits: imperative subject line, wrapped body explaining why.

## Recognition

Contributors are credited in release notes for the change they made. Larger contributions
(evaluators, datasets, annotation work) are credited in the documentation of that component.
Human raters who contribute labels are credited only with their consent and never by name in
data files (raters are pseudonymous).

## License of contributions

The project declares Apache-2.0 for code (`backend/pyproject.toml`) and CC0-1.0 for synthetic
fixtures and benchmark data. **The repository does not yet contain a LICENSE file. Adding it is
a decision for the owner**; the exact change is prepared as a separate draft pull request (see
[`docs/licensing.md`](docs/licensing.md)). By contributing, you
agree that your contribution is provided under the license the project declares at the time of
your contribution.
