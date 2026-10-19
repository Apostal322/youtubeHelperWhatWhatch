## What and why

## How it was verified

- [ ] `ruff check`, `ruff format --check`, `mypy avd`, `pytest -q` (backend)
- [ ] `npm run typecheck`, `npm test`, `npm run build` (extension), if touched
- [ ] Migration added and `alembic check` passes, if the schema changed
- Not verified (say why):

## Integrity checklist

- [ ] No simulated, mocked or heuristic output is presented as real AI or human results
- [ ] No popularity signal reaches evaluation or intrinsic scores
- [ ] Held-out data was not used for tuning; release gates were not weakened
- [ ] Result-changing changes bump a version (prompt, rubric, evaluator, config)
- [ ] No new outbound connection, telemetry or paid API call without operator configuration
- [ ] No unauthorized content downloading
