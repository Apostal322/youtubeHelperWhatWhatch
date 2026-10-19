# Contributor-run benchmark results

Each directory here is one `avd bench-local` run on a contributor's own hardware, imported with
`avd community import`. Every directory contains:

* `report.json` / `report.md`: the benchmark report, exactly as produced;
* `runs/*/summary.json`, `runs/*/plan.json`: per-run summaries (no transcript text);
* `VERIFICATION.json`: what `avd community verify` checked and what it cannot check.

Status is always **contributor-reported**. The checks confirm:

* real (not mocked) local inference was reported;
* the dev split of this repository's dataset, matching its manifest hash;
* a clean commit;
* a recorded model weights digest and hardware;
* internally consistent numbers.

They cannot prove the run happened as reported: the files are unsigned and the numbers are
self-reported. Independent runs that agree (same model digest, same commit) are the evidence.

None of these results is a validation against human judgement.

As of 2026-10-19 this directory is empty: no real local-model run exists yet. See
[`docs/volunteer-pilot.md`](../../docs/volunteer-pilot.md) to contribute one.
