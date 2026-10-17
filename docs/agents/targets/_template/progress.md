# <Target title>: progress

Task: follow `docs/agents/targets/<name>/target.md`. Runs are appended by `tests/search_experiment.py --record <name>`; logs, metadata and source patches live in `logs/targets/<name>/`.

## Fixed evaluation contract

- Command: `<evaluation command>`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO`
- Hardware: <machine>.
- Base revision: <git revision plus any uncommitted infrastructure>.

## Comparison table

| Run | Change | <primary metric> | <secondary metrics> | Verdict |
|---|---|---|---|---|
| baseline | — | | | |

## Resume checkpoint

- Accepted implementation: <run id, patch>.
- Working tree / build: <state>.
- Running processes: <none / which, and their logs>.
- Next step: <one concrete step>.

## Runs

<!-- tests/search_experiment.py appends one entry per run below; add a short note after each. -->
