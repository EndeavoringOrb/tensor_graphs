# Prefill+decode first plan within 30 s: progress (laptop)

Task: follow `docs/agents/targets/prefill_decode_incumbent_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_incumbent_seq_len:laptop`; logs, metadata and source patches live in `logs/targets/prefill_decode_incumbent_seq_len/`.

## Fixed evaluation contract

- Command: `.venv/Scripts/python.exe tests/search_experiment.py --record prefill_decode_incumbent_seq_len:laptop <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len <N> --ignore-checkpoints --run-dir logs/gemma_output_runs/<run_id>`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test bench --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM, OpenCL on.
- Base revision: <git revision plus any uncommitted infrastructure>.

## Comparison table

Accepted implementation: <run id>. N\* = <N>; first incumbent at N\* = <s>.

| Run | Change | N | ok | first incumbent s | final incumbent | compile s | Verdict |
|---|---|---|---|---|---|---|---|
| baseline | — | 1 | | | | | |

## Phase scaling

| N | buckets | saturation s | reduction s | root propagation s | heuristic build s | first incumbent s |
|---|---|---|---|---|---|---|

## Resume checkpoint

- Accepted implementation: <run id, patch>.
- Working tree / build: <state>.
- Running processes: <none / which, and their logs>.
- Next step: <one concrete step>.

## Runs

<!-- tests/search_experiment.py appends one entry per run below; add a short note after each. -->
