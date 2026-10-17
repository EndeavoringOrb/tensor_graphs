# Prefill+decode sequence length within 5% gap: progress

Task: follow `docs/agents/targets/prefill_decode_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_seq_len`; logs, metadata and source patches live in `logs/targets/prefill_decode_seq_len/`.

## Fixed evaluation contract

- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --run-dir logs/gemma_output_runs/<run_id>`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM.
- Base revision: `5fafb33e` plus the uncommitted `cleanup` branch work and this target's infrastructure (dominated-child cycle stripping, `RemoveIdentityContiguous`, `run_graph_logits`).

## Comparison table

| Run | Change | N | ok | final gap | gap target met | compile s | Verdict |
|---|---|---|---|---|---|---|---|
| P000_baseline_seq4 | — | 4 | no | 1 | no | 105.6 | failed: no feasible plan by the search deadline |
| P001_baseline_seq1 | — | 1 | no* | 0.9962 | no | 105.8 | *check-harness bug (expected 2 graphs, 3 compiled); rechecked offline: 3/3 graphs correct |
| P002_baseline_seq0 | — | 0 | no* | 0.9973 | no | 105.6 | *check-harness bug (N taken from the session); rechecked offline: 1/1 graph correct |

## Resume checkpoint

- Accepted implementation: infrastructure only (no optimization yet).
- Working tree / build: see the latest run's patch.
- Running processes: none.
- Next step: rerun N = 0 with the fixed check (expect ok=true, gap about 0.997), then cut the gap at N = 0. The full bucket keeps 18 cyclic components after `dominated_child`, which switches off `RelaxationBound` (root bound 0.02 to 1.5 against an incumbent of 562). Inspect them with `TG_CYCLE_DIAGNOSTICS=1` from the repository root (diagnostics from another directory miss `benchmarks/records.bin` and plan a different graph). Then make the relaxation and the root primal heuristic tolerate cycles.

## Runs

<!-- tests/search_experiment.py appends one entry per run below; add a short note after each. -->

### P000_baseline_seq4
- In progress: 2026-10-17T17:52:23.167316-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P000_baseline_seq4.json`; log `logs/targets/prefill_decode_seq_len/P000_baseline_seq4.log`; patch `logs/targets/prefill_decode_seq_len/P000_baseline_seq4.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 4 --run-dir logs/gemma_output_runs/P000_baseline_seq4`
- Completed: 2026-10-17T17:54:09.048233-04:00; exit 1; runner elapsed 105.859 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.
- Note (baseline): the 9-bucket compile reached the search about 21 s in. In about 84 s of search (1683 iterations, about 20/s) there was no incumbent. `[RelaxationBound]` was skipped for every cyclic bucket (8 of 9; the zero-weight full bucket also has 18 cyclic components), so the root bound stayed at 0.0085 to 0.0187. Of the 78 left-branch conflicts, 77 came from `CycleAvoidancePropagator`. After `dominated_child`: 2231 cyclic components in each prefill/decode bucket with t >= 2, 295 in bucket 6 and 18 in the full bucket. Verdict: failed (it fails loud as specified). This is the baseline to beat.

### P001_baseline_seq1
- In progress: 2026-10-17T18:03:40.838850-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P001_baseline_seq1.json`; log `logs/targets/prefill_decode_seq_len/P001_baseline_seq1.log`; patch `logs/targets/prefill_decode_seq_len/P001_baseline_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --run-dir logs/gemma_output_runs/P001_baseline_seq1`
- Completed: 2026-10-17T18:05:46.043772-04:00; exit 1; runner elapsed 125.187 s.
  `[SearchEngine volume] iterations=61636 elapsed=103.2530885 total-domain-bits=1092521.3898895504 remaining-domain-bits=1091440.487965859 log2-remaining-fraction=-1080.9019236913882 eliminated-bits=1080.9019236913882 remaining-nodes=52736 exhausted=false incomplete-branches=6644`
  `[Program gap] primal-dual-integral=119.73222460241037 normalized=0.99776853835341972 final-gap=0.99620486689607124 final-incumbent=54.089011999999997 final-bound=0.20527500000000001 first-incumbent-elapsed=41.987529993057251 samples=395`
  `[SearchEngine] New best cost: 54.089 at iteration 21994 source=search event-epoch=1792274662.9906521 program-deadline=0`
- First incumbent: 41.987529993 s, cost 54.089; best cost 54.089012.

### P002_baseline_seq0
- In progress: 2026-10-17T18:06:37.223309-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P002_baseline_seq0.json`; log `logs/targets/prefill_decode_seq_len/P002_baseline_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P002_baseline_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P002_baseline_seq0`
- Completed: 2026-10-17T18:08:29.301762-04:00; exit 1; runner elapsed 112.047 s.
  `[SearchEngine volume] iterations=245400 elapsed=103.98266940000001 total-domain-bits=353068.18895443797 remaining-domain-bits=352021.72062216955 log2-remaining-fraction=-1046.4683322684141 eliminated-bits=1046.4683322684141 remaining-nodes=242472 exhausted=false incomplete-branches=2916`
  `[Program gap] primal-dual-integral=119.72081677830251 normalized=0.99767347315252086 final-gap=0.9973424055468546 final-incumbent=562.407104 final-bound=1.49465 first-incumbent-elapsed=14.183868646621704 samples=415`
  `[SearchEngine] New best cost: 562.407 at iteration 8840 source=search event-epoch=1792274811.5608895 program-deadline=0`
- First incumbent: 14.183868647 s, cost 562.407; best cost 562.407104.

- Note (P001, N = 1, 3 graphs): the first incumbent came at 42.0 s, final incumbent 54.089, final bound 0.205, gap 0.9962, normalized integral 0.998. The compile itself passed. The check failed only because `expectedGraphRanges` wrongly assumed the session folds the full bucket into prefill 1; it does not (3 graphs). After the fix, the saved plan loaded without replanning ("Using cached compilation") and all 3 graphs matched transformers (max abs error 3.8e-5). Cycle components per bucket: 258 -> 222 (prefill/decode), 258 -> 18 (full). Relaxation skipped for both weighted buckets (cyclic).
