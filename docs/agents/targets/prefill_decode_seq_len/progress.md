# Prefill+decode sequence length within 5% gap: progress

Task: follow `docs/agents/targets/prefill_decode_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_seq_len`; logs, metadata and source patches live in `logs/targets/prefill_decode_seq_len/`.

## Fixed evaluation contract

- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --run-dir logs/gemma_output_runs/<run_id>`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM, OpenCL on. This file tracks the laptop; the Linux CUDA box (OpenCL off) needs its own baseline and table.
- Base revision: `5ab39db2` (includes this target's infrastructure: dominated-child cycle stripping, `RemoveIdentityContiguous`, `run_graph_logits`). P000 to P002 ran on the uncommitted equivalent.

## Comparison table

Calibration: P000 to P002 ran before automatic calibration existed (costs partly interpolated, `Qwen_OuterProd_NEON` unrecorded). Rerun the baseline with calibration before comparing.

Accepted implementation: storage_rep dependency propagation + unneeded storage dependency pruning in containment chain regions. N\* = 1 (final gap 1.24% < 5%), frontier N_f = 2.

| Run | Change | N | ok | normalized integral | final gap | gap target met | first incumbent s | eliminated bits | compile s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| P000_baseline_seq4 | — | 4 | no | — | 1 | no | none | — | 105.6 | failed: no feasible plan by the search deadline |
| P001_baseline_seq1 | — | 1 | no* | 0.9978 | 0.9962 | no | 41.99 | 1080.9 | 105.8 | *check-harness bug (expected 2 graphs, 3 compiled); rechecked offline: 3/3 graphs correct |
| P002_baseline_seq0 | — | 0 | no* | 0.9977 | 0.9973 | no | 14.18 | 1046.5 | 105.6 | *check-harness bug (N taken from the session); rechecked offline: 1/1 graph correct |
| P003_calibrated_seq0 | automatic calibration | 0 | yes | 0.9977 | 0.9973 | no | 14.84 | 1046.3 | 105.6 | baseline reference 1 of 3 at N_f = 0 |
| P004_calibrated_seq0 | baseline repeat | 0 | yes | 0.9977 | 0.9973 | no | 14.41 | 1046.3 | 105.6 | baseline reference 2 of 3 at N_f = 0 |
| P005_calibrated_seq0 | baseline repeat | 0 | yes | 0.9977 | 0.9973 | no | 14.53 | 1046.3 | 105.6 | baseline reference 3 of 3 at N_f = 0 |
| P006_cycle_unroll_seq0 | cycle unrolling | 0 | yes | 0.6373 | 0.6196 | no | 5.59 | 1185.6 | 105.2 | candidate 1 of 3 at N_f = 0 |
| P007_cycle_unroll_seq0 | cycle unrolling repeat | 0 | yes | 0.6373 | 0.6196 | no | 5.57 | 1185.6 | 105.2 | candidate 2 of 3 at N_f = 0 |
| P008_cycle_unroll_seq0 | cycle unrolling repeat | 0 | yes | 0.6373 | 0.6196 | no | 5.56 | 1185.7 | 105.2 | candidate 3 of 3 at N_f = 0 |
| P009_relaxation_bound_gemma0 | storage_rep + group pruning in relaxation | 0 | yes | 0.1043 | 0.0450 | yes | 22.46 | 1185.6 | 105.2 | candidate 1 of 3 at N* = 0, gap target met |
| P010_relaxation_bound_gemma0 | storage_rep + group pruning repeat | 0 | yes | 0.1070 | 0.0450 | yes | 22.80 | 1185.6 | 105.2 | candidate 2 of 3 at N* = 0, gap target met |
| P011_relaxation_bound_gemma0 | storage_rep + group pruning repeat | 0 | yes | 0.1048 | 0.0450 | yes | 22.52 | 1185.6 | 105.2 | candidate 3 of 3 at N* = 0, gap target met |
| P012_baseline_seq1 | storage_rep + group pruning at N=1 | 1 | yes | 0.1841 | 0.0124 | yes | 35.87 | 1204.1 | 106.1 | candidate 1 of 3 at N* = 1, gap target met |
| P013_baseline_seq1 | storage_rep + group pruning repeat | 1 | yes | 0.1839 | 0.0124 | yes | 35.84 | 1204.2 | 105.7 | candidate 2 of 3 at N* = 1, gap target met |
| P014_baseline_seq1 | storage_rep + group pruning repeat | 1 | yes | 0.1801 | 0.0124 | yes | 35.37 | 1204.2 | 105.8 | candidate 3 of 3 at N* = 1, gap target met |
| P015_baseline_seq2 | baseline at N=2 | 2 | yes | 0.8083 | 0.5787 | no | 80.39 | 5441.5 | 106.1 | baseline reference 1 of 3 at N_f = 2 |
| P016_baseline_seq2 | baseline repeat | 2 | yes | 0.8127 | 0.5787 | no | 81.64 | 5440.2 | 105.7 | baseline reference 2 of 3 at N_f = 2 |
| P017_baseline_seq2 | baseline repeat | 2 | yes | 0.8128 | 0.5787 | no | 81.67 | 5440.5 | 105.8 | baseline reference 3 of 3 at N_f = 2 |

Gate references at N* = 0:
- Baseline (P003, P004, P005):
  - Median normalized integral: 0.99768276
  - Median final gap: 0.99734328
  - Median first incumbent: 14.528 s
- Cycle unrolling candidate (P006, P007, P008):
  - Median normalized integral: 0.63729660
  - Median final gap: 0.61964081
  - Final lower bound: 34.1555 ms
- Accepted relaxation bound enhancement (P009, P010, P011):
  - Median normalized integral: 0.10481616 (down from 0.63729660)
  - Spread (max - min): 0.00268695
  - Median final gap: 0.04495653 (4.50%, down from 0.61964081)
  - Final lower bound: 85.76107 ms (up from 34.1555 ms, proved within 4.5% of optimal incumbent 89.79808 ms)
  - Gap target met: true for all 3 runs (final gap < 0.05).
  - Verdict: ACCEPTED as scaling change establishing N* = 0.

Gate references at N* = 1:
- Accepted relaxation bound enhancement (P012, P013, P014):
  - Median normalized integral: 0.18388988
  - Spread (max - min): 0.00407393
  - Median final gap: 0.01240610 (1.24%, proved well within 5.0% target)
  - Final lower bound: 8.760345 ms (incumbent: 8.870392 ms)
  - Gap target met: true for all 3 runs (final gap 1.24% < 0.05).
  - All 3 compiled graphs passed logits check (max abs error 3.7e-5).
  - Verdict: ACCEPTED as scaling change establishing N* = 1.

Gate references at N_f = 2:
- Baseline (P015, P016, P017):
  - Median normalized integral: 0.81269900
  - Spread (max - min): 0.00449282
  - Median final gap: 0.57874629 (57.87%)
  - Median first incumbent: 81.64 s
  - Final lower bound: 3.991547 ms (incumbent: 9.475399 ms)
  - All 5 compiled graphs passed logits check (ok=true).
  - Verdict: Frontier baseline established at N_f = 2. Target requirement: median normalized integral < 0.80820618 or final gap < 0.05.

## Resume checkpoint

- Accepted implementation: storage_rep dependency propagation + unneeded storage dependency pruning in containment chain regions.
- Target status: N* = 1 established. Frontier N_f = 2 (5 buckets: prefill [0, 1), decode [0, 1), prefill [0, 2), decode [1, 2), full [0, 2)).
- Working tree / build: clean on branch `cleanup`.
- Next step: diagnose lower bound at N = 2 (lower bound 3.99 ms vs makespan 9.48 ms; buckets 0, 1, 3 chain length collapsed to 1; bucket 2 bound 5.34 ms). Optimize lower bound across prefill/decode buckets at N = 2.

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

### P003_calibrated_seq0
- In progress: 2026-10-17T18:40:52.607399-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P003_calibrated_seq0.json`; log `logs/targets/prefill_decode_seq_len/P003_calibrated_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P003_calibrated_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P003_calibrated_seq0`
- Completed: 2026-10-17T18:43:47.891501-04:00; exit 0; runner elapsed 175.265 s.
  `[SearchEngine volume] iterations=220810 elapsed=104.09889080000001 total-domain-bits=351804.96059415169 remaining-domain-bits=350758.66632756399 log2-remaining-fraction=-1046.2942665877054 eliminated-bits=1046.2942665877054 remaining-nodes=217825 exhausted=false incomplete-branches=2973`
  `[Program gap] primal-dual-integral=119.72277086707581 normalized=0.99768975722563169 final-gap=0.9973432808878987 final-incumbent=562.59240699999998 final-bound=1.49465 first-incumbent-elapsed=14.8369140625 samples=414`
  `[SearchEngine] New best cost: 562.592 at iteration 9076 source=search event-epoch=1792276923.0560851 program-deadline=0`
- First incumbent: 14.836914062 s, cost 562.592; best cost 562.592407.
- Note (P003, N = 0): the first run with automatic calibration (`records.bin` backed up to `benchmarks/records.bin.20261017-before-auto-cal`). Calibration round 1 benchmarked 137 configurations, including the previously unrecorded `Qwen_OuterProd_NEON`; round 2 found none pending. The compile printed no missing-record warning. The check loaded the plan and matched transformers (max abs error 3.8e-5). Gap unchanged from P002 (0.9973; incumbent 562.59, bound 1.49), as expected: calibration fixes costs, not the search.

### P004_calibrated_seq0
- In progress: 2026-10-17T19:40:11.765695-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P004_calibrated_seq0.json`; log `logs/targets/prefill_decode_seq_len/P004_calibrated_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P004_calibrated_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P004_calibrated_seq0`
- Completed: 2026-10-17T19:42:25.115838-04:00; exit 0; runner elapsed 133.313 s.
  `[SearchEngine volume] iterations=227295 elapsed=104.0782961 total-domain-bits=351804.96059415169 remaining-domain-bits=350758.66632756399 log2-remaining-fraction=-1046.2942665877054 eliminated-bits=1046.2942665877054 remaining-nodes=224310 exhausted=false incomplete-branches=2973`
  `[Program gap] primal-dual-integral=119.72160356342445 normalized=0.99768002969520375 final-gap=0.9973432808878987 final-incumbent=562.59240699999998 final-bound=1.49465 first-incumbent-elapsed=14.406033515930176 samples=415`
  `[SearchEngine] New best cost: 562.592 at iteration 9076 source=search event-epoch=1792280427.6969168 program-deadline=0`
- First incumbent: 14.406033516 s, cost 562.592; best cost 562.592407.

### P005_calibrated_seq0
- In progress: 2026-10-17T19:42:28.929347-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P005_calibrated_seq0.json`; log `logs/targets/prefill_decode_seq_len/P005_calibrated_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P005_calibrated_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P005_calibrated_seq0`
- Completed: 2026-10-17T19:44:29.012476-04:00; exit 0; runner elapsed 120.047 s.
  `[SearchEngine volume] iterations=226509 elapsed=104.08385869999999 total-domain-bits=351804.96059415169 remaining-domain-bits=350758.66632756399 log2-remaining-fraction=-1046.2942665877054 eliminated-bits=1046.2942665877054 remaining-nodes=223524 exhausted=false incomplete-branches=2973`
  `[Program gap] primal-dual-integral=119.72193105029798 normalized=0.99768275875248313 final-gap=0.9973432808878987 final-incumbent=562.59240699999998 final-bound=1.49465 first-incumbent-elapsed=14.527688503265381 samples=416`
  `[SearchEngine] New best cost: 562.592 at iteration 9076 source=search event-epoch=1792280564.8062317 program-deadline=0`
- First incumbent: 14.527688503 s, cost 562.592; best cost 562.592407.

### P006_cycle_unroll_seq0
- In progress: 2026-10-17T20:10:28.480173-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P006_cycle_unroll_seq0.json`; log `logs/targets/prefill_decode_seq_len/P006_cycle_unroll_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P006_cycle_unroll_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P006_cycle_unroll_seq0`
- Completed: 2026-10-17T20:12:36.212037-04:00; exit 0; runner elapsed 127.703 s.
  `[SearchEngine volume] iterations=33665 elapsed=104.08484110000001 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.33181455778 log2-remaining-fraction=-1185.6287795939134 eliminated-bits=1185.6287795939134 remaining-nodes=33666 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=76.4813036433966 normalized=0.63734419702830503 final-gap=0.61964080969214486 final-incumbent=89.798079999999999 final-bound=34.155524999999997 first-incumbent-elapsed=5.5852639675140381 samples=397`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792282235.5647409 program-deadline=1792282334.9795067`
- First incumbent: 20.585234165 s, cost 89.7981; best cost 89.79808.

### P007_cycle_unroll_seq0
- In progress: 2026-10-17T20:14:46.785238-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P007_cycle_unroll_seq0.json`; log `logs/targets/prefill_decode_seq_len/P007_cycle_unroll_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P007_cycle_unroll_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P007_cycle_unroll_seq0`
- Completed: 2026-10-17T20:16:46.686289-04:00; exit 0; runner elapsed 119.875 s.
  `[SearchEngine volume] iterations=33741 elapsed=104.09065889999999 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.32337680028 log2-remaining-fraction=-1185.6372173514101 eliminated-bits=1185.6372173514101 remaining-nodes=33742 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=76.475592230450729 normalized=0.6372966019204227 final-gap=0.61964080969214486 final-incumbent=89.798079999999999 final-bound=34.155524999999997 first-incumbent-elapsed=5.5702481269836426 samples=398`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792282494.2557721 program-deadline=1792282593.6855488`
- First incumbent: 20.570223331 s, cost 89.7981; best cost 89.79808.

### P008_cycle_unroll_seq0
- In progress: 2026-10-17T20:17:01.229098-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P008_cycle_unroll_seq0.json`; log `logs/targets/prefill_decode_seq_len/P008_cycle_unroll_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P008_cycle_unroll_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P008_cycle_unroll_seq0`
- Completed: 2026-10-17T20:19:00.696024-04:00; exit 0; runner elapsed 119.437 s.
  `[SearchEngine volume] iterations=33898 elapsed=104.0874879 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.30603039521 log2-remaining-fraction=-1185.654563756485 eliminated-bits=1185.654563756485 remaining-nodes=33899 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=76.473012885593988 normalized=0.63727510737994986 final-gap=0.61964080969214486 final-incumbent=89.798079999999999 final-bound=34.155524999999997 first-incumbent-elapsed=5.5634667873382568 samples=397`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792282628.6623824 program-deadline=1792282728.0989373`
- First incumbent: 20.563445091 s, cost 89.7981; best cost 89.79808.
- Note (P006-P008, candidate cycle unrolling, N = 0): with cycle unrolling in relaxation bound and primal heuristic, the full bucket (18 cyclic components) is unrolled into acyclic levels, enabling the LP relaxation bound and primal heuristic to run on the full bucket. Root heuristic delivers an incumbent of 89.7981 ms (vs baseline 562.59 ms) within 5.57s of search. Lower bound jumps from 1.49 ms to 34.16 ms. Gap drops from 99.73% to 61.96%. Normalized integral drops from 0.997683 to 0.637297. All 3 runs are consistent (spread 6.91e-5 vs baseline spread 9.73e-6). Accepted as frontier improvement at N_f = 0.

### P009_scc_strip_seq0
- In progress: 2026-10-17T21:16:32.221160-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P009_scc_strip_seq0.json`; log `logs/targets/prefill_decode_seq_len/P009_scc_strip_seq0.log`; patch `logs/targets/prefill_decode_seq_len/P009_scc_strip_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P009_scc_strip_seq0`
- Completed: 2026-10-17T21:18:33.401970-04:00; exit 0; runner elapsed 121.156 s.
  `[SearchEngine volume] iterations=33619 elapsed=104.0685525 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.33695658261 log2-remaining-fraction=-1185.6236375690787 eliminated-bits=1185.6236375690787 remaining-nodes=33620 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=76.516156674633663 normalized=0.6376346389552805 final-gap=0.61964080969214486 final-incumbent=89.798079999999999 final-bound=34.155524999999997 first-incumbent-elapsed=5.6768958568572998 samples=397`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792286199.4373157 program-deadline=1792286298.7604439`
- First incumbent: 20.676871777 s, cost 89.7981; best cost 89.79808.

### P009_relaxation_bound_gemma0
- In progress: 2026-10-18T01:22:47.840940-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P009_relaxation_bound_gemma0.json`; log `logs/targets/prefill_decode_seq_len/P009_relaxation_bound_gemma0.log`; patch `logs/targets/prefill_decode_seq_len/P009_relaxation_bound_gemma0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P009_relaxation_bound_gemma0`
- Completed: 2026-10-18T01:24:47.515289-04:00; exit 0; runner elapsed 119.625 s.
  `[SearchEngine volume] iterations=33346 elapsed=104.05015710000001 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.36797290208 log2-remaining-fraction=-1185.5926212496124 eliminated-bits=1185.5926212496124 remaining-nodes=33347 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.52082594182005 normalized=0.10434021618183374 final-gap=0.044956529137371255 final-incumbent=89.798079999999999 final-bound=85.761070000000004 first-incumbent-elapsed=7.4614849090576172 samples=390`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792300976.7391081 program-deadline=1792301074.2776484`
- First incumbent: 22.461459637 s, cost 89.7981; best cost 89.79808.

### P010_relaxation_bound_gemma0
- In progress: 2026-10-18T01:25:07.479062-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P010_relaxation_bound_gemma0.json`; log `logs/targets/prefill_decode_seq_len/P010_relaxation_bound_gemma0.log`; patch `logs/targets/prefill_decode_seq_len/P010_relaxation_bound_gemma0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P010_relaxation_bound_gemma0`
- Completed: 2026-10-18T01:27:07.020675-04:00; exit 0; runner elapsed 119.500 s.
  `[SearchEngine volume] iterations=33294 elapsed=104.06042739999999 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.37404709571 log2-remaining-fraction=-1185.5865470559802 eliminated-bits=1185.5865470559802 remaining-nodes=33295 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.843259995666577 normalized=0.10702716663055481 final-gap=0.044956529137371255 final-incumbent=89.798079999999999 final-bound=85.761070000000004 first-incumbent-elapsed=7.7990968227386475 samples=389`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792301116.7285728 program-deadline=1792301213.9294968`
- First incumbent: 22.799076080 s, cost 89.7981; best cost 89.79808.

### P011_relaxation_bound_gemma0
- In progress: 2026-10-18T01:27:15.847023-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P011_relaxation_bound_gemma0.json`; log `logs/targets/prefill_decode_seq_len/P011_relaxation_bound_gemma0.log`; patch `logs/targets/prefill_decode_seq_len/P011_relaxation_bound_gemma0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --run-dir logs/gemma_output_runs/P011_relaxation_bound_gemma0`
- Completed: 2026-10-18T01:29:15.320449-04:00; exit 0; runner elapsed 119.422 s.
  `[SearchEngine volume] iterations=33622 elapsed=104.0563576 total-domain-bits=351804.96059415169 remaining-domain-bits=350619.33656345733 log2-remaining-fraction=-1185.6240306943655 eliminated-bits=1185.6240306943655 remaining-nodes=33623 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.577939504334735 normalized=0.10481616253612279 final-gap=0.044956529137371255 final-incumbent=89.798079999999999 final-bound=85.761070000000004 first-incumbent-elapsed=7.5212869644165039 samples=390`
  `[SearchEngine] New best cost: 89.7981 at iteration 0 source=root-heuristic event-epoch=1792301244.7550669 program-deadline=1792301342.2338026`
- First incumbent: 22.521264315 s, cost 89.7981; best cost 89.79808.

### P012_baseline_seq1
- In progress: 2026-10-18T01:30:47.298480-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P012_baseline_seq1.json`; log `logs/targets/prefill_decode_seq_len/P012_baseline_seq1.log`; patch `logs/targets/prefill_decode_seq_len/P012_baseline_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --run-dir logs/gemma_output_runs/P012_baseline_seq1`
- Completed: 2026-10-18T01:32:50.012155-04:00; exit 0; runner elapsed 122.687 s.
  `[SearchEngine volume] iterations=6165 elapsed=103.75638069999999 total-domain-bits=1088482.0191580458 remaining-domain-bits=1087277.8960452699 log2-remaining-fraction=-1204.123112775851 eliminated-bits=1204.123112775851 remaining-nodes=6163 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=22.097555077296889 normalized=0.1841462923108074 final-gap=0.012406103360483003 final-incumbent=8.8703920000000007 final-bound=8.7603449999999992 first-incumbent-elapsed=20.867709636688232 samples=327`
  `[SearchEngine] New best cost: 8.87039 at iteration 0 source=root-heuristic event-epoch=1792301469.8709514 program-deadline=1792301554.0032692`
- First incumbent: 35.867682219 s, cost 8.87039; best cost 8.870392.
- Note (P012, N = 1, 3 graphs): candidate 1 of 3 at N = 1. Lower bound 8.7603 ms, incumbent 8.8704 ms, final gap 0.0124 (1.24%), normalized integral 0.1841. All 3 graphs passed logits check (max abs error 3.7e-5).

### P013_baseline_seq1
- In progress: 2026-10-18T01:33:08.381939-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P013_baseline_seq1.json`; log `logs/targets/prefill_decode_seq_len/P013_baseline_seq1.log`; patch `logs/targets/prefill_decode_seq_len/P013_baseline_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --run-dir logs/gemma_output_runs/P013_baseline_seq1`
- Completed: 2026-10-18T01:35:10.602392-04:00; exit 0; runner elapsed 122.187 s.
  `[SearchEngine volume] iterations=6195 elapsed=103.2490612 total-domain-bits=1088482.0191580458 remaining-domain-bits=1087277.8463718581 log2-remaining-fraction=-1204.1727861876134 eliminated-bits=1204.1727861876134 remaining-nodes=6193 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=22.066785539555916 normalized=0.18388987949629929 final-gap=0.012406103360483003 final-incumbent=8.8703920000000007 final-bound=8.7603449999999992 first-incumbent-elapsed=20.836553573608398 samples=327`
  `[SearchEngine] New best cost: 8.87039 at iteration 0 source=root-heuristic event-epoch=1792301610.9163272 program-deadline=1792301695.0797987`
- First incumbent: 35.836528540 s, cost 8.87039; best cost 8.870392.
- Note (P013, N = 1, 3 graphs): candidate 2 of 3 at N = 1 repeat. Lower bound 8.7603 ms, incumbent 8.8704 ms, final gap 0.0124 (1.24%), normalized integral 0.1839. All 3 graphs passed logits check (max abs error 3.7e-5).

### P014_baseline_seq1
- In progress: 2026-10-18T01:35:26.703454-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P014_baseline_seq1.json`; log `logs/targets/prefill_decode_seq_len/P014_baseline_seq1.log`; patch `logs/targets/prefill_decode_seq_len/P014_baseline_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --run-dir logs/gemma_output_runs/P014_baseline_seq1`
- Completed: 2026-10-18T01:37:29.075758-04:00; exit 0; runner elapsed 122.328 s.
  `[SearchEngine volume] iterations=6190 elapsed=103.75079119999999 total-domain-bits=1088482.0191580458 remaining-domain-bits=1087277.858674919 log2-remaining-fraction=-1204.1604831267614 eliminated-bits=1204.1604831267614 remaining-nodes=6188 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=21.608683732062595 normalized=0.18007236443385496 final-gap=0.012406103360483003 final-incumbent=8.8703920000000007 final-bound=8.7603449999999992 first-incumbent-elapsed=20.372697114944458 samples=329`
  `[SearchEngine] New best cost: 8.87039 at iteration 0 source=root-heuristic event-epoch=1792301748.9926541 program-deadline=1792301833.6199832`
- First incumbent: 35.372670889 s, cost 8.87039; best cost 8.870392.
- Note (P014, N = 1, 3 graphs): candidate 3 of 3 at N = 1 repeat. Lower bound 8.7603 ms, incumbent 8.8704 ms, final gap 0.0124 (1.24%), normalized integral 0.1801. All 3 graphs passed logits check (max abs error 3.7e-5). All 3 runs confirm N* = 1 met and accepted as scaling change. Frontier advances to N_f = 2.

### P015_baseline_seq2
- In progress: 2026-10-18T01:41:09.147315-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P015_baseline_seq2.json`; log `logs/targets/prefill_decode_seq_len/P015_baseline_seq2.log`; patch `logs/targets/prefill_decode_seq_len/P015_baseline_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --run-dir logs/gemma_output_runs/P015_baseline_seq2`
- Completed: 2026-10-18T01:44:49.637722-04:00; exit 0; runner elapsed 220.453 s.
  `[SearchEngine volume] iterations=598 elapsed=97.026780200000005 total-domain-bits=3697138.3298099763 remaining-domain-bits=3691696.8747256743 log2-remaining-fraction=-5441.455084301997 eliminated-bits=5441.455084301997 remaining-nodes=597 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=96.996872997724566 normalized=0.80830727498103805 final-gap=0.57874628815103191 final-incumbent=9.4753989999999995 final-bound=3.9915470000000002 first-incumbent-elapsed=65.393651485443115 samples=132`
  `[SearchEngine] New best cost: 9.4754 at iteration 0 source=root-heuristic event-epoch=1792302233.8040595 program-deadline=1792302273.4104476`
- First incumbent: 80.393611908 s, cost 9.4754; best cost 9.475399.
- Note (P015, N = 2, 5 graphs): baseline reference 1 of 3 at N_f = 2. First run with automatic calibration at N = 2 (round 1 benchmarked 550 pending configurations; round 2 found 0 pending). Lower bound 3.9915 ms, incumbent 9.4754 ms, final gap 0.5787 (57.87%), normalized integral 0.8083. All 5 graphs passed logits check (ok=true).

### P016_baseline_seq2
- In progress: 2026-10-18T01:45:22.474736-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P016_baseline_seq2.json`; log `logs/targets/prefill_decode_seq_len/P016_baseline_seq2.log`; patch `logs/targets/prefill_decode_seq_len/P016_baseline_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --run-dir logs/gemma_output_runs/P016_baseline_seq2`
- Completed: 2026-10-18T01:47:32.017626-04:00; exit 0; runner elapsed 129.500 s.
  `[SearchEngine volume] iterations=563 elapsed=97.586681600000006 total-domain-bits=3697138.3298099763 remaining-domain-bits=3691698.1197042014 log2-remaining-fraction=-5440.2101057749242 eliminated-bits=5440.2101057749242 remaining-nodes=562 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=97.523880242441237 normalized=0.81269900202034362 final-gap=0.57874628815103191 final-incumbent=9.4753989999999995 final-bound=3.9915470000000002 first-incumbent-elapsed=66.644696235656738 samples=128`
  `[SearchEngine] New best cost: 9.4754 at iteration 0 source=root-heuristic event-epoch=1792302397.2777092 program-deadline=1792302435.6330364`
- First incumbent: 81.644672871 s, cost 9.4754; best cost 9.475399.
- Note (P016, N = 2, 5 graphs): baseline reference 2 of 3 at N_f = 2. Lower bound 3.9915 ms, incumbent 9.4754 ms, final gap 0.5787 (57.87%), normalized integral 0.8127. All 5 graphs passed logits check (ok=true).

### P017_baseline_seq2
- In progress: 2026-10-18T01:47:50.803357-04:00
- Metadata `logs/targets/prefill_decode_seq_len/P017_baseline_seq2.json`; log `logs/targets/prefill_decode_seq_len/P017_baseline_seq2.log`; patch `logs/targets/prefill_decode_seq_len/P017_baseline_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --run-dir logs/gemma_output_runs/P017_baseline_seq2`
- Completed: 2026-10-18T01:49:59.937966-04:00; exit 0; runner elapsed 129.093 s.
  `[SearchEngine volume] iterations=570 elapsed=97.568926500000003 total-domain-bits=3697138.3298099763 remaining-domain-bits=3691697.8785368144 log2-remaining-fraction=-5440.4512731619179 eliminated-bits=5440.4512731619179 remaining-nodes=569 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=97.536010947755983 normalized=0.81280009123129981 final-gap=0.57874628815103191 final-incumbent=9.4753989999999995 final-bound=3.9915470000000002 first-incumbent-elapsed=66.673492908477783 samples=129`
  `[SearchEngine] New best cost: 9.4754 at iteration 0 source=root-heuristic event-epoch=1792302545.3287468 program-deadline=1792302583.6552844`
- First incumbent: 81.673462391 s, cost 9.4754; best cost 9.475399.
- Note (P017, N = 2, 5 graphs): baseline reference 3 of 3 at N_f = 2. Lower bound 3.9915 ms, incumbent 9.4754 ms, final gap 0.5787 (57.87%), normalized integral 0.8128. All 5 graphs passed logits check (ok=true). Three baseline runs establish N_f = 2 references (median normalized integral 0.8127, median final gap 0.5787).
