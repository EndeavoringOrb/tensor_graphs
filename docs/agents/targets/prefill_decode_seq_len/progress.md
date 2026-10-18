# Prefill+decode sequence length within 5% gap: progress

Task: follow `docs/agents/targets/prefill_decode_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_seq_len`; logs, metadata and source patches live in `logs/targets/prefill_decode_seq_len/`.

## Fixed evaluation contract

- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --run-dir logs/gemma_output_runs/<run_id>`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM, OpenCL on. This file tracks the laptop; the Linux CUDA box (OpenCL off) needs its own baseline and table.
- Base revision: `5ab39db2` (includes this target's infrastructure: dominated-child cycle stripping, `RemoveIdentityContiguous`, `run_graph_logits`). P000 to P002 ran on the uncommitted equivalent.

## Comparison table

Calibration: P000 to P002 ran before automatic calibration existed (costs partly interpolated, `Qwen_OuterProd_NEON` unrecorded). Rerun the baseline with calibration before comparing.

Accepted implementation: cycle unrolling in relaxation bound and primal heuristic. N\* = undefined (N = 0 misses the target), frontier N_f = 0.

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

Gate references at N_f = 0:
- Baseline (P003, P004, P005):
  - Median normalized integral: 0.99768276
  - Spread (max - min): 0.00000973 (9.73e-6)
  - Median final gap: 0.99734328
  - Median first incumbent: 14.528 s
- Candidate cycle unrolling (P006, P007, P008):
  - Median normalized integral: 0.63729660
  - Spread (max - min): 0.00006909 (6.91e-5)
  - Improvement: 0.36038616 (>> baseline spread 9.73e-6)
  - Median final gap: 0.61964081 (down from 0.99734328)
  - Median first incumbent: 5.570 s (down from 14.528 s)
  - Median first incumbent cost: 89.7981 ms (down from 562.592 ms)
  - Final lower bound: 34.1555 ms (up from 1.4947 ms)
  - Verdict: ACCEPTED as frontier improvement at N_f = 0.

## Resume checkpoint

- Accepted implementation: cycle unrolling in relaxation bound and primal heuristic (committed).
- Working tree / build: clean on branch `cleanup`.
- Running processes: none.
- Next step: diagnose and close the remaining gap at N = 0 (0.6196 -> < 0.05). Incumbent is 89.80 ms (evaluated at leaf matches simulation to 5 decimals). Root lower bound is 34.16 ms (load=34.19 ms, interval=11.80 ms). The CP-SAT search explores 33k iterations but prunes 0 nodes via lower bound because the bound is never strengthened beyond 34.16 ms. Investigate why interval bound is only 11.8 ms when critical path is ~89 ms, why relaxation lambda=[0.99, 0.01] places load on GPU without memory/transfer feasibility, and how to improve bound propagation or primal-guided branching during search.

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
