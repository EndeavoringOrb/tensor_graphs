# Prefill+decode sequence length within 5% gap: progress (Linux box)

Task: follow `docs/agents/targets/prefill_decode_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_seq_len:box`; logs, metadata and source patches live in `logs/targets/prefill_decode_seq_len/` (run ids prefixed `B`).

## Fixed evaluation contract

- Build: `.venv/bin/python build.py --targets bindings test --log-level INFO`
- Command: `.venv/bin/python tests/search_experiment.py --record prefill_decode_seq_len:box <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --ignore-checkpoints --run-dir logs/gemma_output_runs/<run_id>`
- Hardware: Linux box, 16 threads, 31 GB RAM, Quadro M4000 (CUDA), OpenCL off (`OPENCL_BY_PLATFORM["linux"] = False`).
- Base revision: `f5e0560` (branch `cleanup`).

## Comparison table

Accepted implementation: B053 (B051 + cached storage sizes in the e-node cost fixed point). N\* = 1, N_f = 2.

| Run | Change | N | ok | normalized integral | final gap | gap target met | first incumbent s | compile s | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| B000_baseline_seq0 | — (f5e0560) | 0 | no | (1.0) | 1 | no | 105.8 | 120.3 (killed) | baseline: compile overran the budget |
| B001_offset_fix_seq0 | view lifetimes in preferredOffset + fallback past allocations | 0 | yes | 0.6925 | 0.5858 | no | 30.9 | 117.5 | accepted (frontier, superseded by B002) |
| B002_parallel_ls_seq0 | B001 + parallel local-search starts + completion cycle guard | 0 | yes | 0.6571 | 0.5858 | no | 20.7 | 117.5 | accepted (frontier, superseded by B003) |
| B003_threshold_split_seq0 | B002 + trimmed supports, private values, threshold split in the load bound | 0 | yes | 0.5220 | 0.4189 | no | 21.3 | 117.6 | accepted (frontier, superseded by B004) |
| B004_refine_seq0 | B003 + nested splits refined after the root heuristic | 0 | yes | 0.4100 | 0.2733 | no | 21.4 | 117.5 | accepted (frontier, superseded by B005) |
| B005_fractional_seq0 | B004 + fractional supports (claims), reachability level jump | 0 | yes | 0.3139 | 0.1607 | no | 21.9 | 117.6 | accepted (frontier, superseded by B006) |
| B006_global_split_seq0 | B005 + global case split on hidden classes, dynamic positivity | 0 | yes | 0.2255 | 0.0481 | **yes** | 22.4 | 117.5 | accepted scaling change: N\* = 0 (superseded by B008) |
| B007_accepted_seq1 | B006 at N_f = 1 | 1 | no | (1.0) | 1 | no | — | 81.5 (crash) | failed: 70 s root relaxation, heuristic threw |
| B008_n1_fixes_seq1 | B006 + interval stride plateau stop, producible extraction, consistent byte-valued covering, validate before cache moves | 1 | yes | 0.4208 | 0.1426 | no | 38.9 | 118.0 | accepted (frontier, superseded by B010) |
| B009_n1_fixes_seq0 | B008 code at N\* = 0 (regression check) | 0 | yes | 0.2226 | 0.0481 | **yes** | 22.0 | 117.6 | no regression at N\* = 0 |
| B010_fast_validate_seq1 | B008 + complete-plan checks in reachability and cycle propagators | 1 | yes | 0.3277 | 0.1426 | no | 25.9 | 118.1 | accepted (frontier, superseded by B012) |
| B011_fast_validate_seq0 | B010 code at N\* = 0 (regression check) | 0 | yes | 0.1874 | 0.0481 | **yes** | 17.6 | 117.5 | no regression at N\* = 0 |
| B012_fast_saturation_seq1 | B010 + indexed ConsumerWeightReuseRule (saturation 8.4 -> 1.4 s) | 1 | yes | 0.2800 | 0.1426 | no | 19.2 | 118.0 | accepted (frontier, superseded by B014) |
| B013_fast_saturation_seq0 | B012 code at N\* = 0 (regression check) | 0 | yes | 0.1319 | 0.0481 | **yes** | 10.6 | 117.5 | no regression at N\* = 0 |
| B014_deferred_intervals_seq1 | B012 + interval bound after the root heuristic | 1 | yes | 0.2741 | 0.1426 | no | 12.5 | 118.0 | accepted (frontier, superseded by B016) |
| B015_deferred_intervals_seq0 | B014 code at N\* = 0 (regression check) | 0 | yes | 0.1036 | 0.0481 | **yes** | 7.0 | 117.5 | no regression at N\* = 0 |
| B016_flexible_windows_seq1 | B014 + flexible interval windows (every window of <= 8 links, best partition) | 1 | yes | 0.2695 | 0.1347 | no | 12.7 | 118.0 | accepted (frontier, superseded by B018) |
| B017_flexible_windows_seq0 | B016 code at N\* = 0 (regression check) | 0 | yes | 0.1041 | 0.0481 | **yes** | 7.1 | 117.5 | no regression at N\* = 0 |
| B018_window_moves_seq1 | B016 + window moves (splice each flexible window's exact selection into the plan) | 1 | yes | 0.2258 | 0.1130 | no | 12.5 | 118.1 | accepted (frontier, superseded by B022) |
| B019_window_moves_seq0 | B018 code at N\* = 0 (regression check) | 0 | yes | 0.1053 | 0.0481 | **yes** | 7.2 | 117.5 | no regression at N\* = 0 |
| B020_sliding_windows_seq1 | B018 + sliding windows (4 links step 2, 8 links step 4) in window moves | 1 | yes | 0.2266 | 0.1123 | no | 12.5 | 118.1 | rejected: integral worse (moves 5.7 s vs 2.6 s delay the better incumbent) |
| B021_sliding_windows_seq0 | B020 code at N\* = 0 | 0 | yes | 0.1042 | 0.0481 | **yes** | 7.1 | 117.5 | (reference only) |
| B022_fast_flexible_seq1 | B018 + flexible windows first and in parallel (per-thread weight tables), stride windows after the window moves | 1 | yes | 0.2095 | 0.1130 | no | 12.4 | 118.1 | accepted (frontier, superseded by B024) |
| B023_fast_flexible_seq0 | B022 code at N\* = 0 (regression check) | 0 | yes | 0.1035 | 0.0481 | **yes** | 7.0 | 117.5 | no regression at N\* = 0 |
| B024_global_window_moves_seq1 | B022 + exact-window pruning, best-found window selections, window moves across buckets with cache-set changes, local search starts capped | 1 | yes | 0.2014 | 0.1100 | no | 11.9 | 118.1 | accepted (frontier, superseded by B026) |
| B025_global_window_moves_seq0 | B024 code at N\* = 0 (regression check) | 0 | yes | 0.1050 | 0.0481 | **yes** | 7.2 | 117.5 | no regression at N\* = 0 |
| B026_cache_aware_windows_stage_seq1 | B024 + equivalent alternatives and zero-weight covering in window splices, one-machine value in exact windows | 1 | yes | 0.1809 | 0.0882 | no | 11.6 | 118.0 | accepted (frontier, superseded by B028) |
| B027_cache_aware_windows_stage_seq0 | B026 code at N\* = 0 (regression check) | 0 | yes | 0.1041 | 0.0481 | **yes** | 7.1 | 117.5 | no regression at N\* = 0 |
| B028_long_windows_seq1 | B026 + long windows (9-24 links, cut DP) ending at the flexible partition's boundaries | 1 | yes | 0.1731 | 0.0743 | no | 11.9 | 118.1 | accepted (frontier, superseded by B030) |
| B029_long_windows_seq0 | B028 code at N\* = 0 (regression check) | 0 | yes | 0.1042 | 0.0481 | **yes** | 7.1 | 117.5 | no regression at N\* = 0 |
| B030_cut_dp_stage_seq1 | B028 + one-machine value in the cut DP | 1 | yes | 0.1655 | 0.0660 | no | 11.8 | 118.1 | accepted (frontier, superseded by B032) |
| B031_cut_dp_stage_seq0 | B030 code at N\* = 0 (regression check) | 0 | yes | 0.1048 | 0.0481 | **yes** | 7.1 | 117.5 | no regression at N\* = 0 |
| B032_long_window_moves_seq1 | B030 + window moves on the long-window partition, excess-directed window search (two orders, local polish), tighter float margin | 1 | yes | 0.1543 | **0.0478** | **yes** | 11.9 | 118.1 | accepted (N\* = 1; superseded by B035) |
| B033_long_window_moves_seq0 | B032 code at N\* = 0 (regression check) | 0 | yes | 0.1042 | 0.0476 | **yes** | 7.1 | 117.5 | no regression at N = 0 |
| B034_baseline_seq2 | B032 code at N = 2 | 2 | no | 1.0 | 1.0 | no | - | 117.6 | no plan: decode buckets cyclic (SCCs of 7 > cap 6), heuristic unusable, search finds none |
| B035_cycle_groups_seq1 | B032 + SCC unroll cap 16, cyclic components as groups, members = reachable classes incl. level copies, longest containment chain, degenerate-decomposition fallback | 1 | yes | 0.1439 | **0.0478** | **yes** | 11.0 | 118.1 | ACCEPTED (N\* = 1, better integral) |
| B036_cycle_groups_seq0 | B035 code at N = 0 (regression check) | 0 | yes | 0.1028 | 0.0483 | **yes** | 6.9 | 117.6 | no regression at N = 0 |
| B037_cycle_groups_seq2 | B035 code at N = 2 | 2 | no | 1.0 | 1.0 | no | - | 11.5 | crashed in planning: Invalid kernel UID KernelId(4294967297) (OpenCL enabled; not reproduced in B038) |
| B038_cycle_groups_seq2_rerun | B035 code at N = 2 (rerun) | 2 | no | 1.0 | 1.0 | no | - | 120.2 | no plan: all heuristic plans fail validation (cache requirement / view offsets); root bound 7.41 |
| B039_two_estimates_seq1 | B035 + cache-only marking for classes without a base, starts with cheap and region estimates, validation reserve capped at 20 s | 1 | yes | 0.1470 | 0.0478 | **yes** | 11.4 | 118.1 | N\* = 1 kept (first incumbent 0.4 s later than B035) |
| B040_two_estimates_seq0 | B039 code at N = 0 | 0 | yes | 0.1010 | 0.0483 | **yes** | 6.6 | 117.6 | no regression at N = 0 |
| B041_two_estimates_seq2 | B039 code at N = 2 | 2 | no | 1.0 | 1.0 | no | - | 120.2 | plan 16.43 found (78 s), but the compile ran past 120 s (stride intervals and window searches without deadlines): killed |
| B042_deadlines_seq2 | B041 + deadline for stride intervals, 2 s per window search | 2 | yes | 0.8237 | 0.4226 | no | 77.2 | 119.4 | first complete N = 2 run (superseded by B045) |
| B043_deadlines_seq1 | B042 code at N = 1 | 1 | yes | 0.1459 | 0.0478 | **yes** | 11.4 | 118.0 | N\* = 1 kept |
| B044_early_root_seq2 | B042 + root splits deferred when the load maximization takes > 1 s, region starts 3 s | 2 | no | 1.0 | 1.0 | no | - | 120.2 | plan 16.43 at ~53 s, then a stride interval pass ran past 120 s: killed |
| B045_refine_deadlines_seq2 | B044 + deadline inside interval evaluation and the global split's candidates | 2 | yes | 0.6885 | 0.3839 | no | 53.0 | 119.2 | accepted (frontier, superseded by B051) |
| B046_refine_deadlines_seq1 | B045 code at N = 1 | 1 | yes | 0.1457 | 0.0478 | **yes** | 11.3 | 118.1 | N\* = 1 kept |
| B047_pattern_race_seq2 | B045 + fusion patterns' candidate kernels resolved before the shared pattern set is used (race fix) | 2 | yes | 0.6909 | 0.3839 | no | 53.4 | 118.7 | kept (correctness; same values as B045 within noise) |
| B048_pattern_race_seq1 | B047 code at N = 1 | 1 | yes | 0.1466 | 0.0478 | **yes** | 11.4 | 118.1 | N\* = 1 kept |
| B049_deferred_load_seq2 | B047 + root load maximization deferred after the first plan when one evaluation takes > 0.1 s | 2 | yes | 0.6941 | 0.3839 | no | 44.0 | 119.3 | rejected: incumbent 9 s earlier but the bound stays 0 until the deferred loads finish (integral unchanged) |
| B050_deferred_load_seq1 | B049 code at N = 1 | 1 | yes | 0.1457 | 0.0478 | **yes** | 11.3 | 118.1 | (reference only) |
| B051_shared_identical_seq2 | B047 + a bucket identical to an earlier one takes its saturated and reduced e-graph | 2 | yes | 0.6754 | 0.3824 | no | 50.7 | 119.6 | accepted (frontier, superseded by B053) |
| B052_shared_identical_seq1 | B051 code at N = 1 | 1 | yes | 0.1469 | 0.0478 | **yes** | 11.5 | 118.0 | N\* = 1 kept |
| B053_size_cache_seq2 | B051 + storage sizes computed once per class / enode in the e-node cost fixed point | 2 | yes | 0.6489 | 0.3804 | no | 45.7 | 118.4 | ACCEPTED frontier change at N_f = 2 |
| B054_size_cache_seq1 | B053 code at N = 1 | 1 | yes | 0.1457 | 0.0478 | **yes** | 11.3 | 118.0 | N\* = 1 kept |

## Resume checkpoint

- Accepted: B053 (patch `logs/targets/prefill_decode_seq_len/B053_size_cache_seq2.patch`). N\* = 1 (B054: final gap 0.0478, normalized 0.1457). N = 0 met (B040: 0.0483). N_f = 2: B053 normalized 0.6489, final gap 0.3804 (bound 10.18, incumbent 16.43 at 46 s). Gold saturation dumps (regenerated from the current tree, see B047) are in tests/data/gold_saturation/. Working tree uncommitted on `cleanup` (the agent makes no commits).
- N_f = 1 analysis (B016 code): the decode plan (14.869) is a serial chain: GPU busy 12.16 ms, CPU 4.46 ms, dependency-only critical path 11.16 ms, GPU idle 2.71 ms; the critical chain is 11.09 ms of GPU ops (111 cuBLAS dots ~9.4 ms) + 3.78 ms of CPU ops (248 CPU<->GPU activation copies, 1.74 ms, RMS-norm pieces ~1.8 ms). All-GPU plans are worse (GPU load 16.3 ms: many small kernels). Local search (single-class switches) cannot reach coordinated moves (e.g. k/v projections on the CPU in parallel with q on the GPU need the CPU-side weights cached and CPU RoPE/K-norm too); even allowing every cache-possible base it stays at 14.869. Per chain link the plan spends 10.45 ms where stride-1 windows prove 7.08 ms: parallel branches belonged to no fine window (fixed partly by flexible windows, +0.12). Rough per-layer estimate of the optimum ~13.8. To reach 5% both sides must move: e.g. (a) a coordinated-move neighbourhood (re-optimize one layer's window exactly with the rest fixed, using the exact window enumeration as a selector) with dynamic cache additions; (b) window values that respect the CPU/GPU stage structure (a window's GPU segment cannot start before its CPU segment ends).

## Runs

<!-- tests/search_experiment.py appends one entry per run below; add a short note after each. -->

### B000_baseline_seq0
- In progress: 2026-10-20T09:04:22.682842-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B000_baseline_seq0.json`; log `logs/targets/prefill_decode_seq_len/B000_baseline_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B000_baseline_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B000_baseline_seq0`
- Completed: 2026-10-20T09:06:32.664156-04:00; exit 1; runner elapsed 129.978 s.
  `[SearchEngine volume] iterations=274 elapsed=107.954902632 total-domain-bits=524409.59663406154 remaining-domain-bits=523951.26373278775 log2-remaining-fraction=-458.33290127379587 eliminated-bits=458.33290127379587 remaining-nodes=275 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 283.522 at iteration 0 source=root-heuristic event-epoch=1792501578.0941322 program-deadline=1792501589.3169522`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 108.777179956 s, cost 283.522; best cost 283.522.
- Note (baseline, N = 0, Linux box): saturation 8.5 s, root propagation 3.9 s, relaxation bound 114.76 (load bound, lambda = [0.99, 0.01]; interval 66.6). The root heuristic took 92.8 s: 14 of its 15 validations (about 5.5 s each) were rejected by `MemoryNoOverlapPropagator` while completing offsets, and the 15th gave 283.52 at 105.8 s. The tryPlan completion fell back to the domain maximum for conflicting offsets, so the CPU arena became the whole 29.8 GB cap; materializing it overran the 3 s margin and the compile was killed at 120.3 s. Verdict: failed run (baseline).

### B001_offset_fix_seq0
- In progress: 2026-10-20T09:29:47.951202-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B001_offset_fix_seq0.json`; log `logs/targets/prefill_decode_seq_len/B001_offset_fix_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B001_offset_fix_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B001_offset_fix_seq0`
- Completed: 2026-10-20T09:32:31.281943-04:00; exit 0; runner elapsed 163.329 s.
  `[SearchEngine volume] iterations=4734 elapsed=108.05788293099999 total-domain-bits=524409.59663406154 remaining-domain-bits=523646.17402440723 log2-remaining-fraction=-763.42260965431342 eliminated-bits=763.42260965431342 remaining-nodes=4735 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=83.096317792585324 normalized=0.69246931493821107 final-gap=0.58575008974987897 final-incumbent=277.02822900000001 final-bound=114.75891900000001 first-incumbent-elapsed=30.91444730758667 samples=320`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792503028.1271925 program-deadline=1792503114.2127559`
- First incumbent: 33.914436579 s, cost 277.028; best cost 277.028229.
- Note (B001, N = 0): hypothesis: `HeuristicBrancher::preferredOffset` extended only a view's base lifetime to its last reader, so a view's own offset record (e.g. a REPEAT read by the next op) looked dead and its storage was offered to the reader's output, which `MemoryNoOverlapPropagator` rejects. Extending the view's record too, and completing a conflicting offset at the first page past every fixed allocation of its memory space (before the domain end), makes the first heuristic plan valid: incumbent 277.03 at 30.9 s (was 283.52 at 105.8 s), arena 1.3 GB (was 29.8 GB), compile 117.5 s within budget, logits ok. Gap 0.5858, normalized 0.6925 (baseline failed). Heuristic-only change (placement preference); bounds untouched. Verdict: ACCEPTED as frontier change at N_f = 0, pending the correctness gate.

### B002_parallel_ls_seq0
- In progress: 2026-10-20T10:11:19.374367-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B002_parallel_ls_seq0.json`; log `logs/targets/prefill_decode_seq_len/B002_parallel_ls_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B002_parallel_ls_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B002_parallel_ls_seq0`
- Completed: 2026-10-20T10:13:38.315316-04:00; exit 0; runner elapsed 138.939 s.
  `[SearchEngine volume] iterations=6451 elapsed=108.225898739 total-domain-bits=524409.59663406154 remaining-domain-bits=523642.72650170716 log2-remaining-fraction=-766.87013235437917 eliminated-bits=766.87013235437917 remaining-nodes=6452 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=78.853402565144364 normalized=0.65711168804286968 final-gap=0.58575008974987897 final-incumbent=277.02822900000001 final-bound=114.75891900000001 first-incumbent-elapsed=20.672042608261108 samples=359`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792505509.3126912 program-deadline=1792505605.6406591`
- First incumbent: 23.672032118 s, cost 277.028; best cost 277.028229.
- Note (B002, N = 0): local search ran its 12 starts one after another (11.3 s on one bucket). `improvePrimalPlans` now runs one task per (start, bucket) on 16 threads: 1.5 s. A cycle guard in the local-search completion skips alternatives whose children are classes being completed in the same move (a COPY_TO back to the moved class made the CPU->GPU logits move score `makespan=inf`). First incumbent 20.7 s (was 30.9), same plan 277.03, normalized 0.6571 (was 0.6925), final gap unchanged 0.5858. Gate: oracle 41/41, search_gap, the four native suites pass. Verdict: ACCEPTED as frontier change at N_f = 0.
- Analysis (N = 0 gap): the GPU-side logits dot (4.9 ms) exists, but its transposed fp32 embedding can only be produced by copying the CPU transpose (76 ms on the CPU engine, after a 73 ms CPU transpose), so the CPU route (cast 40 + transposes 73 + dots 54 after the 54 ms storage load) is cheaper and 277 is probably within a few % of optimal. The bound is the weak side: at lambda ~ (1, 0) the final (logits) region is worth only ~8 because the logits are a concat of chunk dots whose supports overlap (each chunk's GPU-side alternative brings shared classes, e.g. the GPU copy of the hidden state, into its support), so only one chunk is charged. The LP relaxation of the unrolled bucket is 1.06 (sharing integrality gap), and HiGHS MILP reaches dual 85.6 in 300 s, so off-the-shelf MIP does not help.

### B003_threshold_split_seq0
- In progress: 2026-10-20T11:11:44.013174-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B003_threshold_split_seq0.json`; log `logs/targets/prefill_decode_seq_len/B003_threshold_split_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B003_threshold_split_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B003_threshold_split_seq0`
- Completed: 2026-10-20T11:14:03.330926-04:00; exit 0; runner elapsed 139.316 s.
  `[SearchEngine volume] iterations=6613 elapsed=107.954716418 total-domain-bits=524409.59663406154 remaining-domain-bits=523642.49055921135 log2-remaining-fraction=-767.10607485019136 eliminated-bits=767.10607485019136 remaining-nodes=6614 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=62.636034293338597 normalized=0.52196695244448832 final-gap=0.41886103599933139 final-incumbent=277.02822900000001 final-bound=160.99189799999999 first-incumbent-elapsed=21.290449857711792 samples=356`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792509134.7062306 program-deadline=1792509230.4157927`
- First incumbent: 24.290437937 s, cost 277.028; best cost 277.028229.
- Note (B003, N = 0): hypothesis: the load bound misses the logits because supports overlap. (1) Trimmed supports: an alternative contributes to `S(c)` only enough of its parts to reach `L(c) - w(a)`. (2) Private values for overlapped children. (3) Threshold split per region at the best lambda (docs/core/relaxation_bound.md, "Case splits"). Offline on the N = 0 checkpoint (`test --eval-relaxation-bucket`): load bound 114.97 -> 161.28 (logits region 8.07 -> 54.38); (1) and (2) alone gave nothing on Gemma, the split gives all of it. Run: root bound 160.99 (stored with the float margin), relaxation 4.06 s (was 3.7 s), same incumbent 277.03 at 21.3 s. Final gap 0.4189 (was 0.5858), normalized 0.5220 (was 0.6571). Gate: oracle 42/42 including new case 37 (a mutant that forbids the shared class without charging its plans claims 8.41 > optimum 7 and fails), `test --relaxation-bound` (70552 checks, 7103 probed checks above the plain load bound on the new chunked graphs, none above brute force), search_gap, the four native suites. Verdict: ACCEPTED as frontier change at N_f = 0.

### B004_refine_seq0
- In progress: 2026-10-20T11:35:19.761823-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B004_refine_seq0.json`; log `logs/targets/prefill_decode_seq_len/B004_refine_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B004_refine_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B004_refine_seq0`
- Completed: 2026-10-20T11:37:38.722005-04:00; exit 0; runner elapsed 138.958 s.
  `[SearchEngine volume] iterations=1867 elapsed=107.985164044 total-domain-bits=524409.59663406154 remaining-domain-bits=523725.82502612192 log2-remaining-fraction=-683.77160793961957 eliminated-bits=683.77160793961957 remaining-nodes=1868 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=49.197136418187846 normalized=0.40997613681823203 final-gap=0.27329127169924627 final-incumbent=277.02822900000001 final-bound=201.31883199999999 first-incumbent-elapsed=21.403359413146973 samples=178`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792510550.5248482 program-deadline=1792510646.1214983`
- First incumbent: 24.403349876 s, cost 277.028; best cost 277.028229.
- Note (B004, N = 0): hypothesis: the search below the root never raises the certified bound (it stayed at the root value in every run), so time is better spent nesting the threshold splits. The with-test of a split may use a nested split for the rest (any valid bound of that branch), `depth` levels deep. Root propagation keeps depth 0. After the root heuristic, `SearchEngine::refineRootBound` nests one level deeper at a time (at most half the remaining time; region values proven at a level seed the next level's bisection) and raises the root node's bound from the root's own bucket bounds (snapshotted after root propagation: the state's bucket bounds are stale until the next propagation after the heuristic's restore) and the refined relaxation bounds. Restored and left-child nodes keep at least their parent's bound (their box is a subset), so the certified bound stays monotone. A first version took `state.lower_bound` after the heuristic and read the fully fixed plan's stale engine work (277 = incumbent): caught in a diagnostic run before any evaluation. Run: bound 160.99 -> 183.47 (0.7 s) -> 193.53 (4.1 s) -> 201.32 (22.7 s); depth 4 did not finish. Final gap 0.2733 (was 0.4189), normalized 0.4100 (was 0.5220). Gate: `test --relaxation-bound` (adds 211656 nested split checks at depths 0-2, 21306 above the plain load bound, none above brute force), oracle 42/42, search_gap, the four native suites. Verdict: ACCEPTED as frontier change at N_f = 0.

### B005_fractional_seq0
- In progress: 2026-10-20T12:35:22.707216-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B005_fractional_seq0.json`; log `logs/targets/prefill_decode_seq_len/B005_fractional_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B005_fractional_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B005_fractional_seq0`
- Completed: 2026-10-20T12:37:41.882682-04:00; exit 0; runner elapsed 139.172 s.
  `[SearchEngine volume] iterations=5216 elapsed=107.958094563 total-domain-bits=524409.59663406154 remaining-domain-bits=523644.9194955909 log2-remaining-fraction=-764.67713847063715 eliminated-bits=764.67713847063715 remaining-nodes=5217 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=37.666806158332562 normalized=0.31389005131943803 final-gap=0.16074658586508173 final-incumbent=277.02822900000001 final-bound=232.49688699999999 first-incumbent-elapsed=21.896390438079834 samples=336`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792514154.0787075 program-deadline=1792514249.182327`
- First incumbent: 24.896380424 s, cost 277.028; best cost 277.028229.
- Note (B005, N = 0): two changes. (1) Fractional supports (claims) in the load recursion (docs/core/relaxation_bound.md, "Fractional supports"): a child is packed at the largest alpha that keeps every class's total claim at most 1, so the chunk dots that share a transposed slab or the GPU copy of the transposed embedding (each needs only a little of it) all count. Private values (B003) were dropped (they never applied on Gemma). Offline plain load bound 114.97 -> 232.83; splits now add almost nothing (refinement 232.41 -> 232.50 in 1.8 s). (2) `SelectionReachability::update` (cyclic buckets) jumps a node that lost its support straight to the smallest parent level + 1 above it, instead of raising the level one step at a time with a rescan per step (O(nodes x incoming) per class made unreachable; the heuristic's validation unselects thousands): tryPlan propagation 4.8 -> 4.5 s at N = 0, more at larger N. A first version followed a self-edge's moving level and crashed `test --selection-reachability`; self-edges never support a level and are skipped, and a missing support now throws. Run: root bound 232.41, incumbent 277.03 at 21.9 s, final gap 0.1607 (was 0.2733), normalized 0.3139 (was 0.4100). An attempted exact DP over region selections (states = required classes) exploded on the logits region and was removed. Gate: `test --relaxation-bound` (all checks now on the claims recursion, none above brute force), oracle 43/43 with new case 38 (a mutant crediting a used-up child's own weight claims 8 > 5 and fails), search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 0.

### B006_global_split_seq0
- In progress: 2026-10-20T13:19:34.465090-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B006_global_split_seq0.json`; log `logs/targets/prefill_decode_seq_len/B006_global_split_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B006_global_split_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B006_global_split_seq0`
- Completed: 2026-10-20T13:21:53.719204-04:00; exit 0; runner elapsed 139.252 s.
  `[SearchEngine volume] iterations=4879 elapsed=107.931355661 total-domain-bits=524409.59663406154 remaining-domain-bits=523645.7526008545 log2-remaining-fraction=-763.84403320704587 eliminated-bits=763.84403320704587 remaining-nodes=4880 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=27.059502953288888 normalized=0.22549585794407406 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=22.362076759338379 samples=332`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792516806.2729397 program-deadline=1792516900.9108732`
- First incumbent: 25.362066507 s, cost 277.028; best cost 277.028229.
- Note (B006, N = 0): hypothesis: per-region comparison of a bound-guided plan showed the gap concentrated in region 2 (8 classes, worth 0.0001, plan 39.8): the region's target group (the fp32 embedding) can be produced by the nearly free GPU cast, so later regions treat the 40 ms CPU cast as paid although every CPU logits chunk reads it. Global split on the class (docs/core/relaxation_bound.md, "Global split"): selected -> its region is worth max(value, L) = +39.8; avoided -> forbid it in every region with infeasibility carried through the children below each region. It only bit once the masked recursion decided positivity dynamically (the per-chunk slice classes had a free view of the cast and were structurally fixed at L = 0). Offline 232.83 -> 264.09 (avoided 264.09, used 272.60). An extraction following the relaxation's argmin alternatives was tried and dropped (plans stayed at 279). Run: root bound 263.62 (float margin), refined 263.70, incumbent 277.03 at 22.4 s. Final gap 0.0481 < 0.05: gap-target-met=true, logits ok. Normalized 0.2255 (was 0.3139). Gate: `test --relaxation-bound` (70552 global split checks, 21970 above the plain bound, none above brute force), oracle 44/44 with new case 39 (a mutant adding L(x) to the region claims 8 > 5 and fails), search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as scaling change: N* = 0 on this box. N_f = 1 next.

### B007_accepted_seq1
- In progress: 2026-10-20T13:22:23.159197-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B007_accepted_seq1.json`; log `logs/targets/prefill_decode_seq_len/B007_accepted_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B007_accepted_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B007_accepted_seq1`
- Completed: 2026-10-20T13:23:54.641630-04:00; exit 1; runner elapsed 91.481 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### B008_n1_fixes_seq1
- In progress: 2026-10-20T14:42:51.609267-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B008_n1_fixes_seq1.json`; log `logs/targets/prefill_decode_seq_len/B008_n1_fixes_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B008_n1_fixes_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B008_n1_fixes_seq1`
- Completed: 2026-10-20T14:45:15.403221-04:00; exit 0; runner elapsed 143.792 s.
  `[SearchEngine volume] iterations=1150 elapsed=107.371291433 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603854.1775470865 log2-remaining-fraction=-633.76196389761753 eliminated-bits=633.76196389761753 remaining-nodes=961 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=50.493532862044461 normalized=0.4207794405170372 final-gap=0.14262835483271111 final-incumbent=14.869035 final-bound=12.748289 first-incumbent-elapsed=38.930760622024536 samples=215`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792521820.5393274 program-deadline=1792521898.6085794`
- First incumbent: 41.930747986 s, cost 14.869; best cost 14.869035.

### B009_n1_fixes_seq0
- In progress: 2026-10-20T14:45:15.461670-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B009_n1_fixes_seq0.json`; log `logs/targets/prefill_decode_seq_len/B009_n1_fixes_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B009_n1_fixes_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B009_n1_fixes_seq0`
- Completed: 2026-10-20T14:47:34.452785-04:00; exit 0; runner elapsed 138.990 s.
  `[SearchEngine volume] iterations=4937 elapsed=108.10258998099999 total-domain-bits=524409.59663406154 remaining-domain-bits=523645.56787705369 log2-remaining-fraction=-764.0287570078508 eliminated-bits=764.0287570078508 remaining-nodes=4938 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=26.709835854957738 normalized=0.22258196545798115 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=21.994683265686035 samples=331`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792521946.7791536 program-deadline=1792522041.7844806`
- First incumbent: 24.994673014 s, cost 277.028; best cost 277.028229.
- Note (B007, N = 1, accepted B006): failed. Root propagation 70.5 s: the interval bound kept doubling its stride although its value plateaued at stride 64 (stride 1024 alone took 17.5 s). Then the root heuristic threw "no usable alternative for an active class". The original code (f5e0560) fails the same way at N = 1 on this box (diagnostic run), with a 21 s relaxation. Verdict: reference for N_f = 1 (failed run).
- Note (B008, N = 1): (1) interval strides stop at the first stride that does not improve (relaxation 70 s -> 7.5 s, same bound 12.748). (2) Extraction only takes alternatives whose children can still be produced (cache-only classes of disallowed bases). (3) The covering of the zero-weight full bucket was greedy over an over-approximate reach (anything reachable through any alternative): it selected 19 of the 246 bases the decode plan cached, and the consistency rounds then dropped the expensive ones (GPU copies of the transposed embedding halves, 38 ms each per decode). It now keeps, bottom-up, the alternative whose children's kept selections cover the most target value (exact union), and targets are worth their size in bytes (estimates priced many at 0). Round 0 now has no violations. Incumbent 300.8 -> 14.87. (4) The heuristic validates its best local-search plan before cache-set moves (now capped at 10 s, validated again only if they predict >= 0.1% better): first incumbent at 38.9 s instead of ~64 s. Run: bound 12.748 (interval), incumbent 14.869, final gap 0.1426, normalized 0.4208, logits ok on 3 graphs. Gate: oracle 44/44, `test --relaxation-bound`, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B009, N = 0, B008 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.2226 (B006 0.2255).

### B010_fast_validate_seq1
- In progress: 2026-10-20T15:06:01.334083-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B010_fast_validate_seq1.json`; log `logs/targets/prefill_decode_seq_len/B010_fast_validate_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B010_fast_validate_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B010_fast_validate_seq1`
- Completed: 2026-10-20T15:08:24.524677-04:00; exit 0; runner elapsed 143.189 s.
  `[SearchEngine volume] iterations=1461 elapsed=107.51868432099999 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603826.5576540395 log2-remaining-fraction=-661.38185694464482 eliminated-bits=661.38185694464482 remaining-nodes=1264 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=39.326740932072738 normalized=0.32772284110060618 final-gap=0.14262835483271111 final-incumbent=14.869035 final-bound=12.748289 first-incumbent-elapsed=25.906313180923462 samples=263`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792523197.1521342 program-deadline=1792523288.2458322`
- First incumbent: 28.906301975 s, cost 14.869; best cost 14.869035.

### B011_fast_validate_seq0
- In progress: 2026-10-20T15:08:24.584663-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B011_fast_validate_seq0.json`; log `logs/targets/prefill_decode_seq_len/B011_fast_validate_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B011_fast_validate_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B011_fast_validate_seq0`
- Completed: 2026-10-20T15:10:43.587920-04:00; exit 0; runner elapsed 139.002 s.
  `[SearchEngine volume] iterations=5550 elapsed=108.136365341 total-domain-bits=524409.59663406154 remaining-domain-bits=523644.24875926116 log2-remaining-fraction=-765.34787480038358 eliminated-bits=765.34787480038358 remaining-nodes=5551 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=22.486072587527229 normalized=0.18738393822939356 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=17.55748462677002 samples=350`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792523331.4157314 program-deadline=1792523430.8582566`
- First incumbent: 20.557474852 s, cost 277.028; best cost 277.028229.
- Note (B010, N = 1): validating a complete plan (tryPlan) took 15.2 s at N = 1, 13 s of it in three propagators (selection reachability, cycle avoidance, "Pearce-Kelly"), each called once per fixed selection (137k calls) with ancestor/descendant searches that are quadratic over a plan. While the root heuristic validates a complete plan (`state.completing_plan`, every selection fixed before propagation) nothing is left to prune: each of them now checks a bucket once per selection revision when all its selections are fixed (a DFS cycle check, or a search from the root over the selected alternatives: a selected class it misses is a conflict) and falls back to normal propagation otherwise. The incremental reachability structure is untouched, so it stays at the state the validation restores. tryPlan 15.2 s -> 2.6 s (propagation 13.0 -> 0.45 s); first incumbent 38.9 -> 25.9 s; same plan and bound. Final gap 0.1426, normalized 0.3277 (was 0.4208). Gate: oracle 44/44 (its native solves validate heuristic plans through these shortcuts), `test --relaxation-bound`, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B011, N = 0, B010 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1874 (first incumbent 17.6 s, was 22.0).

### B012_fast_saturation_seq1
- In progress: 2026-10-20T15:36:31.700003-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B012_fast_saturation_seq1.json`; log `logs/targets/prefill_decode_seq_len/B012_fast_saturation_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B012_fast_saturation_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B012_fast_saturation_seq1`
- Completed: 2026-10-20T15:38:48.097849-04:00; exit 0; runner elapsed 136.396 s.
  `[SearchEngine volume] iterations=1618 elapsed=114.321621495 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603810.9773087495 log2-remaining-fraction=-676.96220223465934 eliminated-bits=676.96220223465934 remaining-nodes=1387 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=33.599868477659022 normalized=0.27999890398049188 final-gap=0.14262835483271111 final-incumbent=14.869035 final-bound=12.748289 first-incumbent-elapsed=19.226744890213013 samples=288`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792525013.8896353 program-deadline=1792525111.6629031`
- First incumbent: 22.226732254 s, cost 14.869; best cost 14.869035.

### B013_fast_saturation_seq0
- In progress: 2026-10-20T15:38:48.156474-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B013_fast_saturation_seq0.json`; log `logs/targets/prefill_decode_seq_len/B013_fast_saturation_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B013_fast_saturation_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B013_fast_saturation_seq0`
- Completed: 2026-10-20T15:41:00.139944-04:00; exit 0; runner elapsed 131.982 s.
  `[SearchEngine volume] iterations=7118 elapsed=115.01796397 total-domain-bits=524409.59663406154 remaining-domain-bits=523641.93441538431 log2-remaining-fraction=-767.66221867722925 eliminated-bits=767.66221867722925 remaining-nodes=7119 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=15.831514215093748 normalized=0.13192928512578123 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=10.566634654998779 samples=377`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792525141.026407 program-deadline=1792525247.4597819`
- First incumbent: 13.566625118 s, cost 277.028; best cost 277.028229.
- Note (B012, N = 1): the full-bucket saturation took 8.4 s at every N; `perf` put 67% of it in `ConsumerWeightReuseRule` (`getStorageWeightName` called for every COPY_TO enode on every weight lookup: `earlierWeightClass` rescanned all copies). The rule now indexes COPY_TO enodes out of STORAGE by the INPUT names of their child class (storage copies without a name are scanned too; every candidate is verified as before; a copy whose child is not in STORAGE can never name a weight, as merged classes share their memory space). Saturation 8.4 s -> 1.4 s with identical e-graphs (class/enode counts and cycle passes at N = 0 and 1; `tests/verify_saturation.py` EQUIVALENT at seq_len 2 and 8 against gold generated from the original rule, run with TG_DISABLE_OPENCL=1 on this box). First incumbent 25.9 -> 19.2 s; normalized 0.2800 (was 0.3277), final gap 0.1426. Gate: oracle 44/44, `test --relaxation-bound`, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B013, N = 0, B012 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1319 (first incumbent 10.6 s).

### B014_deferred_intervals_seq1
- In progress: 2026-10-20T15:48:55.015422-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B014_deferred_intervals_seq1.json`; log `logs/targets/prefill_decode_seq_len/B014_deferred_intervals_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B014_deferred_intervals_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B014_deferred_intervals_seq1`
- Completed: 2026-10-20T15:51:11.215064-04:00; exit 0; runner elapsed 136.198 s.
  `[SearchEngine volume] iterations=1631 elapsed=114.414638899 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603810.2552213117 log2-remaining-fraction=-677.68428967241198 eliminated-bits=677.68428967241198 remaining-nodes=1400 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=32.889152451116026 normalized=0.27407627042596688 final-gap=0.14262835483271111 final-incumbent=14.869035 final-bound=12.748289 first-incumbent-elapsed=12.526405334472656 samples=289`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792525750.4292989 program-deadline=1792525854.9029059`
- First incumbent: 15.526392937 s, cost 14.869; best cost 14.869035.

### B015_deferred_intervals_seq0
- In progress: 2026-10-20T15:51:11.282824-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B015_deferred_intervals_seq0.json`; log `logs/targets/prefill_decode_seq_len/B015_deferred_intervals_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B015_deferred_intervals_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B015_deferred_intervals_seq0`
- Completed: 2026-10-20T15:53:23.328626-04:00; exit 0; runner elapsed 132.044 s.
  `[SearchEngine volume] iterations=7186 elapsed=115.100512659 total-domain-bits=524409.59663406154 remaining-domain-bits=523641.84239272837 log2-remaining-fraction=-767.75424133316847 eliminated-bits=767.75424133316847 remaining-nodes=7187 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.434118687084567 normalized=0.10361765572570472 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=6.9964158535003662 samples=379`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792525880.6006398 program-deadline=1792525990.6042342`
- First incumbent: 9.996405602 s, cost 277.028; best cost 277.028229.
- Note (B014, N = 1): the interval bound (6.4 s at N = 1, 3.3 s at N = 0) moved out of root propagation into `RelaxationBoundPropagator::computeIntervals`, which `refineRootBound` runs first (level 0) after the root heuristic; nested splits follow. Root propagation 7.9 -> 1.5 s; first incumbent 19.2 -> 12.5 s (the gap runs on the load bound, 8.22, for the 6.7 s until the interval bound lands). Same final bound and incumbent. Normalized 0.2741 (was 0.2800), final gap 0.1426. Gate: oracle 44/44, `test --relaxation-bound`, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B015, N = 0, B014 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1036 (first incumbent 7.0 s).
- Diagnostic (not an evaluation, B014 code, N = 2, 5 buckets): bound 24.50 (interval), incumbent 27.07 at ~43 s, final gap 0.095. Saturation 12.9 s, cycle reduction etc. 4.2 s, root propagation 2.1 s, heuristic 23.8 s, intervals 12.1 s.

### B016_flexible_windows_seq1
- In progress: 2026-10-20T17:23:57.857804-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B016_flexible_windows_seq1.json`; log `logs/targets/prefill_decode_seq_len/B016_flexible_windows_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B016_flexible_windows_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B016_flexible_windows_seq1`
- Completed: 2026-10-20T17:26:14.141018-04:00; exit 0; runner elapsed 136.281 s.
  `[SearchEngine volume] iterations=1474 elapsed=114.36944172299999 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603825.5237589271 log2-remaining-fraction=-662.41575205698609 eliminated-bits=662.41575205698609 remaining-nodes=1265 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=32.340194733069914 normalized=0.26950162277558259 final-gap=0.13470490855660772 final-incumbent=14.869035 final-bound=12.866103000000001 first-incumbent-elapsed=12.651290893554688 samples=267`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792531453.4951844 program-deadline=1792531557.8439071`
- First incumbent: 15.651277304 s, cost 14.869; best cost 14.869035.

### B017_flexible_windows_seq0
- In progress: 2026-10-20T17:26:14.209272-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B017_flexible_windows_seq0.json`; log `logs/targets/prefill_decode_seq_len/B017_flexible_windows_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B017_flexible_windows_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B017_flexible_windows_seq0`
- Completed: 2026-10-20T17:28:26.245231-04:00; exit 0; runner elapsed 132.034 s.
  `[SearchEngine volume] iterations=6022 elapsed=115.019320412 total-domain-bits=524409.59663406154 remaining-domain-bits=523643.43821299524 log2-remaining-fraction=-766.15842106629862 eliminated-bits=766.15842106629862 remaining-nodes=6023 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.490320451579159 normalized=0.10408600376315966 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.0534324645996094 samples=354`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792531583.6442947 program-deadline=1792531693.5908728`
- First incumbent: 10.053421974 s, cost 277.028; best cost 277.028229.
- Note (B016, N = 1): diagnosis: comparing the plan's time between consecutive chain links with the stride-1 window values showed per-layer windows (gate projection -> GELU) where the plan spends 0.061 ms and the bound 0.009: the GPU runs the up projection there, which belongs to no stride-1 window, and uniform strides rarely align with the layer blocks. `flexibleIntervalBound` values every window of <= 8 links (exact enumeration first, otherwise lambda / mu values and the cut DP; windows above 512 classes count 0) and keeps the best partition by DP; it runs in the refinement after the interval bound (6 s at N = 1; length 16 gives the same in 61 s). Bound 12.748 -> 12.866, final gap 0.1347 (was 0.1426), normalized 0.2695 (was 0.2741). Gate: `test --relaxation-bound` (adds 24000 flexible window checks at lengths 1, 3, 8 against the minimum makespan), oracle 44/44, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B017, N = 0, B016 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1041.
- Experiments after B016 (diagnostics, offline on the N = 1 decode checkpoint or compile-only runs; none evaluated, all reverted; patches in the agent scratchpad only):
  - One-machine (Jackson) stage bound per enumerated window selection (min head + engine load + min tail): sound (`test --relaxation-bound`), but the flexible bound stayed 12.892: each window still picks its own cheapest selection.
  - Counting in-group COPY_TO members of the opening link inside the next window (such a copy starts after the earliest member's finish and is never that member): sound, raised random-graph interval values slightly, no change on Gemma (12.892).
  - Local search allowed to use every cache-possible base (ignoring cross-bucket consistency): still 14.869. Randomized list-scheduling priorities (200 seeds): still 14.869, so the schedule is tight for the selection.
  - Per-window comparison on the flexible partition: remaining slack 1.98 ms = logits window 0.36 + per-layer attention windows ~0.05 each. In those the plan's CPU lane runs ~0.09 ms of small ops and activation copies; e.g. a `MUL` at 0.0243 has a fused 0.0086 alternative, but it needs another input form and the switch is slightly worse (14.871). The window bound's cheaper selection needs coordinated changes.
  - Skipping local search for starts > 25% worse than the best initial plan: local search time unchanged (3.94 s; the good starts dominate).

### B018_window_moves_seq1
- In progress: 2026-10-20T18:28:29.951800-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B018_window_moves_seq1.json`; log `logs/targets/prefill_decode_seq_len/B018_window_moves_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B018_window_moves_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B018_window_moves_seq1`
- Completed: 2026-10-20T18:30:46.270295-04:00; exit 0; runner elapsed 136.315 s.
  `[SearchEngine volume] iterations=1345 elapsed=114.464572307 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603845.9965012816 log2-remaining-fraction=-641.94300970248878 eliminated-bits=641.94300970248878 remaining-nodes=1156 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=27.09561554598946 normalized=0.22579679621657883 final-gap=0.11295502746949317 final-incumbent=14.504454000000001 final-bound=12.866103000000001 first-incumbent-elapsed=12.460314750671387 samples=248`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792535325.3881154 program-deadline=1792535429.9278135`
- First incumbent: 15.460301876 s, cost 14.869; best cost 14.504454.

### B019_window_moves_seq0
- In progress: 2026-10-20T18:30:46.336737-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B019_window_moves_seq0.json`; log `logs/targets/prefill_decode_seq_len/B019_window_moves_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B019_window_moves_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B019_window_moves_seq0`
- Completed: 2026-10-20T18:32:58.289370-04:00; exit 0; runner elapsed 131.951 s.
  `[SearchEngine volume] iterations=5806 elapsed=115.054242335 total-domain-bits=524409.59663406154 remaining-domain-bits=523643.78353531484 log2-remaining-fraction=-765.81309874670114 eliminated-bits=765.81309874670114 remaining-nodes=5807 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.640703320029955 normalized=0.10533919433358295 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.2113521099090576 samples=352`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792535455.873842 program-deadline=1792535565.6624994`
- First incumbent: 10.211342573 s, cost 277.028; best cost 277.028229.
- Note (B018, N = 1): window moves (`improveByWindows`): for each window of the bound's flexible partition, the selection attaining the window's exact value (`exactWindow` now returns it; `windowSelection`) replaces the plan's choices there; children the result needs are activated greedily, unreached classes drop; the move is kept if the list-scheduled makespan decreases, every cached base stays selected and only cached bases' cache-only alternatives are used. One pass (a second finds nothing), then local search. Phases reordered: first incumbent, then the interval and flexible-window bounds (the window moves reuse that partition through `RelaxationBoundPropagator::computerFor`, checked to be the same relaxation; selections computed in parallel, shared by identical buckets), then window moves (2.6 s) and cache-set moves, then deeper split refinement. Decode plan 14.869 -> 14.5045 (validated). Final gap 0.1130 (was 0.1347), normalized 0.2258 (was 0.2695). Gate: oracle 44/44, `test --relaxation-bound`, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B019, N = 0, B018 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1053.

### B020_sliding_windows_seq1
- In progress: 2026-10-20T18:54:32.216383-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B020_sliding_windows_seq1.json`; log `logs/targets/prefill_decode_seq_len/B020_sliding_windows_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B020_sliding_windows_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B020_sliding_windows_seq1`
- Completed: 2026-10-20T18:56:48.603996-04:00; exit 0; runner elapsed 136.385 s.
  `[SearchEngine volume] iterations=1228 elapsed=114.450758631 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603851.7920347024 log2-remaining-fraction=-636.14747628173791 eliminated-bits=636.14747628173791 remaining-nodes=1039 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=27.186383374157543 normalized=0.22655319478464619 final-gap=0.11225716984938712 final-incumbent=14.493052 final-bound=12.866103000000001 first-incumbent-elapsed=12.451607704162598 samples=235`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792536887.5924726 program-deadline=1792536992.1408775`
- First incumbent: 15.451595068 s, cost 14.869; best cost 14.493052.

### B021_sliding_windows_seq0
- In progress: 2026-10-20T18:56:48.661795-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B021_sliding_windows_seq0.json`; log `logs/targets/prefill_decode_seq_len/B021_sliding_windows_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B021_sliding_windows_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B021_sliding_windows_seq0`
- Completed: 2026-10-20T18:59:00.693561-04:00; exit 0; runner elapsed 132.030 s.
  `[SearchEngine volume] iterations=5796 elapsed=115.044346921 total-domain-bits=524409.59663406154 remaining-domain-bits=523643.79301312688 log2-remaining-fraction=-765.80362093466101 eliminated-bits=765.80362093466101 remaining-nodes=5797 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.507417482938783 normalized=0.10422847902448985 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.0710434913635254 samples=351`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792537018.0357754 program-deadline=1792537127.9647434`
- First incumbent: 10.071032047 s, cost 277.028; best cost 277.028229.
- Note (B020, N = 1): sliding windows of 4 links (step 2) and 8 links (step 4) added to the window moves: decode plan 14.5045 -> 14.4931, final gap 0.1123, but the moves take 5.7 s instead of 2.6 s and the integral is 0.2266 (B018 0.2258). Verdict: REJECTED (frontier metric is the integral), reverted. Also tried (diagnostic, reverted): keeping the 8 best selections per window and letting the list schedule choose (14.5103, 60 s); ranking window selections by max(chain, loads, one-machine stage bound) (no change).

### B022_fast_flexible_seq1
- In progress: 2026-10-20T19:26:36.450354-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B022_fast_flexible_seq1.json`; log `logs/targets/prefill_decode_seq_len/B022_fast_flexible_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B022_fast_flexible_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B022_fast_flexible_seq1`
- Completed: 2026-10-20T19:28:52.852327-04:00; exit 0; runner elapsed 136.400 s.
  `[SearchEngine volume] iterations=1460 elapsed=114.449906175 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603826.665688829 log2-remaining-fraction=-661.2738221550826 eliminated-bits=661.2738221550826 remaining-nodes=1265 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=25.140844833711771 normalized=0.20950704028093142 final-gap=0.11295502746949317 final-incumbent=14.504454000000001 final-bound=12.866103000000001 first-incumbent-elapsed=12.38314962387085 samples=269`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792538811.8005748 program-deadline=1792538916.4174385`
- First incumbent: 15.383136272 s, cost 14.869; best cost 14.504454.

### B023_fast_flexible_seq0
- In progress: 2026-10-20T19:28:52.921201-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B023_fast_flexible_seq0.json`; log `logs/targets/prefill_decode_seq_len/B023_fast_flexible_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B023_fast_flexible_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B023_fast_flexible_seq0`
- Completed: 2026-10-20T19:31:04.930848-04:00; exit 0; runner elapsed 132.008 s.
  `[SearchEngine volume] iterations=7174 elapsed=115.06049282799999 total-domain-bits=524409.59663406154 remaining-domain-bits=523641.86429877859 log2-remaining-fraction=-767.73233528295532 eliminated-bits=767.73233528295532 remaining-nodes=7175 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.422495102688069 normalized=0.10352079252240058 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=6.9838271141052246 samples=376`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792538942.2135916 program-deadline=1792539052.2297728`
- First incumbent: 9.983818769 s, cost 277.028; best cost 277.028229.
- Note (B022, N = 1): 1581 of the flexible windows (about 64 classes each) do not finish their exact enumeration and need the lambda / mu values and cut DP, which wrote the computer's shared weight table, so they ran one at a time (5.2 s). `weight()` / `fillRegionWeights` now use a per-thread table when one is set (`thread_weights_`), and both the exact values and these run in parallel: 1.4 s, same value 12.892. Refinement order: flexible windows right after the first validated plan, then window moves (they reuse that partition), then the stride windows (6.1 s; not binding at N = 0 or 1), then nested splits. Bound 12.866 now lands ~6 s after the first incumbent (was ~12 s), and the 14.5045 plan ~10 s earlier. Normalized 0.2095 (was 0.2258), final gap unchanged 0.1130. Gate: `test --relaxation-bound`, oracle 44/44, search_gap, the four native suites and `--selection-reachability`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B023, N = 0, B022 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1035.

### B024_global_window_moves_seq1
- In progress: 2026-10-20T21:59:48.815148-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B024_global_window_moves_seq1.json`; log `logs/targets/prefill_decode_seq_len/B024_global_window_moves_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B024_global_window_moves_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B024_global_window_moves_seq1`
- Completed: 2026-10-20T22:02:05.134040-04:00; exit 0; runner elapsed 136.315 s.
  `[SearchEngine volume] iterations=1498 elapsed=114.37090925699999 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603824.7110246252 log2-remaining-fraction=-663.22848635888658 eliminated-bits=663.22848635888658 remaining-nodes=1289 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.172976061690271 normalized=0.20144146718075226 final-gap=0.10998765021575979 final-incumbent=14.456123 final-bound=12.866128 first-incumbent-elapsed=11.875320911407471 samples=272`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792548003.6822536 program-deadline=1792548108.8069465`
- First incumbent: 14.875307083 s, cost 14.869; best cost 14.456123.

### B025_global_window_moves_seq0
- In progress: 2026-10-20T22:02:05.199244-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B025_global_window_moves_seq0.json`; log `logs/targets/prefill_decode_seq_len/B025_global_window_moves_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B025_global_window_moves_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B025_global_window_moves_seq0`
- Completed: 2026-10-20T22:04:17.352671-04:00; exit 0; runner elapsed 132.152 s.
  `[SearchEngine volume] iterations=6860 elapsed=115.044136413 total-domain-bits=524409.59663406154 remaining-domain-bits=523642.19377051655 log2-remaining-fraction=-767.40286354499403 eliminated-bits=767.40286354499403 remaining-nodes=6861 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.597766501793094 normalized=0.10498138751494245 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.167884349822998 samples=373`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792548134.7096739 program-deadline=1792548244.5417993`
- First incumbent: 10.167874575 s, cost 277.028; best cost 277.028229.
- Note (B024, N = 1): where the decode plan loses time against the bound (plan 14.51 vs per-window bound sum 12.89, per window of the flexible partition): 16 per-layer windows each ~0.05 over (the GPU waits for a CPU chain, see below), the final window 0.36 over, the rest ~0.4. The final window is the LM head, which the e-graph can split over the vocabulary into 2, 4, 8 or 16 slices (GPU 2.136 / 1.088 / 0.551 / 0.285 ms; CPU 1/16 slice 3.11 ms) with copies and concats. The bound's value there, 4.0591, is exactly the GPU load of 1/2 + 1/4 + 1/8 + 1/16 with the last 1/16 on the CPU in parallel; the plan computed both halves on the GPU (4.42). Its exact enumeration (110 classes) never finished, so window moves had no selection there, and the move also needs a different cache set (the plan caches the GPU weight halves; the split needs the slices cached). Changes: (1) `exactWindow` prunes by a chain bound, a per-engine bound over the needed unchosen classes and a per-engine heaviest-route load bound, and drops classes that add nothing plus duplicate alternatives (exactness checked: `test --relaxation-bound` now compares flexible values with and without it, `setExactWindowPruning(false)`; mutants pruning at 0.8 x the best value are caught). Same bound values on the N = 1 checkpoint, flexible windows 0.65 s instead of 1.4 s. (2) `windowSelections` returns the best selection a 200000-node search finds even when it does not finish (final window: 4.11 / 4.20 vs the plan's 4.27 GPU load). (3) Window moves apply each selection to all weighted buckets with the relaxation at once and adjust the cache set (`spliceWindowSelection`: closure may read any cacheable base from the cache, bases nothing reads from the cache leave it, new ones are activated in every bucket, memory caps and consistency checked, kept if the weighted makespan drops). Before (3) the final-window move was rejected for the orphaned cached half. (4) Root local search runs one round of starts (the best `threads / weighted buckets` plans; 8 of 14 at N = 1), saving ~0.7 s before the first incumbent. Decode plan 14.5045 -> 14.4561 (validated). Normalized 0.2014 (was 0.2095), final gap 0.1100 (was 0.1130). Gate: oracle 44/44, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Verdict: ACCEPTED as frontier change at N_f = 1.
- Note (B025, N = 0, B024 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1050.
- Rejected (diagnostics at N = 1, not recorded as runs): Giffler-Thompson style lookahead in the list schedule (an operation waiting on a higher-priority one becoming ready before the chosen one would finish): 14.569 vs 14.514 (the duration-only tail ranks the V projection above the K RoPE that gates the CPU attention chain). Forward-backward improvement of the list schedule (4 rounds): first plan 14.883 vs 14.869, final 14.520, local search 3x slower. Critical-path order moves on the final plan (an engine-blocked operation moved before its engine predecessor or the predecessor after it, first improvement): 176 valid moves, none lowers the makespan.

### B026_cache_aware_windows_stage_seq1
- In progress: 2026-10-20T22:50:22.042817-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B026_cache_aware_windows_stage_seq1.json`; log `logs/targets/prefill_decode_seq_len/B026_cache_aware_windows_stage_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B026_cache_aware_windows_stage_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B026_cache_aware_windows_stage_seq1`
- Completed: 2026-10-20T22:52:38.347384-04:00; exit 0; runner elapsed 136.303 s.
  `[SearchEngine volume] iterations=1490 elapsed=114.438631946 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603825.1679233073 log2-remaining-fraction=-662.77158767683432 eliminated-bits=662.77158767683432 remaining-nodes=1281 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=21.706124799858738 normalized=0.18088437333215615 final-gap=0.088216753994797351 final-incumbent=14.285392999999999 final-bound=13.025181999999999 first-incumbent-elapsed=11.648543119430542 samples=271`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792551036.6430321 program-deadline=1792551141.9945021`
- First incumbent: 14.648530006 s, cost 14.869; best cost 14.285393.

### B027_cache_aware_windows_stage_seq0
- In progress: 2026-10-20T22:52:38.412256-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B027_cache_aware_windows_stage_seq0.json`; log `logs/targets/prefill_decode_seq_len/B027_cache_aware_windows_stage_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B027_cache_aware_windows_stage_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B027_cache_aware_windows_stage_seq0`
- Completed: 2026-10-20T22:54:50.433116-04:00; exit 0; runner elapsed 132.019 s.
  `[SearchEngine volume] iterations=6836 elapsed=115.04032909999999 total-domain-bits=524409.59663406154 remaining-domain-bits=523642.21141587856 log2-remaining-fraction=-767.38521818298614 eliminated-bits=767.38521818298614 remaining-nodes=6837 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.490327756784055 normalized=0.10408606463986712 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.0550270080566406 samples=374`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792551167.8075671 program-deadline=1792551277.7525532`
- First incumbent: 10.055013895 s, cost 277.028; best cost 277.028229.
- Note (B026, N = 1): in B024 the LM-head window move was evaluated but worse (decode 14.46 -> 17.41): exact-window deduplication had merged the 1/2-slice GPU Dot reading the GPU hidden state with the one reading a CPU->GPU copy of it (that copy adds nothing inside the window), the splice took the copy, and the copy queued behind the 3.11 ms CPU slice. `windowSelections` now returns each chosen alternative with its equivalents and the splice takes the one whose children the bucket already computes. The move then predicted 14.2854 but validation failed (SelectionReachability: the zero-weight full bucket selected a newly cached base unreachably); zero-weight buckets now take a new covering selection (`extractCovering`) when the cache set changes, and weighted buckets outside the group must already select new bases. Validated 14.2854 (LM head: 1/2 + 1/4 + 1/8 + 1/16 slices on the GPU, one 1/16 slice on the CPU in parallel, 7 more weight slices cached). Bound: exact windows also count, per engine, shortest head + load + shortest tail of the selection's operations on it (one-machine value; docs/core/relaxation_bound.md); decode flexible bound 12.892 -> 13.051, root 12.866 -> 13.025. New oracle case 40_one_machine_window_tail (a longest-tail mutant claims a root bound of 9 against the optimum 7 and fails; the random `--relaxation-bound` graphs also catch it). Gate: oracle 45/45, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Normalized 0.1809 (was 0.2014), final gap 0.0882 (was 0.1100). Verdict: ACCEPTED as frontier change at N_f = 1. Also tried: a second window pass after local search (14.2854 -> 14.2818 for 2.6 s more before validation; not kept).
- Note (B027, N = 0, B026 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1041; the N = 0 root bound is unchanged (263.70).

### B028_long_windows_seq1
- In progress: 2026-10-21T00:02:45.197643-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B028_long_windows_seq1.json`; log `logs/targets/prefill_decode_seq_len/B028_long_windows_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B028_long_windows_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B028_long_windows_seq1`
- Completed: 2026-10-21T00:05:01.242206-04:00; exit 0; runner elapsed 136.043 s.
  `[SearchEngine volume] iterations=1264 elapsed=114.327579961 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603850.7044155705 log2-remaining-fraction=-637.23509541363455 eliminated-bits=637.23509541363455 remaining-nodes=1075 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=20.776835696680546 normalized=0.17314029747233789 final-gap=0.074333901769450733 final-incumbent=14.285392999999999 final-bound=13.223504 first-incumbent-elapsed=11.902809143066406 samples=242`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792555380.0592184 program-deadline=1792555485.1564219`
- First incumbent: 14.902796507 s, cost 14.869; best cost 14.285393.

### B029_long_windows_seq0
- In progress: 2026-10-21T00:05:01.311576-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B029_long_windows_seq0.json`; log `logs/targets/prefill_decode_seq_len/B029_long_windows_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B029_long_windows_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B029_long_windows_seq0`
- Completed: 2026-10-21T00:07:13.319539-04:00; exit 0; runner elapsed 132.006 s.
  `[SearchEngine volume] iterations=5440 elapsed=115.078098071 total-domain-bits=524409.59663406154 remaining-domain-bits=523644.48631407984 log2-remaining-fraction=-765.11031998170074 eliminated-bits=765.11031998170074 remaining-nodes=5441 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.498533556153497 normalized=0.10415444630127914 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.0608782768249512 samples=342`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792555510.6590836 program-deadline=1792555620.5982153`
- First incumbent: 10.060868263 s, cost 277.028; best cost 277.028229.
- Note (B028, N = 1): per decode layer the plan spans 0.555 ms (GPU busy 0.401, CPU 0.245) against 0.497 of window bounds. The layer is close to a serial chain (GPU projections and MLP; CPU norms, residuals and attention; 12 copies). The CPU placement of the norms is right (GPU kernels ~9 us each: a norm costs ~0.055 on the GPU against ~0.03 on the CPU with both copies), and moving every K RoPE before the V projection on the GPU makes the plan worse (14.4216 vs 14.2854), so the per-layer loss is mostly the bound's. The K branch (projection, norm, RoPE) forks at the layer's normalized input and joins at the attention scores, more than 8 links apart, so no flexible window holds all of it. Windows up to 24 links: 13.25 (+0.20) but 200 s; their exact values alone gain nothing (the gain is the cut DP's); merging consecutive partition windows only (up to 24/32/48 links) 13.0571. Kept: windows of 9-24 links ending at a boundary of the length-8 partition, valued by the cut DP alone, then the partition DP again: 13.25 in 8.2 s on the checkpoint, as refinement level -1 (after the window moves, before the stride windows). Root bound 13.025 -> 13.2235. `test --relaxation-bound` also checks short/long lengths (1, 4) and (2, 8). Gate: oracle 45/45, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Normalized 0.1731 (was 0.1809), final gap 0.0743 (was 0.0882). Verdict: ACCEPTED as frontier change at N_f = 1. Also tried (offline, no change, reverted): one-machine pruning of partial selections in the exact enumeration (the LM-head window still does not finish).
- Note (B029, N = 0, B028 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1042.

### B030_cut_dp_stage_seq1
- In progress: 2026-10-21T00:25:24.823394-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B030_cut_dp_stage_seq1.json`; log `logs/targets/prefill_decode_seq_len/B030_cut_dp_stage_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B030_cut_dp_stage_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B030_cut_dp_stage_seq1`
- Completed: 2026-10-21T00:27:41.165997-04:00; exit 0; runner elapsed 136.341 s.
  `[SearchEngine volume] iterations=1138 elapsed=114.445103085 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603855.1508811885 log2-remaining-fraction=-632.78862979565747 eliminated-bits=632.78862979565747 remaining-nodes=951 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=19.862487969829875 normalized=0.16552073308191562 final-gap=0.066001544374732948 final-incumbent=14.285392999999999 final-bound=13.342535 first-incumbent-elapsed=11.754411935806274 samples=218`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792556739.6022663 program-deadline=1792556844.8478673`
- First incumbent: 14.754399061 s, cost 14.869; best cost 14.285393.

### B031_cut_dp_stage_seq0
- In progress: 2026-10-21T00:27:41.236988-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B031_cut_dp_stage_seq0.json`; log `logs/targets/prefill_decode_seq_len/B031_cut_dp_stage_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B031_cut_dp_stage_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B031_cut_dp_stage_seq0`
- Completed: 2026-10-21T00:29:53.237719-04:00; exit 0; runner elapsed 131.999 s.
  `[SearchEngine volume] iterations=3904 elapsed=115.047270531 total-domain-bits=524409.59663406154 remaining-domain-bits=523650.51307834015 log2-remaining-fraction=-759.0835557213868 eliminated-bits=759.0835557213868 remaining-nodes=3905 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.579434761871848 normalized=0.10482862301559874 final-gap=0.048103693432628436 final-incumbent=277.02822900000001 final-bound=263.70214800000002 first-incumbent-elapsed=7.143761157989502 samples=317`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792556870.7182269 program-deadline=1792556980.574476`
- First incumbent: 10.143750906 s, cost 277.028; best cost 277.028229.
- Note (B030, N = 1): the cut DP (values of windows whose exact enumeration does not finish, and of the long windows) now carries the one-machine value: per engine the shortest head and shortest tail of the operations placed on it (infinite before the first, so componentwise dominance and closest-pair merging stay valid), value head + load + tail on engines with work. Checkpoint: flexible 13.05 -> 13.17 (LM-head window), with long windows 13.25 -> 13.37 (long phase 8 s -> 14 s). Root bound 13.2235 -> 13.3425. A longest-tail mutant fails `test --relaxation-bound` (its cut-DP checks run with the exact enumeration disabled; oracle graphs always finish exactly, case 40 covers the rule there). Gate: oracle 45/45, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Normalized 0.1655 (was 0.1731), final gap 0.0660 (was 0.0743). Verdict: ACCEPTED as frontier change at N_f = 1. Also tried (offline): long windows on the boundaries of each new partition again (no change); cut DP with 64 Pareto points for long windows (+0.017 for 3x the time).
- Note (B031, N = 0, B030 code): no regression at N* = 0: final gap 0.0481 (target met), normalized 0.1048.

### B032_long_window_moves_seq1
- In progress: 2026-10-21T01:54:04.160328-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B032_long_window_moves_seq1.json`; log `logs/targets/prefill_decode_seq_len/B032_long_window_moves_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B032_long_window_moves_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B032_long_window_moves_seq1`
- Completed: 2026-10-21T01:56:20.422214-04:00; exit 0; runner elapsed 136.260 s.
  `[SearchEngine volume] iterations=871 elapsed=114.387806416 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603894.6484041384 log2-remaining-fraction=-593.29110684571788 eliminated-bits=593.29110684571788 remaining-nodes=704 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=18.521767745186473 normalized=0.15434806454322061 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.852007389068604 samples=170`
  `[SearchEngine] New best cost: 14.869 at iteration 0 source=root-heuristic event-epoch=1792562059.0592103 program-deadline=1792562164.2072155`
- First incumbent: 14.851994753 s, cost 14.869; best cost 14.021854.

### B033_long_window_moves_seq0
- In progress: 2026-10-21T01:56:20.485575-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B033_long_window_moves_seq0.json`; log `logs/targets/prefill_decode_seq_len/B033_long_window_moves_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B033_long_window_moves_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B033_long_window_moves_seq0`
- Completed: 2026-10-21T01:58:32.678216-04:00; exit 0; runner elapsed 132.191 s.
  `[SearchEngine volume] iterations=683 elapsed=115.050054583 total-domain-bits=524409.59663406154 remaining-domain-bits=523840.74579479586 log2-remaining-fraction=-568.85083926568041 eliminated-bits=568.85083926568041 remaining-nodes=684 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.499710775578258 normalized=0.10416425646315215 final-gap=0.047598495819716619 final-incumbent=277.02822900000001 final-bound=263.84210200000001 first-incumbent-elapsed=7.0994272232055664 samples=85`
  `[SearchEngine] New best cost: 277.028 at iteration 0 source=root-heuristic event-epoch=1792562189.9961245 program-deadline=1792562299.8967075`
- First incumbent: 10.099416971 s, cost 277.028; best cost 277.028.
- Note (B032, N = 1): (1) window moves run a second time on the partition with the long windows (after refinement level -1; `runRootPrimalHeuristic` takes a second callback): in the per-layer attention window (fork at the layer's normalized input, join at attention output) the plan's GPU was busy throughout (Q/K/V projections, Q norm and RoPE, K RoPE) while the cut DP saw a selection with more of it on the CPU; spliced: K RoPE on the CPU, the Q scaling's fast AVX2 kernel with its scale cached. 14.2854 -> 14.0909 (a second pass finds nothing more). (2) The window search is directed by the plan: windows whose plan time already equals their bound are skipped, windows more than 0.06 over get 2000000 nodes (else 200000); each search runs half its budget trying shortest chains first and half least total work first, and each best selection is also polished by a local search on the window value (single-class switches, newly needed classes take their least-work alternative, classes that add nothing inside the window stay unchosen; both versions are tried as splices since the window value can favour selections the full schedule does not). LM head: 7 x 1/16 GPU slices (relaxation value 4.23-4.33 found) -> closer to 1/2 + 1/4 + 1/8 + 1/16 on the GPU. 14.2854 -> 14.2164 (short windows), -> 14.0219 (long windows). (3) Float margin: n = original classes with an alternative of positive duration (a plan selects each once; views add exactly 0) instead of all relaxation classes including unrolled copies: margin 0.2% -> 0.13%, bound 13.3425 -> 13.3523. Gate: oracle 45/45, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Final gap 0.0478 < 0.05: N* = 1. Verdict: ACCEPTED (N* raised to 1). Also tried: reordering every layer's Q RoPE before the V projection on the GPU (14.2132 vs 14.0909) and K RoPE before V (14.4216 vs 14.2854): the list schedule's order is better; 4M-node window searches (do not finish in time).
- Note (B033, N = 0, B032 code): final gap 0.0476 (target met), normalized 0.1042; bound 263.84 (float margin).

### B034_baseline_seq2
- In progress: 2026-10-21T01:58:55.419268-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B034_baseline_seq2.json`; log `logs/targets/prefill_decode_seq_len/B034_baseline_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B034_baseline_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B034_baseline_seq2`
- Completed: 2026-10-21T02:02:47.999156-04:00; exit 1; runner elapsed 232.578 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### B035_cycle_groups_seq1
- In progress: 2026-10-21T03:47:31.210198-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B035_cycle_groups_seq1.json`; log `logs/targets/prefill_decode_seq_len/B035_cycle_groups_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B035_cycle_groups_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B035_cycle_groups_seq1`
- Completed: 2026-10-21T03:50:23.190238-04:00; exit 0; runner elapsed 171.976 s.
  `[SearchEngine volume] iterations=948 elapsed=114.375489718 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603876.321405163 log2-remaining-fraction=-611.61810582107864 eliminated-bits=611.61810582107864 remaining-nodes=781 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.268704139414712 normalized=0.14390586782845594 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=10.955468893051147 samples=179`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792568865.7429383 program-deadline=1792568971.7874827`
- First incumbent: 13.955455542 s, cost 14.6941; best cost 14.021854.

### B036_cycle_groups_seq0
- In progress: 2026-10-21T03:50:23.258537-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B036_cycle_groups_seq0.json`; log `logs/targets/prefill_decode_seq_len/B036_cycle_groups_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B036_cycle_groups_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B036_cycle_groups_seq0`
- Completed: 2026-10-21T03:52:35.419636-04:00; exit 0; runner elapsed 132.158 s.
  `[SearchEngine volume] iterations=652 elapsed=115.012825897 total-domain-bits=524409.59663406154 remaining-domain-bits=523845.48263095669 log2-remaining-fraction=-564.11400310485624 eliminated-bits=564.11400310485624 remaining-nodes=653 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.340337177007717 normalized=0.10283614314173098 final-gap=0.048315441616851904 final-incumbent=276.99084499999998 final-bound=263.60791 first-incumbent-elapsed=6.8746352195739746 samples=80`
  `[SearchEngine] New best cost: 276.991 at iteration 0 source=root-heuristic event-epoch=1792569032.5255444 program-deadline=1792569142.6509197`
- First incumbent: 9.874624729 s, cost 276.991; best cost 276.991.

### B037_cycle_groups_seq2
- In progress: 2026-10-21T03:52:35.483308-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B037_cycle_groups_seq2.json`; log `logs/targets/prefill_decode_seq_len/B037_cycle_groups_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B037_cycle_groups_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B037_cycle_groups_seq2`
- Completed: 2026-10-21T03:53:14.144207-04:00; exit 1; runner elapsed 38.658 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### B038_cycle_groups_seq2_rerun
- In progress: 2026-10-21T03:54:01.022118-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B038_cycle_groups_seq2_rerun.json`; log `logs/targets/prefill_decode_seq_len/B038_cycle_groups_seq2_rerun.log`; patch `logs/targets/prefill_decode_seq_len/B038_cycle_groups_seq2_rerun.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B038_cycle_groups_seq2_rerun`
- Completed: 2026-10-21T03:56:28.612126-04:00; exit 1; runner elapsed 147.588 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.
- Note (B034, N = 2, B032 code): no plan. The decode buckets (0, 1, 3) have 18 cyclic components of 7 classes (one per layer: a step's one-position residual, the two-position residual it can be sliced from and scattered back into, their copies), above the unroll cap of 6, so their relaxations were skipped as cyclic and the root heuristic had no plan; the search found none in 120 s.
- Note (B035, N = 1; B036, N = 0): (1) unroll cap 16 (a component of k classes costs k^2 classes). At N = 2 that made the decode relaxations usable but degenerate: necessary groups of the root 5 (N = 1: 1711), region chain 1, one region of 66210 classes, one load evaluation 57 s. (2) The classes of a cyclic component form one group (each can be derived from the others; a cycle-free plan produces some member through an outside alternative): necessary groups 548, but the greedy containment chain still stopped at 2 links because (3) group members included unreachable classes (an original class whose component is entered through another member: only its lower level copies are reachable). Members are now the classes reachable from the root, level copies included, and (4) the containment chain is the longest one (DP) instead of greedy. N = 2 decode: chain 346, 347 regions (largest 2535), root bound 7.41 (decode buckets 7.83). (5) A bucket whose largest region exceeds 32768 classes gets no relaxation bound and only cheap estimates for the heuristic (degenerate mode). At N = 1 the load bound rises 8.22 -> 8.30 and the chain to 1344 links; the first plan improves 14.869 -> 14.694 and the first window pass reaches 14.0219; final values unchanged. Soundness: oracle case 41 (cyclic component group, optimum 4); `test --relaxation-bound` now also checks 2000 random cyclic graphs (layers of full tensor, part and GPU copy) against brute force over cycle-free selections; a mutant hiding reachable originals from member lists is not caught (not shown unsound on these graphs). Gate: oracle 46/46, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Verdict: ACCEPTED (frontier: N_f = 2 still has no plan; N = 1 integral 0.1543 -> 0.1439).
- Note (B037/B038, N = 2, B035 code): B037 crashed in planning with "Invalid kernel UID KernelId(4294967297)" (OpenCL enabled in official runs; the rerun B038 did not hit it). B038: root propagation 31 s (decode buckets 27 s each), root heuristic build 34 s, 26 initial plans predicting 307-1000 (one bucket far too expensive), every validation rejected (CacheRequirementPropagator, ViewSelectOffsetPropagator). Next: make the N = 2 heuristic plans valid and cheaper to build.

### B039_two_estimates_seq1
- In progress: 2026-10-21T05:04:05.187046-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B039_two_estimates_seq1.json`; log `logs/targets/prefill_decode_seq_len/B039_two_estimates_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B039_two_estimates_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B039_two_estimates_seq1`
- Completed: 2026-10-21T05:06:21.340829-04:00; exit 0; runner elapsed 136.152 s.
  `[SearchEngine volume] iterations=941 elapsed=114.44291365399999 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603877.9583310161 log2-remaining-fraction=-609.98117996798828 eliminated-bits=609.98117996798828 remaining-nodes=774 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.640893604001356 normalized=0.14700744670001129 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.356577157974243 samples=178`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792573459.5764139 program-deadline=1792573565.2198486`
- First incumbent: 14.356565237 s, cost 14.6941; best cost 14.021854.

### B040_two_estimates_seq0
- In progress: 2026-10-21T05:06:21.414690-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B040_two_estimates_seq0.json`; log `logs/targets/prefill_decode_seq_len/B040_two_estimates_seq0.log`; patch `logs/targets/prefill_decode_seq_len/B040_two_estimates_seq0.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 0 --ignore-checkpoints --run-dir logs/gemma_output_runs/B040_two_estimates_seq0`
- Completed: 2026-10-21T05:08:33.661309-04:00; exit 0; runner elapsed 132.243 s.
  `[SearchEngine volume] iterations=700 elapsed=115.01273507099999 total-domain-bits=524409.59663406154 remaining-domain-bits=523839.37109567865 log2-remaining-fraction=-570.22553838288877 eliminated-bits=570.22553838288877 remaining-nodes=701 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.125585176707892 normalized=0.10104654313923243 final-gap=0.048315441616851904 final-incumbent=276.99084499999998 final-bound=263.60791 first-incumbent-elapsed=6.6489806175231934 samples=84`
  `[SearchEngine] New best cost: 276.991 at iteration 0 source=root-heuristic event-epoch=1792573590.4414904 program-deadline=1792573700.7925208`
- First incumbent: 9.648969650 s, cost 276.991; best cost 276.991.

### B041_two_estimates_seq2
- In progress: 2026-10-21T05:08:33.726540-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B041_two_estimates_seq2.json`; log `logs/targets/prefill_decode_seq_len/B041_two_estimates_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B041_two_estimates_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B041_two_estimates_seq2`
- Completed: 2026-10-21T05:11:01.108415-04:00; exit 1; runner elapsed 147.380 s.
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792573817.111913 program-deadline=1792573857.9277771`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 79.184135914 s, cost 16.4348; best cost 16.4348.

### B042_deadlines_seq2
- In progress: 2026-10-21T05:26:27.942864-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B042_deadlines_seq2.json`; log `logs/targets/prefill_decode_seq_len/B042_deadlines_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B042_deadlines_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B042_deadlines_seq2`
- Completed: 2026-10-21T05:29:10.277152-04:00; exit 0; runner elapsed 162.331 s.
  `[SearchEngine volume] iterations=2 elapsed=90.914518118000004 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380633.870580567 log2-remaining-fraction=-0.48573580011725426 eliminated-bits=0.48573580011725426 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=98.846381665331478 normalized=0.8237198472110957 final-gap=0.4225514067658942 final-incumbent=16.434753000000001 final-bound=9.4902250000000006 first-incumbent-elapsed=77.152817726135254 samples=7`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792574892.1675906 program-deadline=1792574932.0147855`
- First incumbent: 80.152805090 s, cost 16.4348; best cost 16.434753.

### B043_deadlines_seq1
- In progress: 2026-10-21T05:29:10.335213-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B043_deadlines_seq1.json`; log `logs/targets/prefill_decode_seq_len/B043_deadlines_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B043_deadlines_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B043_deadlines_seq1`
- Completed: 2026-10-21T05:31:26.316580-04:00; exit 0; runner elapsed 135.980 s.
  `[SearchEngine volume] iterations=1027 elapsed=114.35907192099999 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603867.2509226126 log2-remaining-fraction=-620.68858837150037 eliminated-bits=620.68858837150037 remaining-nodes=846 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.512122978722903 normalized=0.14593435815602421 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.38301157951355 samples=193`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792574964.6574733 program-deadline=1792575070.2744753`
- First incumbent: 14.382997990 s, cost 14.6941; best cost 14.021854.
- Note (B039-B042): (1) `markCacheAlternatives` treated every alternative of a class without a base as usable without the cache; CacheRequirementPropagator rejects a scatter/cache e-node without a cache variable, so a bucket-2 `Scatter_F32_ND_CUDA` made every N = 2 plan fail validation. Such alternatives are now marked cache-only (never usable: no base is ever cached). (2) The initial plans' region estimates (relaxation region values) price work below a class's region as paid: at N = 2 bucket 2 then rebuilt the LM-head weight slices itself (FastContiguous 40 ms each, copies 70 ms), its cache bases fell out of the cache set ("a cached base must be selected in every bucket") and every bucket recomputed the 640 MB LM-head weight path (decode ~208 ms per bucket, plans 300+). Cheap from-scratch estimates (`cheapEstimates`, least weighted cost per class, sharing counted per use) give a valid 16.43 at N = 2 but 86.7 at N = 1; starts now use both (cheap first; region starts stop after 8 s of building), and identical buckets share estimates. (3) The validation reserve (4 x build + 10 s) assumed builds without estimate evaluation; capped at 20 s (at N = 2 it left no time for local search). (4) At N = 2 the stride intervals and the window searches (2M-node enumerations of regions with thousands of classes) ran past the program deadline and the supervisor killed the compile (B041); stride intervals now stop at the refinement deadline and each window search at 2 s. N = 2 timeline (B042): saturation 12 s, root propagation 31 s (decode relaxations 27 s each: 347 regions of up to 2535 classes, one load evaluation 0.45 s vs 0.01 s at N = 1), heuristic build 12 s, first incumbent 77 s, flexible windows 23 s (bound 10.1 in the long diagnostic). Also tried: groups along SLICE views (one-position form of a tensor and the two-position form it is sliced from): the chain stays at 321 links, root propagation 46 s (not kept). Verdict: ACCEPTED (frontier: first complete N = 2 run, normalized 0.8237; N* = 1 kept).

### B044_early_root_seq2
- In progress: 2026-10-21T05:38:48.113291-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B044_early_root_seq2.json`; log `logs/targets/prefill_decode_seq_len/B044_early_root_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B044_early_root_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B044_early_root_seq2`
- Completed: 2026-10-21T05:41:15.099921-04:00; exit 1; runner elapsed 146.985 s.
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792575607.5628364 program-deadline=1792575671.8881092`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 55.674727201 s, cost 16.4348; best cost 16.4348.

### B045_refine_deadlines_seq2
- In progress: 2026-10-21T05:46:40.687948-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B045_refine_deadlines_seq2.json`; log `logs/targets/prefill_decode_seq_len/B045_refine_deadlines_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B045_refine_deadlines_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B045_refine_deadlines_seq2`
- Completed: 2026-10-21T05:49:23.020821-04:00; exit 0; runner elapsed 162.330 s.
  `[SearchEngine volume] iterations=3 elapsed=90.521323198000005 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380591.8362703025 log2-remaining-fraction=-42.520046064630151 eliminated-bits=42.520046064630151 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=82.621857381119483 normalized=0.68851547817599568 final-gap=0.38390178422517213 final-incumbent=16.434753000000001 final-bound=10.125422 first-incumbent-elapsed=53.018368244171143 samples=8`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792576080.8420405 program-deadline=1792576144.8236861`
- First incumbent: 56.018354416 s, cost 16.4348; best cost 16.434753.

### B046_refine_deadlines_seq1
- In progress: 2026-10-21T05:51:08.445054-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B046_refine_deadlines_seq1.json`; log `logs/targets/prefill_decode_seq_len/B046_refine_deadlines_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B046_refine_deadlines_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B046_refine_deadlines_seq1`
- Completed: 2026-10-21T05:53:24.538395-04:00; exit 0; runner elapsed 136.092 s.
  `[SearchEngine volume] iterations=1011 elapsed=114.377631225 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603868.4464850735 log2-remaining-fraction=-619.49302591057494 eliminated-bits=619.49302591057494 remaining-nodes=836 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.485622410336433 normalized=0.14571352008613694 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.342846870422363 samples=192`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792576282.8228369 program-deadline=1792576388.4800022`
- First incumbent: 14.342834711 s, cost 14.6941; best cost 14.021854.
- Note (B044-B046): at N = 2 the root threshold and global splits took ~18 s of the decode buckets' 28 s root relaxation and raised nothing; when the load maximization alone takes more than 1 s they are left to the refinement levels (N = 1: 0.5 s, unchanged). Root propagation 31 s -> 13 s; region-estimate starts limited to 3 s (N = 1 builds in 2 s): heuristic build 12.6 s -> 7 s; first incumbent 77 s -> 53 s. B044 was then killed at 120 s by a stride-interval pass (one pass ~20 s at N = 2, the deadline was only checked between strides): interval evaluation now stops at the deadline (remaining windows 0, the call proves nothing), and the global split's candidates too (each finished one is valid). Gate: oracle 46/46, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. B045: normalized 0.6885 (B042 0.8237), final gap 0.3839. Verdict: ACCEPTED as frontier change at N_f = 2.

### B047_pattern_race_seq2
- In progress: 2026-10-21T06:44:01.759431-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B047_pattern_race_seq2.json`; log `logs/targets/prefill_decode_seq_len/B047_pattern_race_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B047_pattern_race_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B047_pattern_race_seq2`
- Completed: 2026-10-21T06:46:43.427558-04:00; exit 0; runner elapsed 161.667 s.
  `[SearchEngine volume] iterations=2 elapsed=89.943118420000005 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380633.870580567 log2-remaining-fraction=-0.48573580011725426 eliminated-bits=0.48573580011725426 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=82.91377326913549 normalized=0.69094811057612904 final-gap=0.38390178422517213 final-incumbent=16.434753000000001 final-bound=10.125422 first-incumbent-elapsed=53.384503126144409 samples=7`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792579522.3043385 program-deadline=1792579585.9198496`
- First incumbent: 56.384488821 s, cost 16.4348; best cost 16.434753.

### B048_pattern_race_seq1
- In progress: 2026-10-21T06:46:43.487067-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B048_pattern_race_seq1.json`; log `logs/targets/prefill_decode_seq_len/B048_pattern_race_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B048_pattern_race_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B048_pattern_race_seq1`
- Completed: 2026-10-21T06:48:59.625494-04:00; exit 0; runner elapsed 136.136 s.
  `[SearchEngine volume] iterations=1024 elapsed=114.27870284700001 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603867.3606226449 log2-remaining-fraction=-620.57888833922334 eliminated-bits=620.57888833922334 remaining-nodes=843 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.58654468368599 normalized=0.14655453903071658 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.43228554725647 samples=192`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792579617.9681487 program-deadline=1792579723.5358763`
- First incumbent: 14.432272434 s, cost 14.6941; best cost 14.021854.
- Note (B047/B048): the intermittent "Invalid kernel UID KernelId(4294967297)" (B037, and 1 of 4 short N = 2 compiles) is a data race in saturation: `FusionRule::Pattern::getCandidateKernelIds()` filled a `mutable` cache lazily, and all rule clones share one `PatternSet` across the saturation threads (backtrace: `FusionRule::apply` <- `Planner::saturate` <- `saturateBucket` in `planAll`'s threads). The candidates are now resolved while the shared set is built (thread-safe static initialisation), before it is shared. Saturation unchanged: `tests/verify_saturation.py` (OpenCL disabled) EQUIVALENT at seq_len 2 and 8 against gold generated from the tree without the fix. The old gold (made at session start, 99 enodes / 154 classes at seq_len 2) no longer matches either this tree or HEAD's rewrite.hpp (96 / 145) - saturation itself changed outside the code, most likely through `benchmarks/records.bin` (rewritten by calibration runs at 02:00); gold regenerated from the current tree (old gold kept in the scratchpad), and HEAD's ConsumerWeightReuseRule matches it too. Gate: oracle 46/46, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`.

### B049_deferred_load_seq2
- In progress: 2026-10-21T08:26:03.248114-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B049_deferred_load_seq2.json`; log `logs/targets/prefill_decode_seq_len/B049_deferred_load_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B049_deferred_load_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B049_deferred_load_seq2`
- Completed: 2026-10-21T08:28:45.100935-04:00; exit 0; runner elapsed 161.851 s.
  `[SearchEngine volume] iterations=3 elapsed=90.788324232999997 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380591.8362703025 log2-remaining-fraction=-42.520046064630151 eliminated-bits=42.520046064630151 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=83.296822883055867 normalized=0.69414019069213218 final-gap=0.38390178422517213 final-incumbent=16.434753000000001 final-bound=10.125422 first-incumbent-elapsed=43.994899272918701 samples=9`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792585634.1420507 program-deadline=1792585707.1471646`
- First incumbent: 46.994886160 s, cost 16.4348; best cost 16.434753.

### B050_deferred_load_seq1
- In progress: 2026-10-21T08:28:45.160071-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B050_deferred_load_seq1.json`; log `logs/targets/prefill_decode_seq_len/B050_deferred_load_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B050_deferred_load_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B050_deferred_load_seq1`
- Completed: 2026-10-21T08:31:01.421243-04:00; exit 0; runner elapsed 136.260 s.
  `[SearchEngine volume] iterations=1027 elapsed=114.334157197 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603867.2509226126 log2-remaining-fraction=-620.68858837150037 eliminated-bits=620.68858837150037 remaining-nodes=846 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.479819019528605 normalized=0.1456651584960717 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.325259923934937 samples=193`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792585739.5482373 program-deadline=1792585845.2229927`
- First incumbent: 14.325244665 s, cost 14.6941; best cost 14.021854.
- Note (B049, N = 2): deferring the decode buckets' root load maximization (9 s) to the first refinement level moves the first incumbent from 53 s to 44 s, but the bound is then 0 (gap 1) until that level ends ~9 s later, so the integral does not improve (0.6941 vs B047 0.6909). Verdict: REJECTED, reverted.
- Rejected at N = 2 (diagnostics, reverted): (1) a third estimate kind (children with several parents free): no better initial plan, slower build. (2) Window moves across groups: per window the bound's selection runs the last layers' MLP on the GPU (plan: two-position CPU dots, 0.39 ms each; window plan 0.853 vs bound 0.113), but such a splice adds the GPU weight copies to the cache set and bucket 0 does not select them; retrying jointly with the other groups' windows holding those bases (window bases, selections computed on demand for windows the plan meets) finds no window: at N = 2 most windows exceed 512 classes (346-link chain, regions up to 2535 classes) and get neither bases nor selections. (3) Groups by e-class logical id (one- and two-position forms of a tensor): no change (chain 323), the forms do not share logical ids. At N = 2 the decode buckets' plan is 16.31 vs bound 11.05, bucket 2 16.56 vs 7.69; both sides are limited by the coarse region chain (only the residual stream's cyclic component per layer is necessary; each intermediate tensor has a one-position and a two-position form, neither necessary).

### B051_shared_identical_seq2
- In progress: 2026-10-21T09:23:03.620401-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B051_shared_identical_seq2.json`; log `logs/targets/prefill_decode_seq_len/B051_shared_identical_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B051_shared_identical_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B051_shared_identical_seq2`
- Completed: 2026-10-21T09:25:43.821951-04:00; exit 0; runner elapsed 160.200 s.
  `[SearchEngine volume] iterations=8 elapsed=93.448760992000004 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380376.90177707 log2-remaining-fraction=-257.45453929714859 eliminated-bits=257.45453929714859 remaining-nodes=4 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=81.049107266583945 normalized=0.67540922722153285 final-gap=0.38242588738632094 final-incumbent=16.434753000000001 final-bound=10.149678 first-incumbent-elapsed=50.706627607345581 samples=9`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792589059.1723089 program-deadline=1792589125.4656932`
- First incumbent: 53.706615686 s, cost 16.4348; best cost 16.434753.

### B052_shared_identical_seq1
- In progress: 2026-10-21T09:25:43.882515-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B052_shared_identical_seq1.json`; log `logs/targets/prefill_decode_seq_len/B052_shared_identical_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B052_shared_identical_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B052_shared_identical_seq1`
- Completed: 2026-10-21T09:27:59.896278-04:00; exit 0; runner elapsed 136.012 s.
  `[SearchEngine volume] iterations=1043 elapsed=114.286099584 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603864.3686602244 log2-remaining-fraction=-623.57085075974464 eliminated-bits=623.57085075974464 remaining-nodes=862 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.627283893698898 normalized=0.14689403244749083 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.490795135498047 samples=196`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792589158.3661175 program-deadline=1792589263.8753343`
- First incumbent: 14.490783215 s, cost 14.6941; best cost 14.021854.
- Note (B051, N = 2): between saturation (12.6 s) and search start (27 s) the planner reduces every bucket (rebuild, e-node infos with heuristic costs, pruning, cycle passes, infos again): bucket 0 alone 5.5 s, then the others in parallel (buckets 1 and 3 ~8 s each; e-node infos 3.2-4.7 s, the recomputation after the cycle passes 1.7-2.5 s). Buckets with the same specification (`Bucket ==`, logged as "identical to bucket p"; buckets 0 and 1 at N = 1 and N = 2) were saturated and reduced separately; the later one now takes the earlier one's saturated e-graph and reduced state (e-graph, e-node infos, cycle report). Search start 27.0 s -> 24.6 s at N = 2; identical results (plan 16.4348, flexible bound 10.1254; N = 1 unchanged). Gate: oracle 46/46, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Verdict: ACCEPTED as frontier change at N_f = 2 (0.6909 -> 0.6754).
- Rejected (diagnostics, reverted): a targeted repair of cached bases a weighted bucket leaves unselected (switch the active class with an alternative reading the base): no change at N = 2 (the last layers' GPU weight copies are never in the cache set; extraction picks the CPU route in every bucket there). Flexible windows above 512 classes at N = 2 (offline, decode checkpoint): 2048 classes 11.08 -> 11.22 in 126 s instead of 10 s.

### B053_size_cache_seq2
- In progress: 2026-10-21T09:39:46.750502-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B053_size_cache_seq2.json`; log `logs/targets/prefill_decode_seq_len/B053_size_cache_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B053_size_cache_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B053_size_cache_seq2`
- Completed: 2026-10-21T09:42:20.610791-04:00; exit 0; runner elapsed 153.859 s.
  `[SearchEngine volume] iterations=9 elapsed=97.173361944000007 total-domain-bits=5380634.3563163672 remaining-domain-bits=5374329.8334256047 log2-remaining-fraction=-6304.522890762426 eliminated-bits=6304.522890762426 remaining-nodes=3 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=77.863153509343078 normalized=0.64885961257785896 final-gap=0.38044094730234163 final-incumbent=16.434753000000001 final-bound=10.1823 first-incumbent-elapsed=45.657874345779419 samples=9`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792590052.2064424 program-deadline=1792590123.5485823`
- First incumbent: 48.657860041 s, cost 16.4348; best cost 16.434753.

### B054_size_cache_seq1
- In progress: 2026-10-21T09:42:20.679915-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B054_size_cache_seq1.json`; log `logs/targets/prefill_decode_seq_len/B054_size_cache_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B054_size_cache_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B054_size_cache_seq1`
- Completed: 2026-10-21T09:44:36.679254-04:00; exit 0; runner elapsed 135.998 s.
  `[SearchEngine volume] iterations=1043 elapsed=114.348746078 total-domain-bits=1604487.9395109841 remaining-domain-bits=1603864.3686602244 log2-remaining-fraction=-623.57085075974464 eliminated-bits=623.57085075974464 remaining-nodes=862 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=17.484060599177294 normalized=0.1457005049931441 final-gap=0.047752030508946872 final-incumbent=14.021853999999999 final-bound=13.352282000000001 first-incumbent-elapsed=11.34686017036438 samples=196`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792590154.9628744 program-deadline=1792590260.6160266`
- First incumbent: 14.346847773 s, cost 14.6941; best cost 14.021854.
- Note (B053, N = 2): profile of the planner setup between saturation and search: `Planner::computeENodeHeuristicCosts` (the cost / critical-path / memory least fixed point over each bucket's e-graph) recomputed `getStorageSpanBytes` for every child and the output on every re-evaluation of an enode. Sizes are now computed once per class and per enode (same values). Bucket reduction: bucket 0 5.45 s -> 3.07 s, bucket 3 8.2 s -> 3.2 s; search start 24.6 s -> 19.5 s; identical plans and bounds (N = 2: 16.4348 / 10.1254 after the flexible windows; N = 1 unchanged). Gate: oracle 46/46, search_gap, the four native suites, `--selection-reachability`, `--relaxation-bound`. Verdict: ACCEPTED as frontier change at N_f = 2 (0.6754 -> 0.6489).

### B055_hashcons_list_seq2
- In progress: 2026-10-21T10:22:06.099181-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B055_hashcons_list_seq2.json`; log `logs/targets/prefill_decode_seq_len/B055_hashcons_list_seq2.log`; patch `logs/targets/prefill_decode_seq_len/B055_hashcons_list_seq2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/B055_hashcons_list_seq2`
- Completed: 2026-10-21T10:24:40.359353-04:00; exit 0; runner elapsed 154.258 s.
  `[SearchEngine volume] iterations=11 elapsed=99.220544712000006 total-domain-bits=5380634.3563163672 remaining-domain-bits=5374323.7211558856 log2-remaining-fraction=-6310.6351604815573 eliminated-bits=6310.6351604815573 remaining-nodes=5 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=77.488893883403904 normalized=0.64574078236169918 final-gap=0.38044094730234163 final-incumbent=16.434753000000001 final-bound=10.1823 first-incumbent-elapsed=44.938707113265991 samples=10`
  `[SearchEngine] New best cost: 16.4348 at iteration 0 source=root-heuristic event-epoch=1792592589.896069 program-deadline=1792592661.957375`
- First incumbent: 47.938694000 s, cost 16.4348; best cost 16.434753.

### B056_hashcons_list_seq1
- In progress: 2026-10-21T10:24:40.426816-04:00
- Metadata `logs/targets/prefill_decode_seq_len/B056_hashcons_list_seq1.json`; log `logs/targets/prefill_decode_seq_len/B056_hashcons_list_seq1.log`; patch `logs/targets/prefill_decode_seq_len/B056_hashcons_list_seq1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/B056_hashcons_list_seq1`
- Note (B055/B056, review): the `ENodeIdList` hash-cons change was never evaluated (B056 has no completion) and is reverted; the tree is B053 again.
- Note (review, B035 mutant): hiding reachable originals of cyclic components from group member lists (members = level copies only) is not caught by any test because it does not produce an invalid bound: no counterexample in 60000 graphs from a new generator of small random graphs with arbitrary back edges (copies, views and free ops forming components of any shape), and every original reaches its lower level copies through a zero-cost alias whose child is always in its support, so plans can be mapped onto the level copies. `test --relaxation-bound` now runs 2000 of these graphs; the real code passes 60000. Sanity check: a mutant dropping the top level's internal alternatives (unsound) fails at trial 400016.
