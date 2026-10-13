# Gap-integral improvements record

Task: follow `docs/agents/improve_gap_integral.md`. Runs are appended by `tests/search_experiment.py --record gap`; logs, metadata and source patches live in `logs/gap_improvements/`.

## Fixed evaluation contract
- Command (cwd repository root): `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Build: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Hardware: Snapdragon X Elite X1E78100 (Windows arm64), 31.5 GB RAM.
- Base revision: `20bda2dd` plus the gap-measurement infrastructure (G0).

## Runs

### G000_build_infrastructure
- In progress: 2026-10-12T20:56:38.125705-04:00
- Metadata `logs/gap_improvements/G000_build_infrastructure.json`; log `logs/gap_improvements/G000_build_infrastructure.log`; patch `logs/gap_improvements/G000_build_infrastructure.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T20:57:34.551805-04:00; exit 0; runner elapsed 56.406 s.

### G001_oracle_infrastructure
- In progress: 2026-10-12T20:57:40.021304-04:00
- Metadata `logs/gap_improvements/G001_oracle_infrastructure.json`; log `logs/gap_improvements/G001_oracle_infrastructure.log`; patch `logs/gap_improvements/G001_oracle_infrastructure.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T20:57:42.659106-04:00; exit 0; runner elapsed 2.609 s.
- Oracle verdict: 28/28 cases passed.

### G002_gap_unit_tests
- In progress: 2026-10-12T20:57:42.972881-04:00
- Metadata `logs/gap_improvements/G002_gap_unit_tests.json`; log `logs/gap_improvements/G002_gap_unit_tests.log`; patch `logs/gap_improvements/G002_gap_unit_tests.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-12T20:57:43.056431-04:00; exit 0; runner elapsed 0.047 s.

### G003_native__private_layout
- In progress: 2026-10-12T20:57:43.299626-04:00
- Metadata `logs/gap_improvements/G003_native__private_layout.json`; log `logs/gap_improvements/G003_native__private_layout.log`; patch `logs/gap_improvements/G003_native__private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-12T20:57:43.399366-04:00; exit 0; runner elapsed 0.063 s.

### G003_native__write_after_read_cache
- In progress: 2026-10-12T20:57:43.630623-04:00
- Metadata `logs/gap_improvements/G003_native__write_after_read_cache.json`; log `logs/gap_improvements/G003_native__write_after_read_cache.log`; patch `logs/gap_improvements/G003_native__write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-12T20:57:43.709094-04:00; exit 0; runner elapsed 0.047 s.

### G003_native__incremental_propagators
- In progress: 2026-10-12T20:57:43.930772-04:00
- Metadata `logs/gap_improvements/G003_native__incremental_propagators.json`; log `logs/gap_improvements/G003_native__incremental_propagators.log`; patch `logs/gap_improvements/G003_native__incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-12T20:57:44.016335-04:00; exit 0; runner elapsed 0.047 s.

### G003_native__start_rank_window
- In progress: 2026-10-12T20:57:44.242078-04:00
- Metadata `logs/gap_improvements/G003_native__start_rank_window.json`; log `logs/gap_improvements/G003_native__start_rank_window.log`; patch `logs/gap_improvements/G003_native__start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-12T20:57:44.313595-04:00; exit 0; runner elapsed 0.047 s.

### G004_baseline_decode
- In progress: 2026-10-12T20:57:50.316110-04:00
- Metadata `logs/gap_improvements/G004_baseline_decode.json`; log `logs/gap_improvements/G004_baseline_decode.log`; patch `logs/gap_improvements/G004_baseline_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T21:07:50.781437-04:00; exit 0; runner elapsed 600.422 s.
  `[SearchEngine volume] iterations=86687 elapsed=585.9834386 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82479 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.344000000 budget=600.0`
  `[Program deadline] overshoot=0.184663`
  `[Program gap] primal-dual-integral=594.99550414939677 normalized=0.99165917358232791 final-gap=0.9890828260156469 final-incumbent=75.009704999999997 final-bound=0.81889400000000001 first-incumbent-elapsed=136.02542734146118 samples=2037`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791853206.4378331 program-deadline=1791853670.4124308`
- First incumbent: 136.025402308 s, cost 75.0097; best cost 75.009705.

### G005_baseline_repeat_decode
- In progress: 2026-10-12T21:08:05.751554-04:00
- Metadata `logs/gap_improvements/G005_baseline_repeat_decode.json`; log `logs/gap_improvements/G005_baseline_repeat_decode.log`; patch `logs/gap_improvements/G005_baseline_repeat_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T21:18:06.023621-04:00; exit 0; runner elapsed 600.266 s.
  `[SearchEngine volume] iterations=86690 elapsed=585.86402859999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82482 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.156000000 budget=600.0`
  `[Program deadline] overshoot=0.0360277`
  `[Program gap] primal-dual-integral=594.97932576048686 normalized=0.99163220960081144 final-gap=0.9890828260156469 final-incumbent=75.009704999999997 final-bound=0.81889400000000001 first-incumbent-elapsed=134.51996207237244 samples=2031`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791853820.3513463 program-deadline=1791854285.8314052`
- First incumbent: 134.519941092 s, cost 75.0097; best cost 75.009705.

### G006_baseline_repeat_decode
- In progress: 2026-10-12T21:18:06.206482-04:00
- Metadata `logs/gap_improvements/G006_baseline_repeat_decode.json`; log `logs/gap_improvements/G006_baseline_repeat_decode.log`; patch `logs/gap_improvements/G006_baseline_repeat_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T21:28:06.472516-04:00; exit 0; runner elapsed 600.234 s.
  `[SearchEngine volume] iterations=86627 elapsed=585.9346041 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82419 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.172000000 budget=600.0`
  `[Program deadline] overshoot=0.0236514`
  `[Program gap] primal-dual-integral=594.98983919234604 normalized=0.99164973198724338 final-gap=0.9890828260156469 final-incumbent=75.009704999999997 final-bound=0.81889400000000001 first-incumbent-elapsed=135.45565629005432 samples=2020`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791854421.7397568 program-deadline=1791854886.2841282`
- First incumbent: 135.455628633 s, cost 75.0097; best cost 75.009705.

### G007_build_interval_env
- In progress: 2026-10-12T21:28:41.874668-04:00
- Metadata `logs/gap_improvements/G007_build_interval_env.json`; log `logs/gap_improvements/G007_build_interval_env.log`; patch `logs/gap_improvements/G007_build_interval_env.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`

- G000 build, G001 oracle 28/28 (now including the certified-bound validity check: root_lower_bound and max_certified_bound <= optimum on every case), G002 gap unit tests PASS, G003 native suites PASS.
- **Original gap baseline (O9 + gap measurement): G004/G005/G006.** Normalized integral 0.99165917 / 0.99163221 / 0.99164973 (median 0.99164973, spread 2.7e-5); final gap 0.98908283; certified bound 0.52775 at root -> 0.818894 at deadline; final incumbent 75.009705; eliminated bits 3114375.5045604468 (identical to O9); first incumbent 136.025 / 134.520 / 135.456 s (median 135.456; O9 earlier: 135.287 / 131.236 / 132.502). Overshoots 0.185 / 0.036 / 0.024 s.
- G007 build adds the diagnostic TG_SEARCH_GAP_INTERVAL override; G008-G011 alternate sampling off/on to check that gap sampling does not cost first-incumbent time (diagnostic).
- Completed: 2026-10-12T21:29:40.162279-04:00; exit 0; runner elapsed 58.266 s.

### G008_diag_ab_sampling_off
- In progress: 2026-10-12T21:29:40.387329-04:00
- Metadata `logs/gap_improvements/G008_diag_ab_sampling_off.json`; log `logs/gap_improvements/G008_diag_ab_sampling_off.log`; patch `logs/gap_improvements/G008_diag_ab_sampling_off.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T21:39:40.647533-04:00; exit 0; runner elapsed 600.218 s.
  `[SearchEngine volume] iterations=86589 elapsed=585.79691190000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737137.6591798724 log2-remaining-fraction=-3114375.5045604468 eliminated-bits=3114375.5045604468 remaining-nodes=82381 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.156000000 budget=600.0`
  `[Program deadline] overshoot=0.0357914`
  `[Program gap] primal-dual-integral=596.72646727481788 normalized=0.99454411212469651 final-gap=0.99296424376019077 final-incumbent=75.009704999999997 final-bound=0.52775000000000005 first-incumbent-elapsed=134.72908759117126 samples=3`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791855115.1970127 program-deadline=1791855580.4679501`
- First incumbent: 134.729062557 s, cost 75.0097; best cost 75.009705.

### G009_diag_ab_sampling_on
- In progress: 2026-10-12T21:39:40.839097-04:00
- Metadata `logs/gap_improvements/G009_diag_ab_sampling_on.json`; log `logs/gap_improvements/G009_diag_ab_sampling_on.log`; patch `logs/gap_improvements/G009_diag_ab_sampling_on.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- G007 build PASS. G008 (diagnostic, TG_SEARCH_GAP_INTERVAL=1e6, i.e. only root/incumbent/deadline samples): first incumbent 134.729 s, inside the G004-G006 range (134.520-136.025), so periodic gap sampling has no measurable cost on time to first incumbent. (Its normalized integral 0.99454 is higher only because the rising bound is not sampled; evaluations always use the default 0.25 s interval.) G009-G011 were INTERRUPTED by the user's request to free the machine (G009 killed at start; no metrics; G010/G011 never started).

## Comparison table
| Run | Implementation | Normalized integral | Final gap | Final bound | Final incumbent | First incumbent s | Eliminated bits | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| G004 | baseline (O9 + gap measurement) | 0.99165917 | 0.98908283 | 0.818894 | 75.009705 | 136.025 | 3114375.5045604468 | original gap baseline |
| G005 | baseline repeat | 0.99163221 | 0.98908283 | 0.818894 | 75.009705 | 134.520 | 3114375.5045604468 | original gap baseline |
| G006 | baseline repeat | 0.99164973 | 0.98908283 | 0.818894 | 75.009705 | 135.456 | 3114375.5045604468 | original gap baseline |
| G019 | H-G1 root relaxation bound | 0.69519291 | 0.60022876 | 29.986723 | 75.009705 | 142.528 | 3113714.3574055554 | rejected (bits, first incumbent) |
| G036 | H-G3 = H-G1 + root heuristic + local search | 0.22489496 | 0.19000642 | 29.986723 | 37.020939 | 25.794 | 3113714.3574055554 | rejected (bits) |
| G043 | H-G4b = H-G3 + largest-box-first + multi-start | 0.22517474 | 0.19000642 | 29.986723 | 37.020939 | 25.997 | 3114674.2440806637 | candidate |
| G044 | H-G4b confirmation 1 | 0.22507279 | 0.19000642 | 29.986723 | 37.020939 | 25.925 | 3114673.7763642874 | confirmation |
| G045 | H-G4b confirmation 2 | 0.22634102 | 0.19000642 | 29.986723 | 37.020939 | 26.864 | 3114673.7139701955 | **accepted (optimization O-G1)** |
| G065 | H-G7 startup speedup | 0.20517314 | 0.19000642 | 29.986723 | 37.020939 | 11.184 | 3114678.9098478779 | candidate |
| G066 | H-G7 confirmation 1 | 0.20451764 | 0.19000642 | 29.986723 | 37.020939 | 10.700 | 3114679.4985159514 | confirmation |
| G067 | H-G7 confirmation 2 | 0.20490511 | 0.19000642 | 29.986723 | 37.020939 | 10.986 | 3114677.5184289343 | **accepted (O-G2)** |
| G111 | H-G9 path-weighted interval bound + H-G8 cache-set moves | 0.17075091 | 0.15415536 | 31.246 | 36.94059 | 11.694 | 3114677.405939702 | candidate |
| G112 | H-G9 confirmation 1 | 0.17115002 | 0.15415536 | 31.246 | 36.94059 | 11.974 | 3114676.8035819973 | confirmation |
| G113 | H-G9 confirmation 2 | 0.17028015 | 0.15415536 | 31.246 | 36.94059 | 11.360 | 3114677.7607348966 | **accepted (O-G3)** |
| G128 | H-G10 exact window value | 0.13164706 | 0.11326227 | 32.756615 | 36.94059 | 12.362 | 3114678.0885961642 | candidate |
| G129 | H-G10 confirmation 1 | 0.13140356 | 0.11326227 | 32.756615 | 36.94059 | 12.196 | 3114675.052975974 | confirmation |
| G130 | H-G10 confirmation 2 | 0.13074864 | 0.11326227 | 32.756615 | 36.94059 | 11.755 | 3114678.244931801 | **accepted (O-G4)** |
| G196 | H-G14+H-G15+H-G16 cut DP, dyadic windows, chain-coupled DP | 0.05418719 | 0.03532226 | 35.635765 | 36.94059 | 11.656 | 3114679.8382587526 | candidate (superseded by H-G17) |
| G220 | H-G17 window partitions (+H-G14..16) | 0.04457380 | 0.02388 | 36.058403 | 36.94059 | 12.635 | 3114677.0745383548 | candidate |
| G221 | H-G17 confirmation 1 | 0.04418856 | 0.02388 | 36.058403 | 36.94059 | 12.406 | 3114677.980819905 | confirmation |
| G222 | H-G17 confirmation 2 | 0.04451394 | 0.02388 | 36.058403 | 36.94059 | 12.606 | 3114677.7607348966 | **accepted (O-G5)** |

Accepted-implementation reference for the gate (**H-G17, accepted 2026-10-13**): median normalized 0.04451394 (runs 0.04457380 / 0.04418856 / 0.04451394, spread 3.9e-4), median first incumbent 12.606 s (slowest 12.635 s), eliminated bits ≥ 3114677.07 (gate threshold from the original baseline 3114375.5045604468), first/deadline cost 39.3272 / 36.94059. Earlier accepted: O-G1 H-G4b 0.22517, O-G2 H-G7 0.20491, O-G3 H-G9 0.17075, O-G4 H-G10 0.13140.

Original gap baseline (for reference): median normalized 0.99164973, median first incumbent 135.456 s, eliminated bits 3114375.5045604468, costs 75.0097 / 75.009705.

## Resume checkpoint (current, 2026-10-13)
- **Accepted implementation: H-G17 (optimization O-G5), median normalized 0.04451 over G220/G221/G222 — the ≤ 0.05 goal is met.** Exact source: `logs/gap_improvements/G220_HG17_decode.patch` (identical to G221/G222, hash 98382B97…) on HEAD `66e37b62` (branch `faster-search`, uncommitted). On top of O-G4 it adds: the cut-DP window bound with chain coupling (`cutDpWindow`, `relaxation_bound.hpp`), bucket deduplication (`relaxationSignature`), parallel window evaluation, half-stride shifted windows and the best window partition (`bestWindowPartition`), brute-force tests in `tests/relaxation_bound.hpp`, and the "Exact window value / Cut DP / Window partitions / Identical buckets" paragraphs of `docs/core/relaxation_bound.md`.
- Working tree = accepted source; extension and test.exe built from it (G216). No running processes.
- Correctness of the accepted build: oracle G217 36/36 (certified-bound validity), gap unit tests G218, native G219 (5 suites; `--relaxation-bound` brute force over selections × dispatch orders incl. cut DP and window partitions).
- Decomposition of 0.0445: ≈0.021 before the first incumbent (12.6 s), ≈0.023 from the remaining gap 0.0239 (incumbent 36.94 vs bound 36.06).
- Possible further work: four phases per stride (+0.12 bound, +1.3 s root time, would need a faster root to keep the first-incumbent gate), the optimistic-summary pass in `computeENodeHeuristicCosts` (~1 s of startup under parallel contention), and primal improvements (per-layer schedules ≈1.303 vs 1.32 found by CP-SAT offline).

## Older checkpoint (H-G4b, superseded)
- **Accepted implementation: H-G4b (optimization O-G1), median normalized 0.22517 over G043/G044/G045 (goal ≤ 0.25 met).** Exact source: `logs/gap_improvements/G043_HG4b_decode.patch` (identical to the G044/G045 patches), on top of HEAD `a6a34bf0` (branch `faster-search`, not committed). Contents: `core/plan/relaxation_bound.hpp` (relaxation structures, support-set recursion over the necessary-group chain, lambda search), `core/plan/propagators/relaxation_bound.hpp` (root-only GLOBAL batch, `SearchState::bucket_relaxation_lower_bounds`), `core/plan/primal_heuristic.hpp` (region-estimate extraction, zero-weight covering extraction, cache consistency, list scheduling, local search), `core/plan/search_engine.hpp` (`tryPlan`, `runRootPrimalHeuristic` with multi-start local search, `recordIncumbent`, largest-box-first after a heuristic incumbent, root propagation timing log), `core/plan/search_node.hpp` / `selector.hpp` (largest-box-first comparator), `core/plan/search_state.hpp`, `core/plan/propagator.hpp`, `tests/relaxation_bound.hpp` + `test.cpp` (`--relaxation-bound`), oracle cases 29–32, `docs/core/relaxation_bound.md`.
- Working tree = accepted source (verified: the runner's captured patch was byte-identical to G043's), plus a doc-only refinement of `docs/core/relaxation_bound.md` (local search and largest-box-first sections). Extension and test.exe built from it (G039). No running processes.
- Correctness of the accepted build: oracle G040 32/32 (certified-bound validity included), gap unit tests G041 PASS, native G042 PASS (`--private-layout`, `--write-after-read-cache`, `--incremental-propagators`, `--start-rank-window`, `--relaxation-bound`).
- Trial in progress, not built or evaluated: **H-G5** background simulated annealing on each weighted bucket (refactors local search into `SelectionEditor`), with snapshots validated from the root at iteration boundaries, plus a relative tolerance so plans that only tie the incumbent are not re-validated. Saved as `logs/gap_improvements/HG5_background_improver_trial.diff` (against the accepted source) and full files `HG5_trial_primal_heuristic.hpp`, `HG5_trial_search_engine.hpp`. To resume: copy both files over `tensor_graphs_cpp/core/plan/{primal_heuristic,search_engine}.hpp`, touch them, build, run the oracle and native suites, then a 120 s diagnostic (watch the `[PrimalHeuristic] background` lines and the volume) before a timed evaluation.
- Other next steps (by expected value): (1) cut root propagation (10.2–10.6 s at gap 1; profile with `build.py --profile`, the accepted build logs `[SearchEngine] root propagation took`); (2) cut startup before `solve()` (about 14.5 s; diagnostic timeline script idea: run the worker directly with `TG_PROGRAM_DEADLINE` set and timestamp each output line); (3) a stronger bound than min-max engine load (the exact relaxation optimum is 30.36; the bound gives 29.99), for example one that couples load with precedence.

### G012_runner_smoke_gap_unit_tests
- In progress: 2026-10-12T21:41:15.418673-04:00
- Metadata `logs/gap_improvements/G012_runner_smoke_gap_unit_tests.json`; log `logs/gap_improvements/G012_runner_smoke_gap_unit_tests.log`; patch `logs/gap_improvements/G012_runner_smoke_gap_unit_tests.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-12T21:41:15.498098-04:00; exit 0; runner elapsed 0.063 s.
- G012: runner moved to tests/search_experiment.py (tools/ is git-ignored); smoke test of the gap unit tests through it PASS.

## Open items (deferred, not blocking)
- [ ] **Unfinished A/B test of gap-sampling overhead.** Goal: confirm that periodic gap sampling (`TG_SEARCH_GAP_INTERVAL` default 0.25 s) costs no time to first incumbent. Done so far: G008 (sampling off) = 134.729 s, inside the baseline range G004-G006 (134.520-136.025 s). Not done: G009-G011 were interrupted on 2026-10-12 to free the machine for the overnight agent. To finish, run alternating pairs **one at a time with nothing else running** (new run ids, since ids cannot be reused), then compare first-incumbent medians (sampling off vs on):
  ```bash
  TG_SEARCH_GAP_INTERVAL=1000000 .venv/Scripts/python.exe tests/search_experiment.py --record gap <id>_ab_sampling_off -m tests.test_gemma_output --config decode --optimal --timeout 600
  .venv/Scripts/python.exe tests/search_experiment.py --record gap <id>_ab_sampling_on -m tests.test_gemma_output --config decode --optimal --timeout 600
  ```
  If "on" is consistently slower by more than ~1 s, raise the default interval or make the open-node minimum incremental, then re-baseline.

### G013_build_HG1_relaxation_bound
- In progress: 2026-10-12T22:18:08.305566-04:00
- Metadata `logs/gap_improvements/G013_build_HG1_relaxation_bound.json`; log `logs/gap_improvements/G013_build_HG1_relaxation_bound.log`; patch `logs/gap_improvements/G013_build_HG1_relaxation_bound.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:19:05.710591-04:00; exit 0; runner elapsed 57.375 s.

### G014_oracle_HG1
- In progress: 2026-10-12T22:19:08.965357-04:00
- Metadata `logs/gap_improvements/G014_oracle_HG1.json`; log `logs/gap_improvements/G014_oracle_HG1.log`; patch `logs/gap_improvements/G014_oracle_HG1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T22:19:11.275468-04:00; exit 0; runner elapsed 2.282 s.
- Oracle verdict: 28/28 cases passed.

### G015_diag_HG1_bound_60s
- In progress: 2026-10-12T22:19:18.442737-04:00
- Metadata `logs/gap_improvements/G015_diag_HG1_bound_60s.json`; log `logs/gap_improvements/G015_diag_HG1_bound_60s.log`; patch `logs/gap_improvements/G015_diag_HG1_bound_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T22:20:18.626935-04:00; exit 0; runner elapsed 60.156 s.
  `[SearchEngine volume] iterations=27765 elapsed=45.2890443 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=27722 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.093000000 budget=60.0`
  `[Program deadline] overshoot=0.000230312`
  `[Program gap] primal-dual-integral=60 normalized=1 final-gap=1 final-incumbent=inf final-bound=0.52775000000000005 first-incumbent-elapsed=none samples=131`
- First incumbent: none within budget (+infinity); best cost None.

### G016_build_HG1_fix_trigger
- In progress: 2026-10-12T22:21:27.680549-04:00
- Metadata `logs/gap_improvements/G016_build_HG1_fix_trigger.json`; log `logs/gap_improvements/G016_build_HG1_fix_trigger.log`; patch `logs/gap_improvements/G016_build_HG1_fix_trigger.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:22:25.235097-04:00; exit 0; runner elapsed 57.531 s.

### G017_oracle_HG1_fix
- In progress: 2026-10-12T22:22:32.164265-04:00
- Metadata `logs/gap_improvements/G017_oracle_HG1_fix.json`; log `logs/gap_improvements/G017_oracle_HG1_fix.log`; patch `logs/gap_improvements/G017_oracle_HG1_fix.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T22:22:34.484714-04:00; exit 0; runner elapsed 2.282 s.
- Oracle verdict: 28/28 cases passed.

### G018_diag_HG1_bound_60s
- In progress: 2026-10-12T22:22:34.695145-04:00
- Metadata `logs/gap_improvements/G018_diag_HG1_bound_60s.json`; log `logs/gap_improvements/G018_diag_HG1_bound_60s.log`; patch `logs/gap_improvements/G018_diag_HG1_bound_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T22:23:34.961182-04:00; exit 0; runner elapsed 60.234 s.
  `[SearchEngine volume] iterations=28013 elapsed=45.752861899999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=27967 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.171000000 budget=60.0`
  `[Program deadline] overshoot=0.00177503`
  `[Program gap] primal-dual-integral=60 normalized=1 final-gap=1 final-incumbent=inf final-bound=29.986723000000001 first-incumbent-elapsed=none samples=134`
- First incumbent: none within budget (+infinity); best cost None.

### G019_HG1_decode
- In progress: 2026-10-12T22:23:47.234628-04:00
- Metadata `logs/gap_improvements/G019_HG1_decode.json`; log `logs/gap_improvements/G019_HG1_decode.log`; patch `logs/gap_improvements/G019_HG1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T22:33:47.936786-04:00; exit 0; runner elapsed 600.656 s.
  `[SearchEngine volume] iterations=541694 elapsed=585.82030980000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=537468 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.579000000 budget=600.0`
  `[Program deadline] overshoot=0.0158408`
  `[Program gap] primal-dual-integral=417.11574563116648 normalized=0.69519290938527745 final-gap=0.60022875706550238 final-incumbent=75.009704999999997 final-bound=29.986723000000001 first-incumbent-elapsed=142.52773904800415 samples=2168`
  `[SearchEngine] New best cost: 75.0097 at iteration 78266 event-epoch=1791858369.846884 program-deadline=1791858827.3191702`
- First incumbent: 142.527713776 s, cost 75.0097; best cost 75.009705.

### G020_build_HG2_primal
- In progress: 2026-10-12T22:35:41.625275-04:00
- Metadata `logs/gap_improvements/G020_build_HG2_primal.json`; log `logs/gap_improvements/G020_build_HG2_primal.log`; patch `logs/gap_improvements/G020_build_HG2_primal.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:36:39.822007-04:00; exit 0; runner elapsed 58.172 s.

### G021_native_relaxation_bound
- In progress: 2026-10-12T22:36:46.877267-04:00
- Metadata `logs/gap_improvements/G021_native_relaxation_bound.json`; log `logs/gap_improvements/G021_native_relaxation_bound.log`; patch `logs/gap_improvements/G021_native_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-12T22:36:47.244563-04:00; exit 0; runner elapsed 0.328 s.

### G022_oracle_HG2
- In progress: 2026-10-12T22:36:52.620574-04:00
- Metadata `logs/gap_improvements/G022_oracle_HG2.json`; log `logs/gap_improvements/G022_oracle_HG2.log`; patch `logs/gap_improvements/G022_oracle_HG2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T22:36:55.583874-04:00; exit 0; runner elapsed 2.937 s.
- Oracle verdict: 32/32 cases passed.

### G023_diag_HG2_60s
- In progress: 2026-10-12T22:37:00.950180-04:00
- Metadata `logs/gap_improvements/G023_diag_HG2_60s.json`; log `logs/gap_improvements/G023_diag_HG2_60s.log`; patch `logs/gap_improvements/G023_diag_HG2_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-12T22:38:01.170397-04:00; exit 0; runner elapsed 60.203 s.
  `[SearchEngine volume] iterations=26468 elapsed=45.7586467 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=26468 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.141000000 budget=60.0`
  `[Program deadline] overshoot=0.0265462`
  `[Program gap] primal-dual-integral=51.455030701298988 normalized=0.85758384502164975 final-gap=0.75253029526494153 final-incumbent=121.173309 final-bound=29.986723000000001 first-incumbent-elapsed=25.470644950866699 samples=130`
  `[SearchEngine] New best cost: 121.173 at iteration 0 source=root-heuristic event-epoch=1791859046.4954872 program-deadline=1791859081.0248706`
- First incumbent: 25.470616579 s, cost 121.173; best cost 121.173309.

### G024_build_diag_weightings
- In progress: 2026-10-12T22:38:17.218623-04:00
- Metadata `logs/gap_improvements/G024_build_diag_weightings.json`; log `logs/gap_improvements/G024_build_diag_weightings.log`; patch `logs/gap_improvements/G024_build_diag_weightings.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:39:15.486026-04:00; exit 0; runner elapsed 58.250 s.

### G025_diag_weightings_40s
- In progress: 2026-10-12T22:39:15.672510-04:00
- Metadata `logs/gap_improvements/G025_diag_weightings_40s.json`; log `logs/gap_improvements/G025_diag_weightings_40s.log`; patch `logs/gap_improvements/G025_diag_weightings_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-12T22:39:56.028056-04:00; exit 0; runner elapsed 40.328 s.
  `[SearchEngine volume] iterations=977 elapsed=25.452013900000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=978 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.265000000 budget=40.0`
  `[Program deadline] overshoot=0.0224278`
  `[Program gap] primal-dual-integral=36.568071954758416 normalized=0.91420179886896036 final-gap=0.75253029526494153 final-incumbent=121.173309 final-bound=29.986723000000001 first-incumbent-elapsed=26.131926536560059 samples=51`
  `[SearchEngine] New best cost: 121.173 at iteration 0 source=root-heuristic event-epoch=1791859181.8781421 program-deadline=1791859195.7462463`
- First incumbent: 26.131895781 s, cost 121.173; best cost 121.173.

### G026_build_diag_violations
- In progress: 2026-10-12T22:40:14.690266-04:00
- Metadata `logs/gap_improvements/G026_build_diag_violations.json`; log `logs/gap_improvements/G026_build_diag_violations.log`; patch `logs/gap_improvements/G026_build_diag_violations.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:41:12.410516-04:00; exit 0; runner elapsed 57.688 s.

### G027_diag_violations_40s
- In progress: 2026-10-12T22:41:12.584631-04:00
- Metadata `logs/gap_improvements/G027_diag_violations_40s.json`; log `logs/gap_improvements/G027_diag_violations_40s.log`; patch `logs/gap_improvements/G027_diag_violations_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-12T22:41:52.817413-04:00; exit 0; runner elapsed 40.219 s.
  `[SearchEngine volume] iterations=975 elapsed=25.299574 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=976 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.156000000 budget=40.0`
  `[Program deadline] overshoot=0.046989`
  `[Program gap] primal-dual-integral=36.60956449498579 normalized=0.91523911237464473 final-gap=0.75253029526494153 final-incumbent=121.173309 final-bound=29.986723000000001 first-incumbent-elapsed=26.299593687057495 samples=51`
  `[SearchEngine] New best cost: 121.173 at iteration 0 source=root-heuristic event-epoch=1791859298.9514234 program-deadline=1791859312.6518543`
- First incumbent: 26.299569130 s, cost 121.173; best cost 121.173.

### G028_build_HG2_covering
- In progress: 2026-10-12T22:43:02.499001-04:00
- Metadata `logs/gap_improvements/G028_build_HG2_covering.json`; log `logs/gap_improvements/G028_build_HG2_covering.log`; patch `logs/gap_improvements/G028_build_HG2_covering.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:44:00.361905-04:00; exit 0; runner elapsed 57.844 s.

### G029_diag_HG2_covering_40s
- In progress: 2026-10-12T22:44:00.539615-04:00
- Metadata `logs/gap_improvements/G029_diag_HG2_covering_40s.json`; log `logs/gap_improvements/G029_diag_HG2_covering_40s.log`; patch `logs/gap_improvements/G029_diag_HG2_covering_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-12T22:44:40.789567-04:00; exit 0; runner elapsed 40.219 s.
  `[SearchEngine volume] iterations=1060 elapsed=25.697258900000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=1061 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.156000000 budget=40.0`
  `[Program deadline] overshoot=0.0460784`
  `[Program gap] primal-dual-integral=28.854288072370576 normalized=0.72135720180926444 final-gap=0.23750711879765091 final-incumbent=39.327216999999997 final-bound=29.986723000000001 first-incumbent-elapsed=25.3825364112854 samples=54`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791859466.001812 program-deadline=1791859480.6193039`
- First incumbent: 25.382508039 s, cost 39.3272; best cost 39.327217.

### G030_build_HG3_local_search
- In progress: 2026-10-12T22:48:36.935698-04:00
- Metadata `logs/gap_improvements/G030_build_HG3_local_search.json`; log `logs/gap_improvements/G030_build_HG3_local_search.log`; patch `logs/gap_improvements/G030_build_HG3_local_search.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:49:35.522793-04:00; exit 0; runner elapsed 58.563 s.

### G031_diag_HG3_local_search_120s
- In progress: 2026-10-12T22:49:40.021830-04:00
- Metadata `logs/gap_improvements/G031_diag_HG3_local_search_120s.json`; log `logs/gap_improvements/G031_diag_HG3_local_search_120s.log`; patch `logs/gap_improvements/G031_diag_HG3_local_search_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-12T22:51:40.267068-04:00; exit 0; runner elapsed 120.218 s.
  `[SearchEngine volume] iterations=28445 elapsed=105.7184647 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=28441 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.156000000 budget=120.0`
  `[Program deadline] overshoot=0.0360744`
  `[Program gap] primal-dual-integral=43.586895778094892 normalized=0.36322413148412408 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=25.61226487159729 samples=344`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791859805.7089376 program-deadline=1791859900.0967007`
- First incumbent: 25.612236977 s, cost 39.3272; best cost 37.020939.

### G032_build_HG3
- In progress: 2026-10-12T22:52:07.222093-04:00
- Metadata `logs/gap_improvements/G032_build_HG3.json`; log `logs/gap_improvements/G032_build_HG3.log`; patch `logs/gap_improvements/G032_build_HG3.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T22:53:05.328096-04:00; exit 0; runner elapsed 58.078 s.

### G033_oracle_HG3
- In progress: 2026-10-12T22:53:05.503249-04:00
- Metadata `logs/gap_improvements/G033_oracle_HG3.json`; log `logs/gap_improvements/G033_oracle_HG3.log`; patch `logs/gap_improvements/G033_oracle_HG3.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T22:53:08.404922-04:00; exit 0; runner elapsed 2.875 s.
- Oracle verdict: 32/32 cases passed.

### G034_gap_unit_tests
- In progress: 2026-10-12T22:53:08.618802-04:00
- Metadata `logs/gap_improvements/G034_gap_unit_tests.json`; log `logs/gap_improvements/G034_gap_unit_tests.log`; patch `logs/gap_improvements/G034_gap_unit_tests.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-12T22:53:08.690325-04:00; exit 0; runner elapsed 0.031 s.

### G035_native_HG3_private_layout
- In progress: 2026-10-12T22:53:08.850419-04:00
- Metadata `logs/gap_improvements/G035_native_HG3_private_layout.json`; log `logs/gap_improvements/G035_native_HG3_private_layout.log`; patch `logs/gap_improvements/G035_native_HG3_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-12T22:53:09.054035-04:00; exit 0; runner elapsed 0.172 s.

### G035_native_HG3_write_after_read_cache
- In progress: 2026-10-12T22:53:09.228961-04:00
- Metadata `logs/gap_improvements/G035_native_HG3_write_after_read_cache.json`; log `logs/gap_improvements/G035_native_HG3_write_after_read_cache.log`; patch `logs/gap_improvements/G035_native_HG3_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-12T22:53:09.303512-04:00; exit 0; runner elapsed 0.047 s.

### G035_native_HG3_incremental_propagators
- In progress: 2026-10-12T22:53:09.477800-04:00
- Metadata `logs/gap_improvements/G035_native_HG3_incremental_propagators.json`; log `logs/gap_improvements/G035_native_HG3_incremental_propagators.log`; patch `logs/gap_improvements/G035_native_HG3_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-12T22:53:10.061834-04:00; exit 0; runner elapsed 0.547 s.

### G035_native_HG3_start_rank_window
- In progress: 2026-10-12T22:53:10.232698-04:00
- Metadata `logs/gap_improvements/G035_native_HG3_start_rank_window.json`; log `logs/gap_improvements/G035_native_HG3_start_rank_window.log`; patch `logs/gap_improvements/G035_native_HG3_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-12T22:53:10.303252-04:00; exit 0; runner elapsed 0.047 s.

### G035_native_HG3_relaxation_bound
- In progress: 2026-10-12T22:53:10.479562-04:00
- Metadata `logs/gap_improvements/G035_native_HG3_relaxation_bound.json`; log `logs/gap_improvements/G035_native_HG3_relaxation_bound.log`; patch `logs/gap_improvements/G035_native_HG3_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-12T22:53:11.080245-04:00; exit 0; runner elapsed 0.563 s.

### G036_HG3_decode
- In progress: 2026-10-12T22:53:17.424346-04:00
- Metadata `logs/gap_improvements/G036_HG3_decode.json`; log `logs/gap_improvements/G036_HG3_decode.log`; patch `logs/gap_improvements/G036_HG3_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T23:03:17.764622-04:00; exit 0; runner elapsed 600.312 s.
  `[SearchEngine volume] iterations=198626 elapsed=585.77112650000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=198591 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.234000000 budget=600.0`
  `[Program deadline] overshoot=0.0123253`
  `[Program gap] primal-dual-integral=134.93697452423575 normalized=0.22489495754039293 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=25.793687343597412 samples=2104`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791860023.2942424 program-deadline=1791860597.5005858`
- First incumbent: 25.793656588 s, cost 39.3272; best cost 37.020939.
- Notes G013–G036 (2026-10-12):
  - **H-G1 root relaxation bound** (`core/plan/relaxation_bound.hpp`, `RelaxationBoundPropagator`, doc `docs/core/relaxation_bound.md`): a Lagrangian over engine loads with the support-set recursion over a containment chain of necessary COPY_TO groups. G013 built it but the propagator never ran: the root pass enqueues dirty variables, never `kInvalidVarId`. G016 fixed the trigger (GLOBAL batch at the first quiescence). Oracle G017 28/28; bound 29.99 per decode bucket in 0.06 s (G018; CP-SAT exact relaxation 30.36, old root bound 0.53). Offline prototype (scratchpad `chainlb.py`): essentially exact against CP-SAT weighted extraction (18.927 vs 18.929 at λ=0.5, 29.90 vs 30.09 at λ=0.8).
  - **G019 H-G1 alone: normalized 0.6952** (baseline 0.9916), first incumbent 142.5 s, cost 75.0097. **Rejected:** eliminated bits 3113714.36 < 3114375.50, and first incumbent > 136.5 s. Cause of the bits: with lower bounds near 30, the search after the incumbent goes deep (541k iterations) instead of revisiting near-root siblings, which is where the baseline gained its last 661 bits. G015/G031 show the same 3113714.36 even with the bound inactive or at 60–120 s: this is the lazy-frontier value right after the root.
  - Adversarial oracle cases 29–32 (overlapping supports, multi-engine charge, zero-weight bucket, zero-duration sources); randomized brute-force native test `test.exe --relaxation-bound` (6000 graphs with COPY_TO groups, layered chains; G021: 52,552 weighted checks pass).
  - **H-G2 root primal heuristic** (`core/plan/primal_heuristic.hpp`, `SearchEngine::tryPlan`): extraction guided by region estimates, list scheduling, cache-consistency loop, validation by full propagation and normal leaf evaluation. G023: validated incumbent 121.17 at 25.5 s. G025–G027: bucket 16 (zero weight) left 74 cached classes unselected, disallowing 146 cache bases. G028–G029: covering extraction for zero-weight buckets: 39.33.
  - **H-G3 local search** (per weighted bucket, parallel; moves screened by max engine load, accepted by list-scheduled makespan): G031 39.33 → 37.02 in 0.1 s.
  - **G036 H-G3: normalized 0.2249**, first incumbent 25.79 s (39.33; 37.02 shortly after), bound 29.99, final gap 0.190, overshoot 0.012. **Rejected (volume gate):** eliminated bits 3113714.36 < 3114375.50. Oracle G033 32/32, gap unit tests G034 PASS, native G035 PASS (5 suites).
  - Next: H-G4 = H-G3 + largest-box-first node order once the root heuristic has an incumbent (the proof work refreshes the regions that dominate the frontier) + multi-start local search + root propagation timing log. Then 120 s diagnostic for bits, then evaluation.

### G037_build_HG4
- In progress: 2026-10-12T23:03:44.138380-04:00
- Metadata `logs/gap_improvements/G037_build_HG4.json`; log `logs/gap_improvements/G037_build_HG4.log`; patch `logs/gap_improvements/G037_build_HG4.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T23:04:42.694123-04:00; exit 0; runner elapsed 58.531 s.

### G038_diag_HG4_120s
- In progress: 2026-10-12T23:04:46.906997-04:00
- Metadata `logs/gap_improvements/G038_diag_HG4_120s.json`; log `logs/gap_improvements/G038_diag_HG4_120s.log`; patch `logs/gap_improvements/G038_diag_HG4_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-12T23:06:47.241254-04:00; exit 0; runner elapsed 120.297 s.
  `[SearchEngine volume] iterations=543 elapsed=105.4611537 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736976.785025093 log2-remaining-fraction=-3114536.3787152264 eliminated-bits=3114536.3787152264 remaining-nodes=544 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.234000000 budget=120.0`
  `[Program deadline] overshoot=0.127202`
  `[Program gap] primal-dual-integral=44.401944247013262 normalized=0.37001620205844382 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=25.918676614761353 samples=236`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791860712.9035916 program-deadline=1791860806.9849432`
- First incumbent: 25.918648481 s, cost 39.3272; best cost 39.3272.

### G039_build_HG4b
- In progress: 2026-10-12T23:07:27.687939-04:00
- Metadata `logs/gap_improvements/G039_build_HG4b.json`; log `logs/gap_improvements/G039_build_HG4b.log`; patch `logs/gap_improvements/G039_build_HG4b.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-12T23:08:26.100260-04:00; exit 0; runner elapsed 58.390 s.

### G040_oracle_HG4b
- In progress: 2026-10-12T23:08:26.277608-04:00
- Metadata `logs/gap_improvements/G040_oracle_HG4b.json`; log `logs/gap_improvements/G040_oracle_HG4b.log`; patch `logs/gap_improvements/G040_oracle_HG4b.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-12T23:08:29.154683-04:00; exit 0; runner elapsed 2.859 s.
- Oracle verdict: 32/32 cases passed.

### G041_gap_unit_tests_HG4b
- In progress: 2026-10-12T23:08:29.369865-04:00
- Metadata `logs/gap_improvements/G041_gap_unit_tests_HG4b.json`; log `logs/gap_improvements/G041_gap_unit_tests_HG4b.log`; patch `logs/gap_improvements/G041_gap_unit_tests_HG4b.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-12T23:08:29.440670-04:00; exit 0; runner elapsed 0.047 s.

### G042_native_HG4b_private_layout
- In progress: 2026-10-12T23:08:29.613409-04:00
- Metadata `logs/gap_improvements/G042_native_HG4b_private_layout.json`; log `logs/gap_improvements/G042_native_HG4b_private_layout.log`; patch `logs/gap_improvements/G042_native_HG4b_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-12T23:08:30.123663-04:00; exit 0; runner elapsed 0.500 s.

### G042_native_HG4b_write_after_read_cache
- In progress: 2026-10-12T23:08:30.298680-04:00
- Metadata `logs/gap_improvements/G042_native_HG4b_write_after_read_cache.json`; log `logs/gap_improvements/G042_native_HG4b_write_after_read_cache.log`; patch `logs/gap_improvements/G042_native_HG4b_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-12T23:08:30.368043-04:00; exit 0; runner elapsed 0.047 s.

### G042_native_HG4b_incremental_propagators
- In progress: 2026-10-12T23:08:30.535278-04:00
- Metadata `logs/gap_improvements/G042_native_HG4b_incremental_propagators.json`; log `logs/gap_improvements/G042_native_HG4b_incremental_propagators.log`; patch `logs/gap_improvements/G042_native_HG4b_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-12T23:08:30.616600-04:00; exit 0; runner elapsed 0.047 s.

### G042_native_HG4b_start_rank_window
- In progress: 2026-10-12T23:08:30.773377-04:00
- Metadata `logs/gap_improvements/G042_native_HG4b_start_rank_window.json`; log `logs/gap_improvements/G042_native_HG4b_start_rank_window.log`; patch `logs/gap_improvements/G042_native_HG4b_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-12T23:08:31.163309-04:00; exit 0; runner elapsed 0.359 s.

### G042_native_HG4b_relaxation_bound
- In progress: 2026-10-12T23:08:31.338829-04:00
- Metadata `logs/gap_improvements/G042_native_HG4b_relaxation_bound.json`; log `logs/gap_improvements/G042_native_HG4b_relaxation_bound.log`; patch `logs/gap_improvements/G042_native_HG4b_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-12T23:08:32.164893-04:00; exit 0; runner elapsed 0.797 s.

### G043_HG4b_decode
- In progress: 2026-10-12T23:08:35.582629-04:00
- Metadata `logs/gap_improvements/G043_HG4b_decode.json`; log `logs/gap_improvements/G043_HG4b_decode.log`; patch `logs/gap_improvements/G043_HG4b_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T23:18:35.841934-04:00; exit 0; runner elapsed 600.219 s.
  `[SearchEngine volume] iterations=2369 elapsed=585.7174685 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736838.9196596553 log2-remaining-fraction=-3114674.2440806637 eliminated-bits=3114674.2440806637 remaining-nodes=2370 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.140000000 budget=600.0`
  `[Program deadline] overshoot=0.0560868`
  `[Program gap] primal-dual-integral=135.10484123078405 normalized=0.22517473538464008 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=25.996814012527466 samples=1535`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791860941.6645105 program-deadline=1791861515.6677215`
- First incumbent: 25.996788979 s, cost 39.3272; best cost 37.020939.

### G044_HG4b_confirm1_decode
- In progress: 2026-10-12T23:18:57.485213-04:00
- Metadata `logs/gap_improvements/G044_HG4b_confirm1_decode.json`; log `logs/gap_improvements/G044_HG4b_confirm1_decode.log`; patch `logs/gap_improvements/G044_HG4b_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T23:28:57.802581-04:00; exit 0; runner elapsed 600.297 s.
  `[SearchEngine volume] iterations=2359 elapsed=585.78632730000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736839.3873760318 log2-remaining-fraction=-3114673.7763642874 eliminated-bits=3114673.7763642874 remaining-nodes=2360 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.218000000 budget=600.0`
  `[Program deadline] overshoot=0.0327597`
  `[Program gap] primal-dual-integral=135.04367665190313 normalized=0.22507279441983855 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=25.925038814544678 samples=1526`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791861563.4852955 program-deadline=1791862137.5602763`
- First incumbent: 25.925019264 s, cost 39.3272; best cost 37.020939.

### G045_HG4b_confirm2_decode
- In progress: 2026-10-12T23:29:06.158869-04:00
- Metadata `logs/gap_improvements/G045_HG4b_confirm2_decode.json`; log `logs/gap_improvements/G045_HG4b_confirm2_decode.log`; patch `logs/gap_improvements/G045_HG4b_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-12T23:39:06.544326-04:00; exit 0; runner elapsed 600.359 s.
  `[SearchEngine volume] iterations=2358 elapsed=585.27541940000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736839.4497701237 log2-remaining-fraction=-3114673.7139701955 eliminated-bits=3114673.7139701955 remaining-nodes=2359 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.282000000 budget=600.0`
  `[Program deadline] overshoot=0.178706`
  `[Program gap] primal-dual-integral=135.80461449581509 normalized=0.22634102415969182 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=26.864474296569824 samples=1526`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791862173.1011949 program-deadline=1791862746.2367716`
- First incumbent: 26.864423275 s, cost 39.3272; best cost 37.020939.
- Notes G037–G045 (2026-10-12):
  - **H-G4 = H-G3 + largest-box-first node order** once the root heuristic has an incumbent (`SearchNodeCompare::largest_box_first`; stored log2 volume, or depth when volumes are not measured) + multi-start local search from every initial weighting + root propagation timing log. G038 (120 s diagnostic): eliminated bits 3114536.38 after 543 iterations, the first time above the baseline value. Multi-start found nothing better than 37.02 and took 12 s before validating, so H-G4b validates each start's local optimum immediately and drops duplicate initial plans (G039 build; G040 oracle 32/32, G041 gap unit tests PASS, G042 native PASS).
  - **G043 H-G4b: normalized 0.22517**, bits 3114674.24, first incumbent 25.997 s (39.3272), deadline 37.020939, overshoot within 1 s, root propagation 10.56 s. **G044: 0.22507**, bits 3114673.78, first incumbent 25.925 s. **G045: 0.22634**, bits 3114673.71, first incumbent 26.864 s. Patches identical across the three runs.
  - **Gate verdict: ACCEPTED as optimization O-G1.** (1) correctness: oracle 32/32 with certified-bound validity, gap unit tests and all native suites pass on this build; (2) primary: median 0.22517 vs accepted 0.99165 (improvement 0.766, far beyond the baseline spread 2.7e-5); (3) volume: 3114674.24 / 3114673.78 / 3114673.71, each ≥ 3114375.50; (4) first incumbent: median 25.997 s ≤ 136.456 s, slowest 26.864 s ≤ 138.025 s; (5) costs: first-incumbent 39.3272 ≤ 75.0097, deadline 37.020939 ≤ 75.009705.
  - Soundness summary: the bound is a Lagrangian relaxation of max engine load over selections only (memory, order and cache budget dropped). Every inequality is argued in `docs/core/relaxation_bound.md`, checked by oracle cases 29–32 and by 6000 random brute-force graphs (`--relaxation-bound`), and margined for float accumulation. The heuristic and local search only propose plans: every incumbent comes from `tryPlan` (all propagators, all variables fixed, normal leaf evaluation). Largest-box-first only reorders the queue, so it neither prunes nor drops plans.
  - Note: a few multi-start attempts were rejected by `CostLowerBoundPropagator` because they only tied the incumbent (37.020943 vs 37.020939 after float rounding). This was harmless; H-G5 adds a relative tolerance.

### G046_build_HG6_interval_bound
- In progress: 2026-10-13T06:31:31.674136-04:00
- Metadata `logs/gap_improvements/G046_build_HG6_interval_bound.json`; log `logs/gap_improvements/G046_build_HG6_interval_bound.log`; patch `logs/gap_improvements/G046_build_HG6_interval_bound.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:32:30.296592-04:00; exit 0; runner elapsed 58.594 s.

### G047_native_HG6_relaxation_bound
- In progress: 2026-10-13T06:32:33.695312-04:00
- Metadata `logs/gap_improvements/G047_native_HG6_relaxation_bound.json`; log `logs/gap_improvements/G047_native_HG6_relaxation_bound.log`; patch `logs/gap_improvements/G047_native_HG6_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T06:32:34.542237-04:00; exit 0; runner elapsed 0.813 s.

### G048_diag_HG6_40s
- In progress: 2026-10-13T06:32:38.862480-04:00
- Metadata `logs/gap_improvements/G048_diag_HG6_40s.json`; log `logs/gap_improvements/G048_diag_HG6_40s.log`; patch `logs/gap_improvements/G048_diag_HG6_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-13T06:33:19.136406-04:00; exit 0; runner elapsed 40.235 s.
  `[SearchEngine volume] iterations=0 elapsed=25.938153400000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.172000000 budget=40.0`
  `[Program deadline] overshoot=0.0401258`
  `[Program gap] primal-dual-integral=28.401244126130106 normalized=0.71003110315325269 final-gap=0.18580222938159402 final-incumbent=37.020938999999998 final-bound=30.142365999999999 first-incumbent-elapsed=25.704895973205566 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791887584.6539755 program-deadline=1791887598.9491076`
- First incumbent: 25.704867840 s, cost 39.3272; best cost 37.020939.

### G049_build_diag_intervals
- In progress: 2026-10-13T06:33:45.644142-04:00
- Metadata `logs/gap_improvements/G049_build_diag_intervals.json`; log `logs/gap_improvements/G049_build_diag_intervals.log`; patch `logs/gap_improvements/G049_build_diag_intervals.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:34:42.802226-04:00; exit 0; runner elapsed 57.125 s.

### G050_diag_intervals_30s
- In progress: 2026-10-13T06:34:42.984478-04:00
- Metadata `logs/gap_improvements/G050_diag_intervals_30s.json`; log `logs/gap_improvements/G050_diag_intervals_30s.log`; patch `logs/gap_improvements/G050_diag_intervals_30s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T06:35:13.236851-04:00; exit 0; runner elapsed 30.203 s.
  `[SearchEngine volume] iterations=0 elapsed=15.963960999999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.141000000 budget=30.0`
  `[Program deadline] overshoot=0.0430272`
  `[Program gap] primal-dual-integral=26.329279780758256 normalized=0.87764265935860852 final-gap=0.18580222938159402 final-incumbent=37.020938999999998 final-bound=30.142365999999999 first-incumbent-elapsed=25.442290782928467 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791887708.5125761 program-deadline=1791887713.0703132`
- First incumbent: 25.442262888 s, cost 39.3272; best cost 37.020939.

### G051_build_HG6b_alt_window
- In progress: 2026-10-13T06:36:22.685182-04:00
- Metadata `logs/gap_improvements/G051_build_HG6b_alt_window.json`; log `logs/gap_improvements/G051_build_HG6b_alt_window.log`; patch `logs/gap_improvements/G051_build_HG6b_alt_window.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:37:21.535795-04:00; exit 0; runner elapsed 58.812 s.

### G052_native_HG6b_relaxation_bound
- In progress: 2026-10-13T06:37:21.721051-04:00
- Metadata `logs/gap_improvements/G052_native_HG6b_relaxation_bound.json`; log `logs/gap_improvements/G052_native_HG6b_relaxation_bound.log`; patch `logs/gap_improvements/G052_native_HG6b_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T06:37:23.307393-04:00; exit 0; runner elapsed 1.562 s.

### G053_diag_HG6b_intervals_30s
- In progress: 2026-10-13T06:37:28.079782-04:00
- Metadata `logs/gap_improvements/G053_diag_HG6b_intervals_30s.json`; log `logs/gap_improvements/G053_diag_HG6b_intervals_30s.log`; patch `logs/gap_improvements/G053_diag_HG6b_intervals_30s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T06:37:58.319374-04:00; exit 0; runner elapsed 30.203 s.
  `[SearchEngine volume] iterations=0 elapsed=15.809157900000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.140000000 budget=30.0`
  `[Program deadline] overshoot=0.0466588`
  `[Program gap] primal-dual-integral=27.252005162886217 normalized=0.90840017209620727 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=26.556961536407471 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791887874.7223539 program-deadline=1791887878.1654203`
- First incumbent: 26.556933641 s, cost 39.3272; best cost 37.020939.

### G054_build_HG7_parallel_cycle_reduction
- In progress: 2026-10-13T06:42:11.444327-04:00
- Metadata `logs/gap_improvements/G054_build_HG7_parallel_cycle_reduction.json`; log `logs/gap_improvements/G054_build_HG7_parallel_cycle_reduction.log`; patch `logs/gap_improvements/G054_build_HG7_parallel_cycle_reduction.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:43:09.762388-04:00; exit 0; runner elapsed 58.282 s.

### G055_build_diag_planall_timing
- In progress: 2026-10-13T06:44:09.617982-04:00
- Metadata `logs/gap_improvements/G055_build_diag_planall_timing.json`; log `logs/gap_improvements/G055_build_diag_planall_timing.log`; patch `logs/gap_improvements/G055_build_diag_planall_timing.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:45:08.300359-04:00; exit 0; runner elapsed 58.641 s.

### G056_build_HG7b_csr_cycles
- In progress: 2026-10-13T06:46:22.265090-04:00
- Metadata `logs/gap_improvements/G056_build_HG7b_csr_cycles.json`; log `logs/gap_improvements/G056_build_HG7b_csr_cycles.log`; patch `logs/gap_improvements/G056_build_HG7b_csr_cycles.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:47:20.438777-04:00; exit 0; runner elapsed 58.140 s.

### G057_build_diag_cycle_steps
- In progress: 2026-10-13T06:48:03.254174-04:00
- Metadata `logs/gap_improvements/G057_build_diag_cycle_steps.json`; log `logs/gap_improvements/G057_build_diag_cycle_steps.log`; patch `logs/gap_improvements/G057_build_diag_cycle_steps.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:49:01.970533-04:00; exit 0; runner elapsed 58.688 s.

### G058_build_HG7c_memo_nodiag
- In progress: 2026-10-13T06:51:15.318868-04:00
- Metadata `logs/gap_improvements/G058_build_HG7c_memo_nodiag.json`; log `logs/gap_improvements/G058_build_HG7c_memo_nodiag.log`; patch `logs/gap_improvements/G058_build_HG7c_memo_nodiag.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:52:13.323879-04:00; exit 0; runner elapsed 57.953 s.

### G059_build_profile_diag
- In progress: 2026-10-13T06:52:52.068326-04:00
- Metadata `logs/gap_improvements/G059_build_profile_diag.json`; log `logs/gap_improvements/G059_build_profile_diag.log`; patch `logs/gap_improvements/G059_build_profile_diag.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO --profile`
- Completed: 2026-10-13T06:54:18.712709-04:00; exit 0; runner elapsed 86.609 s.

### G060_build_profile_HG7d
- In progress: 2026-10-13T06:55:59.481892-04:00
- Metadata `logs/gap_improvements/G060_build_profile_HG7d.json`; log `logs/gap_improvements/G060_build_profile_HG7d.log`; patch `logs/gap_improvements/G060_build_profile_HG7d.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO --profile`
- Completed: 2026-10-13T06:57:25.626765-04:00; exit 0; runner elapsed 86.125 s.

### G061_build_HG7
- In progress: 2026-10-13T06:58:07.543200-04:00
- Metadata `logs/gap_improvements/G061_build_HG7.json`; log `logs/gap_improvements/G061_build_HG7.log`; patch `logs/gap_improvements/G061_build_HG7.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T06:59:05.809372-04:00; exit 0; runner elapsed 58.219 s.

### G062_oracle_HG7
- In progress: 2026-10-13T06:59:05.990637-04:00
- Metadata `logs/gap_improvements/G062_oracle_HG7.json`; log `logs/gap_improvements/G062_oracle_HG7.log`; patch `logs/gap_improvements/G062_oracle_HG7.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T06:59:09.294663-04:00; exit 0; runner elapsed 3.265 s.
- Oracle verdict: 32/32 cases passed.

### G063_gap_unit_tests_HG7
- In progress: 2026-10-13T06:59:09.461035-04:00
- Metadata `logs/gap_improvements/G063_gap_unit_tests_HG7.json`; log `logs/gap_improvements/G063_gap_unit_tests_HG7.log`; patch `logs/gap_improvements/G063_gap_unit_tests_HG7.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T06:59:09.549071-04:00; exit 0; runner elapsed 0.063 s.

### G064_native_HG7_private_layout
- In progress: 2026-10-13T06:59:09.718699-04:00
- Metadata `logs/gap_improvements/G064_native_HG7_private_layout.json`; log `logs/gap_improvements/G064_native_HG7_private_layout.log`; patch `logs/gap_improvements/G064_native_HG7_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T06:59:09.825357-04:00; exit 0; runner elapsed 0.062 s.

### G064_native_HG7_write_after_read_cache
- In progress: 2026-10-13T06:59:09.990863-04:00
- Metadata `logs/gap_improvements/G064_native_HG7_write_after_read_cache.json`; log `logs/gap_improvements/G064_native_HG7_write_after_read_cache.log`; patch `logs/gap_improvements/G064_native_HG7_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T06:59:10.433990-04:00; exit 0; runner elapsed 0.406 s.

### G064_native_HG7_incremental_propagators
- In progress: 2026-10-13T06:59:10.598188-04:00
- Metadata `logs/gap_improvements/G064_native_HG7_incremental_propagators.json`; log `logs/gap_improvements/G064_native_HG7_incremental_propagators.log`; patch `logs/gap_improvements/G064_native_HG7_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T06:59:10.686897-04:00; exit 0; runner elapsed 0.047 s.

### G064_native_HG7_start_rank_window
- In progress: 2026-10-13T06:59:10.845952-04:00
- Metadata `logs/gap_improvements/G064_native_HG7_start_rank_window.json`; log `logs/gap_improvements/G064_native_HG7_start_rank_window.log`; patch `logs/gap_improvements/G064_native_HG7_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T06:59:11.437011-04:00; exit 0; runner elapsed 0.547 s.

### G064_native_HG7_relaxation_bound
- In progress: 2026-10-13T06:59:11.606135-04:00
- Metadata `logs/gap_improvements/G064_native_HG7_relaxation_bound.json`; log `logs/gap_improvements/G064_native_HG7_relaxation_bound.log`; patch `logs/gap_improvements/G064_native_HG7_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T06:59:12.448453-04:00; exit 0; runner elapsed 0.796 s.

### G065_HG7_decode
- In progress: 2026-10-13T06:59:16.344264-04:00
- Metadata `logs/gap_improvements/G065_HG7_decode.json`; log `logs/gap_improvements/G065_HG7_decode.log`; patch `logs/gap_improvements/G065_HG7_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T07:09:16.705309-04:00; exit 0; runner elapsed 600.312 s.
  `[SearchEngine volume] iterations=2480 elapsed=591.54398660000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736834.2538924413 log2-remaining-fraction=-3114678.9098478779 eliminated-bits=3114678.9098478779 remaining-nodes=2481 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.250000000 budget=600.0`
  `[Program deadline] overshoot=0.12956`
  `[Program gap] primal-dual-integral=123.10388401723444 normalized=0.20517314002872405 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.18430495262146 samples=1614`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791889167.6140811 program-deadline=1791889756.4298038`
- First incumbent: 11.184277296 s, cost 39.3272; best cost 37.020939.

### G066_HG7_confirm1_decode
- In progress: 2026-10-13T07:09:23.516394-04:00
- Metadata `logs/gap_improvements/G066_HG7_confirm1_decode.json`; log `logs/gap_improvements/G066_HG7_confirm1_decode.log`; patch `logs/gap_improvements/G066_HG7_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T07:19:23.864920-04:00; exit 0; runner elapsed 600.297 s.
  `[SearchEngine volume] iterations=2493 elapsed=592.01359539999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736833.6652243675 log2-remaining-fraction=-3114679.4985159514 eliminated-bits=3114679.4985159514 remaining-nodes=2494 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.234000000 budget=600.0`
  `[Program deadline] overshoot=0.109403`
  `[Program gap] primal-dual-integral=122.7105868470256 normalized=0.20451764474504266 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=10.700037956237793 samples=1624`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791889774.3081472 program-deadline=1791890363.6081336`
- First incumbent: 10.700013638 s, cost 39.3272; best cost 37.020939.

### G067_HG7_confirm2_decode
- In progress: 2026-10-13T07:19:29.381439-04:00
- Metadata `logs/gap_improvements/G067_HG7_confirm2_decode.json`; log `logs/gap_improvements/G067_HG7_confirm2_decode.log`; patch `logs/gap_improvements/G067_HG7_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Notes G046–G066 (2026-10-13):
  - **H-G6 interval bound** (trial, not retained): chain of necessary groups where each link necessarily depends on the previous; t_i = earliest finish of an active member of G_i; classes that necessarily depend on G_{i-1} and lie in the earliest member's sub-DAG run inside [t_{i-1}, t_i], so each window is at least max(critical path, max_lambda weighted load) with lambda chosen per window. Sound: native brute force over selections AND dispatch orders (`--relaxation-bound`, 6000 graphs, G047/G052) passes. On Gemma (G048, G053): per-class weighting 30.17, per-alternative weighting (counts COPY_TO of the previous link) 24.39, vs load bound 30.01: no useful gain, +0.7 s root time. Saved as `logs/gap_improvements/HG6_interval_bound_trial.diff`. Diagnostics G050/G053: critical-path sum 24.39 equals the global CP; the LM-head window needs ≥ 10.87 (load), COPY_TO runs on the CPU engine (3207 copy alternatives).
  - **H-G7 startup speedup** (pure speedup; no bound, pruning or encoding change). Timeline diagnostics (scratch `timeline.py`, worker output time-stamped): per-bucket cycle reduction was serial (17 × 0.58 s), root propagation 10.5 s (profile build G059: CacheFreshnessPropagator 5.6 s, StaticPrivateLayoutPropagator 3.7 s). Changes: (1) planner per-bucket prune/cycle-reduction in parallel after a first bucket, with a planAll-scoped memo of `estimateCost` (pure function of its arguments; bypassed when a call observer is installed); (2) `egraph_cycles{0..3}.txt` cycle listings only with `TG_CYCLE_DIAGNOSTICS=1` (they cost ~1.7 s per bucket to format under contention); (3) `findCyclicComponents` uses flat CSR adjacency and iterative Tarjan (same components and edge order); (4) `CacheFreshnessPropagator::ensureIndex` in one pass over selection variables (was cached bases × buckets × classes = 236M lookups), same insertion order; (5) `StaticPrivateLayout` shared layout: per-bucket conditions, base conflict matrix and per-bucket plain placement in parallel, same placement results and fix order. Root propagation 10.5 s → 1.8 s; first incumbent 26 s → 11 s.
  - G061 build, G062 oracle 32/32, G063 gap unit tests PASS, G064 native (5 suites) PASS.
  - **G065 H-G7: normalized 0.20517**, first incumbent 11.184 s (39.3272), deadline 37.020939, bits 3114678.91. **G066: 0.20452**, first incumbent 10.700 s, bits 3114679.50. Patches G065/G066/G067 byte-identical (hash 87A18E7D…).
- Completed: 2026-10-13T07:29:29.754379-04:00; exit 0; runner elapsed 600.328 s.
  `[SearchEngine volume] iterations=2448 elapsed=591.85460939999996 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.6453113849 log2-remaining-fraction=-3114677.5184289343 eliminated-bits=3114677.5184289343 remaining-nodes=2449 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.266000000 budget=600.0`
  `[Program deadline] overshoot=0.12986`
  `[Program gap] primal-dual-integral=122.94306500685343 normalized=0.20490510834475573 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=10.985766410827637 samples=1591`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791890380.4548881 program-deadline=1791890969.469146`
- First incumbent: 10.985742092 s, cost 39.3272; best cost 37.020939.

- G067 H-G7 confirmation 2: normalized 0.20491, first incumbent 10.986 s, bits 3114677.52, costs 39.3272 / 37.020939.
- **Gate verdict H-G7: ACCEPTED as optimization O-G2** (2026-10-13). (1) correctness: oracle G062 32/32 with certified-bound validity, gap unit tests G063, native G064 (5 suites) pass; (2) primary: median 0.20491 (runs 0.20517 / 0.20452 / 0.20491) vs accepted 0.22517, improvement 0.020 > accepted spread 1.27e-3; (3) bits 3114678.91 / 3114679.50 / 3114677.52 ≥ 3114375.50 (and ≥ the O-G1 minimum 3114673.71); (4) first incumbent median 10.986 s ≤ 25.997 + 1, slowest 11.184 ≤ 26.864 + 2; (5) costs 39.3272 / 37.020939 unchanged. Accepted source: `logs/gap_improvements/G065_HG7_decode.patch` on HEAD 66e37b62.

### G068_build_diag_windows
- In progress: 2026-10-13T07:29:42.112039-04:00
- Metadata `logs/gap_improvements/G068_build_diag_windows.json`; log `logs/gap_improvements/G068_build_diag_windows.log`; patch `logs/gap_improvements/G068_build_diag_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:30:40.651221-04:00; exit 0; runner elapsed 58.500 s.

### G069_diag_windows_45s
- In progress: 2026-10-13T07:30:44.894954-04:00
- Metadata `logs/gap_improvements/G069_diag_windows_45s.json`; log `logs/gap_improvements/G069_diag_windows_45s.log`; patch `logs/gap_improvements/G069_diag_windows_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T07:31:30.626070-04:00; exit 0; runner elapsed 45.703 s.
  `[SearchEngine volume] iterations=202 elapsed=36.8311718 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737035.9621487404 log2-remaining-fraction=-3114477.2015915788 eliminated-bits=3114477.2015915788 remaining-nodes=203 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.641000000 budget=45.0`
  `[Program deadline] overshoot=0.0929921`
  `[Program gap] primal-dual-integral=17.948198737122148 normalized=0.39884886082493665 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.552812099456787 samples=73`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891056.5315013 program-deadline=1791891089.978723`
- First incumbent: 11.552778244 s, cost 39.3272; best cost 37.020939.

### G070_build_diag_nodes
- In progress: 2026-10-13T07:31:54.303073-04:00
- Metadata `logs/gap_improvements/G070_build_diag_nodes.json`; log `logs/gap_improvements/G070_build_diag_nodes.log`; patch `logs/gap_improvements/G070_build_diag_nodes.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:32:52.886784-04:00; exit 0; runner elapsed 58.547 s.

### G071_diag_nodes_30s
- In progress: 2026-10-13T07:32:53.063755-04:00
- Metadata `logs/gap_improvements/G071_diag_nodes_30s.json`; log `logs/gap_improvements/G071_diag_nodes_30s.log`; patch `logs/gap_improvements/G071_diag_nodes_30s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T07:33:23.343229-04:00; exit 0; runner elapsed 30.234 s.
  `[SearchEngine volume] iterations=76 elapsed=21.831543799999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737065.6694930592 log2-remaining-fraction=-3114447.49424726 eliminated-bits=3114447.49424726 remaining-nodes=77 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.187000000 budget=30.0`
  `[Program deadline] overshoot=0.00846481`
  `[Program gap] primal-dual-integral=15.144601147185794 normalized=0.50482003823952648 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.610090017318726 samples=24`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891184.7626188 program-deadline=1791891203.1525536`
- First incumbent: 11.610065222 s, cost 39.3272; best cost 37.020939.

### G072_build_diag_exact_windows
- In progress: 2026-10-13T07:34:50.473019-04:00
- Metadata `logs/gap_improvements/G072_build_diag_exact_windows.json`; log `logs/gap_improvements/G072_build_diag_exact_windows.log`; patch `logs/gap_improvements/G072_build_diag_exact_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:35:49.147904-04:00; exit 0; runner elapsed 58.641 s.

### G073_diag_exact_windows_90s
- In progress: 2026-10-13T07:35:49.342884-04:00
- Metadata `logs/gap_improvements/G073_diag_exact_windows_90s.json`; log `logs/gap_improvements/G073_diag_exact_windows_90s.log`; patch `logs/gap_improvements/G073_diag_exact_windows_90s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 90`
- Completed: 2026-10-13T07:37:19.708170-04:00; exit 0; runner elapsed 90.328 s.
  `[SearchEngine volume] iterations=473 elapsed=81.877920200000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736982.2880252821 log2-remaining-fraction=-3114530.8757150369 eliminated-bits=3114530.8757150369 remaining-nodes=474 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=90.266000000 budget=90.0`
  `[Program deadline] overshoot=0.148318`
  `[Program gap] primal-dual-integral=26.69052128833318 normalized=0.29656134764814646 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.789554595947266 samples=197`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891361.2201819 program-deadline=1791891439.4306633`
- First incumbent: 11.789518595 s, cost 39.3272; best cost 37.020939.

### G074_build_diag_kids
- In progress: 2026-10-13T07:38:23.305753-04:00
- Metadata `logs/gap_improvements/G074_build_diag_kids.json`; log `logs/gap_improvements/G074_build_diag_kids.log`; patch `logs/gap_improvements/G074_build_diag_kids.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:39:21.990410-04:00; exit 0; runner elapsed 58.657 s.

### G075_diag_kids_60s
- In progress: 2026-10-13T07:39:22.185894-04:00
- Metadata `logs/gap_improvements/G075_diag_kids_60s.json`; log `logs/gap_improvements/G075_diag_kids_60s.log`; patch `logs/gap_improvements/G075_diag_kids_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T07:40:22.772875-04:00; exit 0; runner elapsed 60.547 s.
  `[SearchEngine volume] iterations=319 elapsed=51.888754200000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736999.9407263806 log2-remaining-fraction=-3114513.2230139384 eliminated-bits=3114513.2230139384 remaining-nodes=320 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.484000000 budget=60.0`
  `[Program deadline] overshoot=0.0611455`
  `[Program gap] primal-dual-integral=20.759709289040416 normalized=0.34599515481734028 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.504251480102539 samples=118`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891573.7791576 program-deadline=1791891622.2749341`
- First incumbent: 11.504223585 s, cost 39.3272; best cost 37.020939.

### G076_build_diag_consumers
- In progress: 2026-10-13T07:40:43.271520-04:00
- Metadata `logs/gap_improvements/G076_build_diag_consumers.json`; log `logs/gap_improvements/G076_build_diag_consumers.log`; patch `logs/gap_improvements/G076_build_diag_consumers.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:41:41.981328-04:00; exit 0; runner elapsed 58.672 s.

### G077_diag_consumers_60s
- In progress: 2026-10-13T07:41:42.181430-04:00
- Metadata `logs/gap_improvements/G077_diag_consumers_60s.json`; log `logs/gap_improvements/G077_diag_consumers_60s.log`; patch `logs/gap_improvements/G077_diag_consumers_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T07:42:42.943408-04:00; exit 0; runner elapsed 60.735 s.
  `[SearchEngine volume] iterations=316 elapsed=51.831951599999996 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737000.3107386846 log2-remaining-fraction=-3114512.8530016346 eliminated-bits=3114512.8530016346 remaining-nodes=317 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.656000000 budget=60.0`
  `[Program deadline] overshoot=0.0377069`
  `[Program gap] primal-dual-integral=20.869511080314851 normalized=0.3478251846719142 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.640247583389282 samples=117`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891713.9093592 program-deadline=1791891762.2691412`
- First incumbent: 11.640218019 s, cost 39.3272; best cost 37.020939.

### G078_build_HG8_cache_moves
- In progress: 2026-10-13T07:44:25.500213-04:00
- Metadata `logs/gap_improvements/G078_build_HG8_cache_moves.json`; log `logs/gap_improvements/G078_build_HG8_cache_moves.log`; patch `logs/gap_improvements/G078_build_HG8_cache_moves.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:45:24.460642-04:00; exit 0; runner elapsed 58.937 s.

### G079_diag_HG8_90s
- In progress: 2026-10-13T07:45:24.664443-04:00
- Metadata `logs/gap_improvements/G079_diag_HG8_90s.json`; log `logs/gap_improvements/G079_diag_HG8_90s.log`; patch `logs/gap_improvements/G079_diag_HG8_90s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 90`
- Completed: 2026-10-13T07:46:55.213239-04:00; exit 0; runner elapsed 90.515 s.
  `[SearchEngine volume] iterations=471 elapsed=82.075563599999995 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736982.4964968283 log2-remaining-fraction=-3114530.6672434909 eliminated-bits=3114530.6672434909 remaining-nodes=472 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=90.453000000 budget=90.0`
  `[Program deadline] overshoot=0.317808`
  `[Program gap] primal-dual-integral=26.817429577535229 normalized=0.29797143975039142 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=11.946168184280396 samples=196`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791891936.6973684 program-deadline=1791892014.7512279`
- First incumbent: 11.946140528 s, cost 39.3272; best cost 37.020939.

### G080_build_diag_cacheset
- In progress: 2026-10-13T07:47:06.312096-04:00
- Metadata `logs/gap_improvements/G080_build_diag_cacheset.json`; log `logs/gap_improvements/G080_build_diag_cacheset.log`; patch `logs/gap_improvements/G080_build_diag_cacheset.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:48:04.968609-04:00; exit 0; runner elapsed 58.625 s.

### G081_diag_cacheset_40s
- In progress: 2026-10-13T07:48:05.156585-04:00
- Metadata `logs/gap_improvements/G081_diag_cacheset_40s.json`; log `logs/gap_improvements/G081_diag_cacheset_40s.log`; patch `logs/gap_improvements/G081_diag_cacheset_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-13T07:48:45.418694-04:00; exit 0; runner elapsed 40.235 s.
  `[SearchEngine volume] iterations=147 elapsed=31.567111400000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737048.6064667078 log2-remaining-fraction=-3114464.5572736114 eliminated-bits=3114464.5572736114 remaining-nodes=148 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.156000000 budget=40.0`
  `[Program deadline] overshoot=0.0278821`
  `[Program gap] primal-dual-integral=17.391941486727941 normalized=0.4347985371681985 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=12.03814697265625 samples=50`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892097.2819598 program-deadline=1791892125.2438359`
- First incumbent: 12.038123846 s, cost 39.3272; best cost 37.020939.

### G082_build_diag_cacheset_joint
- In progress: 2026-10-13T07:49:00.724361-04:00
- Metadata `logs/gap_improvements/G082_build_diag_cacheset_joint.json`; log `logs/gap_improvements/G082_build_diag_cacheset_joint.log`; patch `logs/gap_improvements/G082_build_diag_cacheset_joint.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:49:59.711059-04:00; exit 0; runner elapsed 58.953 s.

### G083_diag_cacheset_joint_40s
- In progress: 2026-10-13T07:49:59.911636-04:00
- Metadata `logs/gap_improvements/G083_diag_cacheset_joint_40s.json`; log `logs/gap_improvements/G083_diag_cacheset_joint_40s.log`; patch `logs/gap_improvements/G083_diag_cacheset_joint_40s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-13T07:50:40.599289-04:00; exit 0; runner elapsed 40.640 s.
  `[SearchEngine volume] iterations=154 elapsed=31.818806500000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737047.4401813427 log2-remaining-fraction=-3114465.7235589763 eliminated-bits=3114465.7235589763 remaining-nodes=155 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.593000000 budget=40.0`
  `[Program deadline] overshoot=0.116124`
  `[Program gap] primal-dual-integral=17.404205117027633 normalized=0.43510512792569084 final-gap=0.18824461114454313 final-incumbent=36.94059 final-bound=29.986723000000001 first-incumbent-elapsed=12.086161375045776 samples=53`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892212.0910203 program-deadline=1791892240.004885`
- First incumbent: 12.086135387 s, cost 39.3272; best cost 37.020939.

### G084_build_diag_coarsen
- In progress: 2026-10-13T07:51:21.365548-04:00
- Metadata `logs/gap_improvements/G084_build_diag_coarsen.json`; log `logs/gap_improvements/G084_build_diag_coarsen.log`; patch `logs/gap_improvements/G084_build_diag_coarsen.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:52:20.289748-04:00; exit 0; runner elapsed 58.891 s.

### G085_diag_coarsen_k1
- In progress: 2026-10-13T07:52:20.466689-04:00
- Metadata `logs/gap_improvements/G085_diag_coarsen_k1.json`; log `logs/gap_improvements/G085_diag_coarsen_k1.log`; patch `logs/gap_improvements/G085_diag_coarsen_k1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:52:45.897869-04:00; exit 0; runner elapsed 25.375 s.
  `[SearchEngine volume] iterations=5 elapsed=16.901250999999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737693.032051034 log2-remaining-fraction=-3113820.1316892849 eliminated-bits=3113820.1316892849 remaining-nodes=6 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.313000000 budget=25.0`
  `[Program deadline] overshoot=0.103827`
  `[Program gap] primal-dual-integral=14.221570237147036 normalized=0.56886280948588142 final-gap=0.18824461114454313 final-incumbent=36.94059 final-bound=29.986723000000001 first-incumbent-elapsed=11.645421743392944 samples=7`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892352.2102692 program-deadline=1791892365.564877`
- First incumbent: 11.645392179 s, cost 39.3272; best cost 37.020939.

### G085_diag_coarsen_k4
- In progress: 2026-10-13T07:52:46.090289-04:00
- Metadata `logs/gap_improvements/G085_diag_coarsen_k4.json`; log `logs/gap_improvements/G085_diag_coarsen_k4.log`; patch `logs/gap_improvements/G085_diag_coarsen_k4.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:53:11.650176-04:00; exit 0; runner elapsed 25.531 s.
  `[SearchEngine volume] iterations=4 elapsed=16.844254599999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737756.3030162384 log2-remaining-fraction=-3113756.8607240808 eliminated-bits=3113756.8607240808 remaining-nodes=5 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.438000000 budget=25.0`
  `[Program deadline] overshoot=0.0198317`
  `[Program gap] primal-dual-integral=13.798264601351356 normalized=0.55193058405405426 final-gap=0.18824461114454313 final-incumbent=36.94059 final-bound=29.986723000000001 first-incumbent-elapsed=11.121904373168945 samples=6`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892377.3074026 program-deadline=1791892391.1855268`
- First incumbent: 11.121875763 s, cost 39.3272; best cost 37.020939.

### G085_diag_coarsen_k16
- In progress: 2026-10-13T07:53:11.870031-04:00
- Metadata `logs/gap_improvements/G085_diag_coarsen_k16.json`; log `logs/gap_improvements/G085_diag_coarsen_k16.log`; patch `logs/gap_improvements/G085_diag_coarsen_k16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:53:38.252227-04:00; exit 1; runner elapsed 26.328 s.
  `[SearchEngine volume] iterations=0 elapsed=15.9264647 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=26.266000000 budget=25.0`
  `[Program deadline] overshoot=1.10981`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892405.8774068 program-deadline=1791892416.9753661`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 13.902040720 s, cost 39.3272; best cost 37.020939.

### G085_diag_coarsen_k78
- In progress: 2026-10-13T07:53:38.525703-04:00
- Metadata `logs/gap_improvements/G085_diag_coarsen_k78.json`; log `logs/gap_improvements/G085_diag_coarsen_k78.log`; patch `logs/gap_improvements/G085_diag_coarsen_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:54:04.690565-04:00; exit 0; runner elapsed 26.109 s.
  `[SearchEngine volume] iterations=0 elapsed=14.791511099999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=26.016000000 budget=25.0`
  `[Program deadline] overshoot=0.859359`
  `[Program gap] primal-dual-integral=16.860040085889672 normalized=0.67440160343558686 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=14.877837181091309 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892433.5324616 program-deadline=1791892443.6546628`
- First incumbent: 14.877798796 s, cost 39.3272; best cost 37.020939.

### G085_diag_coarsen_k200
- In progress: 2026-10-13T07:54:04.922292-04:00
- Metadata `logs/gap_improvements/G085_diag_coarsen_k200.json`; log `logs/gap_improvements/G085_diag_coarsen_k200.log`; patch `logs/gap_improvements/G085_diag_coarsen_k200.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:54:31.330904-04:00; exit 0; runner elapsed 26.360 s.
  `[SearchEngine volume] iterations=0 elapsed=16.0747736 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=26.281000000 budget=25.0`
  `[Program deadline] overshoot=0.810507`
  `[Program gap] primal-dual-integral=15.351764406774269 normalized=0.61407057627097072 final-gap=0.19000641771944243 final-incumbent=37.020938999999998 final-bound=29.986723000000001 first-incumbent-elapsed=13.029539346694946 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892458.0608563 program-deadline=1791892470.0313461`
- First incumbent: 13.029510260 s, cost 39.3272; best cost 37.020939.

### G086_build_diag_perclass
- In progress: 2026-10-13T07:54:53.140821-04:00
- Metadata `logs/gap_improvements/G086_build_diag_perclass.json`; log `logs/gap_improvements/G086_build_diag_perclass.log`; patch `logs/gap_improvements/G086_build_diag_perclass.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T07:55:52.116355-04:00; exit 0; runner elapsed 58.937 s.

### G087_diag_perclass_k1
- In progress: 2026-10-13T07:55:52.290543-04:00
- Metadata `logs/gap_improvements/G087_diag_perclass_k1.json`; log `logs/gap_improvements/G087_diag_perclass_k1.log`; patch `logs/gap_improvements/G087_diag_perclass_k1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:56:19.459521-04:00; exit 1; runner elapsed 27.110 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G087_diag_perclass_k4
- In progress: 2026-10-13T07:56:19.697682-04:00
- Metadata `logs/gap_improvements/G087_diag_perclass_k4.json`; log `logs/gap_improvements/G087_diag_perclass_k4.log`; patch `logs/gap_improvements/G087_diag_perclass_k4.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:56:47.294948-04:00; exit 1; runner elapsed 27.563 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G087_diag_perclass_k16
- In progress: 2026-10-13T07:56:47.522759-04:00
- Metadata `logs/gap_improvements/G087_diag_perclass_k16.json`; log `logs/gap_improvements/G087_diag_perclass_k16.log`; patch `logs/gap_improvements/G087_diag_perclass_k16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:57:14.669788-04:00; exit 1; runner elapsed 27.110 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G087_diag_perclass_k78
- In progress: 2026-10-13T07:57:14.897263-04:00
- Metadata `logs/gap_improvements/G087_diag_perclass_k78.json`; log `logs/gap_improvements/G087_diag_perclass_k78.log`; patch `logs/gap_improvements/G087_diag_perclass_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T07:57:42.040861-04:00; exit 1; runner elapsed 27.094 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G088_diag_exact_k78
- In progress: 2026-10-13T07:58:26.968513-04:00
- Metadata `logs/gap_improvements/G088_diag_exact_k78.json`; log `logs/gap_improvements/G088_diag_exact_k78.log`; patch `logs/gap_improvements/G088_diag_exact_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 240`
- Completed: 2026-10-13T08:02:27.392406-04:00; exit 0; runner elapsed 240.375 s.
  `[SearchEngine volume] iterations=1047 elapsed=231.7125456 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736918.4879746872 log2-remaining-fraction=-3114594.6757656317 eliminated-bits=3114594.6757656317 remaining-nodes=1048 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=240.313000000 budget=240.0`
  `[Program deadline] overshoot=0.0223601`
  `[Program gap] primal-dual-integral=62.894350833721056 normalized=0.26205979514050443 final-gap=0.17640224479359967 final-incumbent=36.94059 final-bound=30.424187 first-incumbent-elapsed=24.882855176925659 samples=530`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791892731.9471116 program-deadline=1791892947.0642872`
- First incumbent: 24.882824421 s, cost 39.3272; best cost 36.94059.

### G089_build_HG9_path_bound
- In progress: 2026-10-13T08:04:05.514644-04:00
- Metadata `logs/gap_improvements/G089_build_HG9_path_bound.json`; log `logs/gap_improvements/G089_build_HG9_path_bound.log`; patch `logs/gap_improvements/G089_build_HG9_path_bound.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:05:04.294861-04:00; exit 0; runner elapsed 58.750 s.

### G090_native_HG9_relaxation_bound
- In progress: 2026-10-13T08:05:04.484851-04:00
- Metadata `logs/gap_improvements/G090_native_HG9_relaxation_bound.json`; log `logs/gap_improvements/G090_native_HG9_relaxation_bound.log`; patch `logs/gap_improvements/G090_native_HG9_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T08:05:05.708552-04:00; exit 0; runner elapsed 1.188 s.

### G091_diag_HG9_k1
- In progress: 2026-10-13T08:05:10.292204-04:00
- Metadata `logs/gap_improvements/G091_diag_HG9_k1.json`; log `logs/gap_improvements/G091_diag_HG9_k1.log`; patch `logs/gap_improvements/G091_diag_HG9_k1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:06:12.404101-04:00; exit 1; runner elapsed 62.062 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G091_diag_HG9_k16
- In progress: 2026-10-13T08:06:12.624219-04:00
- Metadata `logs/gap_improvements/G091_diag_HG9_k16.json`; log `logs/gap_improvements/G091_diag_HG9_k16.log`; patch `logs/gap_improvements/G091_diag_HG9_k16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:07:15.072357-04:00; exit 1; runner elapsed 62.406 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G091_diag_HG9_k78
- In progress: 2026-10-13T08:07:15.275549-04:00
- Metadata `logs/gap_improvements/G091_diag_HG9_k78.json`; log `logs/gap_improvements/G091_diag_HG9_k78.log`; patch `logs/gap_improvements/G091_diag_HG9_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:08:15.908694-04:00; exit 0; runner elapsed 60.594 s.
  `[SearchEngine volume] iterations=162 elapsed=51.9569057 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737046.3396808477 log2-remaining-fraction=-3114466.8240594715 eliminated-bits=3114466.8240594715 remaining-nodes=163 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.531000000 budget=60.0`
  `[Program deadline] overshoot=0.192625`
  `[Program gap] primal-dual-integral=32.308624172828772 normalized=0.53847706954714625 final-gap=0.18824461114454313 final-incumbent=36.94059 final-bound=29.986723000000001 first-incumbent-elapsed=25.808400630950928 samples=66`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791893261.17752 program-deadline=1791893295.3691497`
- First incumbent: 25.808370352 s, cost 39.3272; best cost 37.020939.

### G092_diag_HG9_perclass_k78
- In progress: 2026-10-13T08:08:21.504278-04:00
- Metadata `logs/gap_improvements/G092_diag_HG9_perclass_k78.json`; log `logs/gap_improvements/G092_diag_HG9_perclass_k78.log`; patch `logs/gap_improvements/G092_diag_HG9_perclass_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 90`
- Completed: 2026-10-13T08:09:51.984607-04:00; exit 0; runner elapsed 90.437 s.
  `[SearchEngine volume] iterations=358 elapsed=80.359339500000004 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736996.8599753366 log2-remaining-fraction=-3114516.3037649826 eliminated-bits=3114516.3037649826 remaining-nodes=358 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=90.344000000 budget=90.0`
  `[Program deadline] overshoot=0.10853`
  `[Program gap] primal-dual-integral=37.230269030884841 normalized=0.41366965589872046 final-gap=0.15388438571230184 final-incumbent=36.94059 final-bound=31.25601 first-incumbent-elapsed=27.550338506698608 samples=144`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791893329.15922 program-deadline=1791893391.6089096`
- First incumbent: 27.550310373 s, cost 39.3272; best cost 36.94059.

### G092_diag_HG9_perclass_k16
- In progress: 2026-10-13T08:09:52.172862-04:00
- Metadata `logs/gap_improvements/G092_diag_HG9_perclass_k16.json`; log `logs/gap_improvements/G092_diag_HG9_perclass_k16.log`; patch `logs/gap_improvements/G092_diag_HG9_perclass_k16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 90`
- Completed: 2026-10-13T08:11:24.084099-04:00; exit 1; runner elapsed 91.875 s.
  `[SearchEngine volume] iterations=0 elapsed=83.338424399999994 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=91.812000000 budget=90.0`
  `[Program deadline] overshoot=1.59955`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791893478.9717512 program-deadline=1791893482.2641699`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 86.707581282 s, cost 39.3272; best cost 37.020939.

### G093_diag_HG9_windows_k78
- In progress: 2026-10-13T08:11:36.606545-04:00
- Metadata `logs/gap_improvements/G093_diag_HG9_windows_k78.json`; log `logs/gap_improvements/G093_diag_HG9_windows_k78.log`; patch `logs/gap_improvements/G093_diag_HG9_windows_k78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 240`
- Completed: 2026-10-13T08:15:36.935840-04:00; exit 0; runner elapsed 240.282 s.
  `[SearchEngine volume] iterations=1067 elapsed=231.43283410000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736917.4956761773 log2-remaining-fraction=-3114595.6680641416 eliminated-bits=3114595.6680641416 remaining-nodes=1067 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=240.203000000 budget=240.0`
  `[Program deadline] overshoot=0.0791051`
  `[Program gap] primal-dual-integral=58.246800835457798 normalized=0.24269500348107415 final-gap=0.15388438571230184 final-incumbent=36.94059 final-bound=31.25601 first-incumbent-elapsed=25.111372709274292 samples=545`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791893521.8190885 program-deadline=1791893736.7077394`
- First incumbent: 25.111349106 s, cost 39.3272; best cost 36.94059.

### G094_build_HG8b_cache_add
- In progress: 2026-10-13T08:16:43.949352-04:00
- Metadata `logs/gap_improvements/G094_build_HG8b_cache_add.json`; log `logs/gap_improvements/G094_build_HG8b_cache_add.log`; patch `logs/gap_improvements/G094_build_HG8b_cache_add.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:17:42.555218-04:00; exit 0; runner elapsed 58.579 s.

### G095_diag_HG8b_120s
- In progress: 2026-10-13T08:17:42.746957-04:00
- Metadata `logs/gap_improvements/G095_diag_HG8b_120s.json`; log `logs/gap_improvements/G095_diag_HG8b_120s.log`; patch `logs/gap_improvements/G095_diag_HG8b_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T08:19:44.879779-04:00; exit 1; runner elapsed 122.110 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G096_build_HG8b_nointerval
- In progress: 2026-10-13T08:19:55.826857-04:00
- Metadata `logs/gap_improvements/G096_build_HG8b_nointerval.json`; log `logs/gap_improvements/G096_build_HG8b_nointerval.log`; patch `logs/gap_improvements/G096_build_HG8b_nointerval.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:20:55.706973-04:00; exit 0; runner elapsed 59.828 s.

### G097_diag_HG8b_120s
- In progress: 2026-10-13T08:20:55.894375-04:00
- Metadata `logs/gap_improvements/G097_diag_HG8b_120s.json`; log `logs/gap_improvements/G097_diag_HG8b_120s.log`; patch `logs/gap_improvements/G097_diag_HG8b_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T08:22:58.008242-04:00; exit 1; runner elapsed 122.078 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G098_build_HG8b_fix
- In progress: 2026-10-13T08:23:13.776044-04:00
- Metadata `logs/gap_improvements/G098_build_HG8b_fix.json`; log `logs/gap_improvements/G098_build_HG8b_fix.log`; patch `logs/gap_improvements/G098_build_HG8b_fix.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:24:12.860992-04:00; exit 0; runner elapsed 59.046 s.

### G099_diag_HG8b_120s
- In progress: 2026-10-13T08:24:13.048939-04:00
- Metadata `logs/gap_improvements/G099_diag_HG8b_120s.json`; log `logs/gap_improvements/G099_diag_HG8b_120s.log`; patch `logs/gap_improvements/G099_diag_HG8b_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T08:26:13.506147-04:00; exit 0; runner elapsed 120.422 s.
  `[SearchEngine volume] iterations=597 elapsed=111.9933624 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736966.8698500602 log2-remaining-fraction=-3114546.2938902592 eliminated-bits=3114546.2938902592 remaining-nodes=598 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.343000000 budget=120.0`
  `[Program deadline] overshoot=0.228199`
  `[Program gap] primal-dual-integral=32.286258097679251 normalized=0.26905215081399375 final-gap=0.18824461114454313 final-incumbent=36.94059 final-bound=29.986723000000001 first-incumbent-elapsed=11.862759113311768 samples=267`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894265.0036693 program-deadline=1791894373.1409326`
- First incumbent: 11.862736702 s, cost 39.3272; best cost 37.020939.

### G100_native_HG9_perclass
- In progress: 2026-10-13T08:26:42.634656-04:00
- Metadata `logs/gap_improvements/G100_native_HG9_perclass.json`; log `logs/gap_improvements/G100_native_HG9_perclass.log`; patch `logs/gap_improvements/G100_native_HG9_perclass.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T08:26:43.929437-04:00; exit 0; runner elapsed 1.266 s.

### G101_native_HG9_perclass_k2
- In progress: 2026-10-13T08:26:44.120172-04:00
- Metadata `logs/gap_improvements/G101_native_HG9_perclass_k2.json`; log `logs/gap_improvements/G101_native_HG9_perclass_k2.log`; patch `logs/gap_improvements/G101_native_HG9_perclass_k2.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T08:26:44.965948-04:00; exit 0; runner elapsed 0.813 s.

### G102_build_HG9b_strides
- In progress: 2026-10-13T08:28:01.161637-04:00
- Metadata `logs/gap_improvements/G102_build_HG9b_strides.json`; log `logs/gap_improvements/G102_build_HG9b_strides.log`; patch `logs/gap_improvements/G102_build_HG9b_strides.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:28:59.868397-04:00; exit 0; runner elapsed 58.672 s.

### G103_native_HG9b_relaxation_bound
- In progress: 2026-10-13T08:29:00.060581-04:00
- Metadata `logs/gap_improvements/G103_native_HG9b_relaxation_bound.json`; log `logs/gap_improvements/G103_native_HG9b_relaxation_bound.log`; patch `logs/gap_improvements/G103_native_HG9b_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T08:29:01.142928-04:00; exit 0; runner elapsed 1.047 s.

### G104_diag_HG9b_stride32
- In progress: 2026-10-13T08:29:05.411308-04:00
- Metadata `logs/gap_improvements/G104_diag_HG9b_stride32.json`; log `logs/gap_improvements/G104_diag_HG9b_stride32.log`; patch `logs/gap_improvements/G104_diag_HG9b_stride32.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:30:05.999115-04:00; exit 0; runner elapsed 60.547 s.
  `[SearchEngine volume] iterations=306 elapsed=51.861977699999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737001.2898918546 log2-remaining-fraction=-3114511.8738484643 eliminated-bits=3114511.8738484643 remaining-nodes=306 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.469000000 budget=60.0`
  `[Program deadline] overshoot=0.0815663`
  `[Program gap] primal-dual-integral=19.07671950683055 normalized=0.31794532511384249 final-gap=0.16447614940638464 final-incumbent=36.94059 final-bound=30.864744000000002 first-incumbent-elapsed=10.93952465057373 samples=112`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894556.4394095 program-deadline=1791894605.499913`
- First incumbent: 10.939496517 s, cost 39.3272; best cost 36.94059.

### G104_diag_HG9b_stride64
- In progress: 2026-10-13T08:30:06.192143-04:00
- Metadata `logs/gap_improvements/G104_diag_HG9b_stride64.json`; log `logs/gap_improvements/G104_diag_HG9b_stride64.log`; patch `logs/gap_improvements/G104_diag_HG9b_stride64.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:31:06.692980-04:00; exit 0; runner elapsed 60.469 s.
  `[SearchEngine volume] iterations=302 elapsed=51.929555499999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737001.6922401744 log2-remaining-fraction=-3114511.4715001448 eliminated-bits=3114511.4715001448 remaining-nodes=302 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.406000000 budget=60.0`
  `[Program deadline] overshoot=0.0240221`
  `[Program gap] primal-dual-integral=18.718262248540025 normalized=0.31197103747566707 final-gap=0.15931935575474024 final-incumbent=36.94059 final-bound=31.055239 first-incumbent-elapsed=10.812852144241333 samples=111`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894617.0819333 program-deadline=1791894666.2691066`
- First incumbent: 10.812826633 s, cost 39.3272; best cost 36.94059.

### G104_diag_HG9b_stride128
- In progress: 2026-10-13T08:31:06.879461-04:00
- Metadata `logs/gap_improvements/G104_diag_HG9b_stride128.json`; log `logs/gap_improvements/G104_diag_HG9b_stride128.log`; patch `logs/gap_improvements/G104_diag_HG9b_stride128.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:32:07.152276-04:00; exit 0; runner elapsed 60.234 s.
  `[SearchEngine volume] iterations=279 elapsed=51.809154300000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737004.7820203726 log2-remaining-fraction=-3114508.3817199469 eliminated-bits=3114508.3817199469 remaining-nodes=279 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.172000000 budget=60.0`
  `[Program deadline] overshoot=0.00823617`
  `[Program gap] primal-dual-integral=18.821009444759358 normalized=0.31368349074598928 final-gap=0.15829733634465504 final-incumbent=36.94059 final-bound=31.092993 first-incumbent-elapsed=10.993772506713867 samples=104`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894677.9602511 program-deadline=1791894726.9665003`
- First incumbent: 10.993750811 s, cost 39.3272; best cost 36.94059.

### G104_diag_HG9b_stride256
- In progress: 2026-10-13T08:32:07.352818-04:00
- Metadata `logs/gap_improvements/G104_diag_HG9b_stride256.json`; log `logs/gap_improvements/G104_diag_HG9b_stride256.log`; patch `logs/gap_improvements/G104_diag_HG9b_stride256.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T08:33:07.711006-04:00; exit 0; runner elapsed 60.312 s.
  `[SearchEngine volume] iterations=264 elapsed=51.589677199999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737007.8645088901 log2-remaining-fraction=-3114505.2992314291 eliminated-bits=3114505.2992314291 remaining-nodes=264 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.250000000 budget=60.0`
  `[Program deadline] overshoot=0.0939589`
  `[Program gap] primal-dual-integral=19.183495116715751 normalized=0.3197249186119292 final-gap=0.15687510675925864 final-incumbent=36.94059 final-bound=31.145530999999998 first-incumbent-elapsed=11.506099939346313 samples=106`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894738.9595425 program-deadline=1791894787.4534731`
- First incumbent: 11.506069422 s, cost 39.3272; best cost 36.94059.

### G105_diag_HG9b_stride512
- In progress: 2026-10-13T08:33:16.346548-04:00
- Metadata `logs/gap_improvements/G105_diag_HG9b_stride512.json`; log `logs/gap_improvements/G105_diag_HG9b_stride512.log`; patch `logs/gap_improvements/G105_diag_HG9b_stride512.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-13T08:33:56.609106-04:00; exit 0; runner elapsed 40.218 s.
  `[SearchEngine volume] iterations=134 elapsed=31.669377600000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737051.7248791766 log2-remaining-fraction=-3114461.4388611428 eliminated-bits=3114461.4388611428 remaining-nodes=134 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.156000000 budget=40.0`
  `[Program deadline] overshoot=0.0155752`
  `[Program gap] primal-dual-integral=16.094776689410491 normalized=0.40236941723526226 final-gap=0.15398671217758025 final-incumbent=36.94059 final-bound=31.252230000000001 first-incumbent-elapsed=11.661862850189209 samples=45`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894808.1045063 program-deadline=1791894836.4426715`
- First incumbent: 11.661834717 s, cost 39.3272; best cost 36.94059.

### G105_diag_HG9b_stride1500
- In progress: 2026-10-13T08:33:56.842818-04:00
- Metadata `logs/gap_improvements/G105_diag_HG9b_stride1500.json`; log `logs/gap_improvements/G105_diag_HG9b_stride1500.log`; patch `logs/gap_improvements/G105_diag_HG9b_stride1500.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 40`
- Completed: 2026-10-13T08:34:37.211962-04:00; exit 0; runner elapsed 40.312 s.
  `[SearchEngine volume] iterations=30 elapsed=30.032345899999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737108.6716345337 log2-remaining-fraction=-3114404.4921057858 eliminated-bits=3114404.4921057858 remaining-nodes=30 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=40.219000000 budget=40.0`
  `[Program deadline] overshoot=0.0460329`
  `[Program gap] primal-dual-integral=20.107012691232281 normalized=0.50267531728080705 final-gap=0.15398671217758025 final-incumbent=36.94059 final-bound=31.252230000000001 first-incumbent-elapsed=16.384961605072021 samples=17`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791894853.3356049 program-deadline=1791894876.9506629`
- First incumbent: 16.384942055 s, cost 39.3272; best cost 36.94059.

### G106_build_HG9c
- In progress: 2026-10-13T08:36:22.912593-04:00
- Metadata `logs/gap_improvements/G106_build_HG9c.json`; log `logs/gap_improvements/G106_build_HG9c.log`; patch `logs/gap_improvements/G106_build_HG9c.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T08:37:22.063918-04:00; exit 0; runner elapsed 59.110 s.

### G107_oracle_HG9c
- In progress: 2026-10-13T08:37:22.256775-04:00
- Metadata `logs/gap_improvements/G107_oracle_HG9c.json`; log `logs/gap_improvements/G107_oracle_HG9c.log`; patch `logs/gap_improvements/G107_oracle_HG9c.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T08:37:25.399750-04:00; exit 0; runner elapsed 3.094 s.
- Oracle verdict: 34/34 cases passed.

### G108_gap_unit_tests_HG9c
- In progress: 2026-10-13T08:37:44.495091-04:00
- Metadata `logs/gap_improvements/G108_gap_unit_tests_HG9c.json`; log `logs/gap_improvements/G108_gap_unit_tests_HG9c.log`; patch `logs/gap_improvements/G108_gap_unit_tests_HG9c.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T08:37:44.578249-04:00; exit 0; runner elapsed 0.047 s.

### G109_native_HG9c_private_layout
- In progress: 2026-10-13T08:37:44.770238-04:00
- Metadata `logs/gap_improvements/G109_native_HG9c_private_layout.json`; log `logs/gap_improvements/G109_native_HG9c_private_layout.log`; patch `logs/gap_improvements/G109_native_HG9c_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T08:37:44.875629-04:00; exit 0; runner elapsed 0.078 s.

### G109_native_HG9c_write_after_read_cache
- In progress: 2026-10-13T08:37:45.044659-04:00
- Metadata `logs/gap_improvements/G109_native_HG9c_write_after_read_cache.json`; log `logs/gap_improvements/G109_native_HG9c_write_after_read_cache.log`; patch `logs/gap_improvements/G109_native_HG9c_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T08:37:45.126066-04:00; exit 0; runner elapsed 0.047 s.

### G109_native_HG9c_incremental_propagators
- In progress: 2026-10-13T08:37:45.301675-04:00
- Metadata `logs/gap_improvements/G109_native_HG9c_incremental_propagators.json`; log `logs/gap_improvements/G109_native_HG9c_incremental_propagators.log`; patch `logs/gap_improvements/G109_native_HG9c_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T08:37:45.734188-04:00; exit 0; runner elapsed 0.406 s.

### G109_native_HG9c_start_rank_window
- In progress: 2026-10-13T08:37:45.924653-04:00
- Metadata `logs/gap_improvements/G109_native_HG9c_start_rank_window.json`; log `logs/gap_improvements/G109_native_HG9c_start_rank_window.log`; patch `logs/gap_improvements/G109_native_HG9c_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T08:37:46.001174-04:00; exit 0; runner elapsed 0.047 s.

### G109_native_HG9c_relaxation_bound
- In progress: 2026-10-13T08:37:46.182376-04:00
- Metadata `logs/gap_improvements/G109_native_HG9c_relaxation_bound.json`; log `logs/gap_improvements/G109_native_HG9c_relaxation_bound.log`; patch `logs/gap_improvements/G109_native_HG9c_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T08:37:47.771816-04:00; exit 0; runner elapsed 1.547 s.

### G110_diag_HG9c_45s
- In progress: 2026-10-13T08:37:47.940607-04:00
- Metadata `logs/gap_improvements/G110_diag_HG9c_45s.json`; log `logs/gap_improvements/G110_diag_HG9c_45s.log`; patch `logs/gap_improvements/G110_diag_HG9c_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T08:38:33.221217-04:00; exit 0; runner elapsed 45.250 s.
  `[SearchEngine volume] iterations=195 elapsed=36.789902900000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737038.8642374065 log2-remaining-fraction=-3114474.2995029129 eliminated-bits=3114474.2995029129 remaining-nodes=195 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.172000000 budget=45.0`
  `[Program deadline] overshoot=0.0440743`
  `[Program gap] primal-dual-integral=16.787177685469668 normalized=0.37304839301043707 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=11.568198680877686 samples=70`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791895079.5977046 program-deadline=1791895113.0295341`
- First incumbent: 11.568170547 s, cost 39.3272; best cost 36.94059.

### G111_HG9_decode
- In progress: 2026-10-13T08:38:39.121143-04:00
- Metadata `logs/gap_improvements/G111_HG9_decode.json`; log `logs/gap_improvements/G111_HG9_decode.log`; patch `logs/gap_improvements/G111_HG9_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T08:48:39.427696-04:00; exit 0; runner elapsed 600.266 s.
  `[SearchEngine volume] iterations=2446 elapsed=591.73296249999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.7578006173 log2-remaining-fraction=-3114677.4059397019 eliminated-bits=3114677.4059397019 remaining-nodes=2446 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.203000000 budget=600.0`
  `[Program deadline] overshoot=0.0890749`
  `[Program gap] primal-dual-integral=102.45054727766725 normalized=0.17075091212944543 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=11.693721532821655 samples=1577`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791895130.9005404 program-deadline=1791895719.206847`
- First incumbent: 11.693693399 s, cost 39.3272; best cost 36.94059.

### G112_HG9_confirm1_decode
- In progress: 2026-10-13T08:48:47.434725-04:00
- Metadata `logs/gap_improvements/G112_HG9_confirm1_decode.json`; log `logs/gap_improvements/G112_HG9_confirm1_decode.log`; patch `logs/gap_improvements/G112_HG9_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T08:58:47.804292-04:00; exit 0; runner elapsed 600.328 s.
  `[SearchEngine volume] iterations=2433 elapsed=591.62155819999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736836.3601583219 log2-remaining-fraction=-3114676.8035819973 eliminated-bits=3114676.8035819973 remaining-nodes=2433 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.265000000 budget=600.0`
  `[Program deadline] overshoot=0.120534`
  `[Program gap] primal-dual-integral=102.69001092078933 normalized=0.17115001820131556 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=11.974072217941284 samples=1568`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791895739.4995794 program-deadline=1791896327.5255299`
- First incumbent: 11.974049568 s, cost 39.3272; best cost 36.94059.

### G113_HG9_confirm2_decode
- In progress: 2026-10-13T08:58:47.989157-04:00
- Metadata `logs/gap_improvements/G113_HG9_confirm2_decode.json`; log `logs/gap_improvements/G113_HG9_confirm2_decode.log`; patch `logs/gap_improvements/G113_HG9_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Notes G068–G112 (2026-10-13):
  - **Window diagnostics** (G068–G077, temporary instrumentation, removed): the incumbent (37.02) keeps the CPU busy for the whole makespan (GPU busy 5.6). Exact enumeration of each fine window (min over window selections of max(critical path, engine loads)) sums to 32.05 (G073), so fine windows cap any window bound near 32. The LM-head window: CPU whole-head kernel 12.75; split chunks cost 4.56 (CPU) / 4.14 (GPU) per sixteenth, so 12.75 is essentially optimal there and the lambda relaxation (≈10.7) is what is weak. Per-layer windows lose work that is not "necessarily after" the previous link (one alternative avoids it), e.g. a 0.155 class per layer.
  - **H-G8 cache-set moves** (`improveCacheSet`): local search records cached bases whose required selection blocks an otherwise load-reducing move; dropping the 7 jointly blocking bases (tried together first, then singly), re-running local search with them releasable, and checking that no weighted bucket still uses their cache-only alternatives gives 37.0209 → 36.9406 (G083). A cache-add counterpart (G094–G099: 72 candidate bases tried singly) found nothing and was removed.
  - **H-G9 path-weighted interval bound**: per-class window weighting (a class counts in window i if it necessarily depends on link i-1); per window, max over lambda (golden section) and mu ∈ {0, .2, .4, .6, .8, .9} of mu·(a dependency chain) + (1-mu)·(lambda-weighted load), computed by `evaluateRegionPath` (the chain continues through one anchor's P or through any child's critical path). Any subsequence of the interval chain is a valid chain, so windows use strides chain/4, chain/16, chain/64 (=16 windows stop), max taken. Gemma: stride 32 → 30.86, 64 → 31.06, 256 → 31.15, 354 → 31.27 (31.246 stored after the float margin), vs load bound 29.99. Root cost +0.45 s. Brute force (`--relaxation-bound`, strides 1/2/3 on 6000 graphs, selections × all dispatch orders) passes (G103); oracle cases 33 (parallel work outside the window) and 34 (sequential chain with engine choices; load bound 2.5, chain term 5) are tight: root bound = optimum = 5 (G107: 34/34).
  - G106 build, G107 oracle 34/34, G108 gap unit tests PASS, G109 native 5 suites PASS.
  - **G111 H-G9 (+H-G8): normalized 0.17075**, bound 31.246, incumbent 36.94059 (36.94 found ~23 s), first incumbent 11.694 s (39.3272), bits 3114677.41. **G112: 0.17115**, first incumbent 11.974 s, bits 3114676.80. Patches G111/G112/G113 identical (hash 191FF84F…).
- Completed: 2026-10-13T09:08:48.343881-04:00; exit 0; runner elapsed 600.312 s.
  `[SearchEngine volume] iterations=2454 elapsed=592.04551670000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.4030054226 log2-remaining-fraction=-3114677.7607348966 eliminated-bits=3114677.7607348966 remaining-nodes=2454 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.250000000 budget=600.0`
  `[Program deadline] overshoot=0.129374`
  `[Program gap] primal-dual-integral=102.16809260216625 normalized=0.17028015433694377 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=11.360461473464966 samples=1583`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791896339.436204 program-deadline=1791896928.0757682`
- First incumbent: 11.360435724 s, cost 39.3272; best cost 36.94059.

- G113 H-G9 confirmation 2: normalized 0.17028, first incumbent 11.360 s, bits 3114677.76.
- **Gate verdict H-G9 (+H-G8): ACCEPTED as optimization O-G3** (2026-10-13). (1) oracle G107 34/34 with certified-bound validity (new cases 33, 34), gap unit tests G108, native G109 (5 suites, brute force over selections and dispatch orders for strides 1–3) pass; (2) median 0.17075 (0.17075 / 0.17115 / 0.17028) vs O-G2 0.20491, improvement 0.034 » spread 6.5e-4; (3) bits 3114677.41 / 3114676.80 / 3114677.76 ≥ 3114375.50; (4) first incumbent median 11.694 s ≤ 10.986 + 1, slowest 11.974 ≤ 11.184 + 2; (5) first-incumbent cost 39.3272 = before, deadline cost 36.94059 < 37.020939. Accepted source: `logs/gap_improvements/G111_HG9_decode.patch` on HEAD 66e37b62, plus the doc sections "Interval bound" and step 7 in `docs/core/relaxation_bound.md` (written after G113 started).

### G114_build_diag_export_windows
- In progress: 2026-10-13T09:09:03.337721-04:00
- Metadata `logs/gap_improvements/G114_build_diag_export_windows.json`; log `logs/gap_improvements/G114_build_diag_export_windows.log`; patch `logs/gap_improvements/G114_build_diag_export_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:10:02.745802-04:00; exit 0; runner elapsed 59.375 s.

### G115_diag_export_windows
- In progress: 2026-10-13T09:10:02.923524-04:00
- Metadata `logs/gap_improvements/G115_diag_export_windows.json`; log `logs/gap_improvements/G115_diag_export_windows.log`; patch `logs/gap_improvements/G115_diag_export_windows.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T09:10:33.416965-04:00; exit 0; runner elapsed 30.468 s.
  `[SearchEngine volume] iterations=63 elapsed=21.464333700000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737067.1918972994 log2-remaining-fraction=-3114445.9718430201 eliminated-bits=3114445.9718430201 remaining-nodes=63 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.375000000 budget=30.0`
  `[Program deadline] overshoot=0.097044`
  `[Program gap] primal-dual-integral=14.910612478157747 normalized=0.49702041593859153 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=12.083359718322754 samples=20`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897015.0939822 program-deadline=1791897033.0106518`
- First incumbent: 12.083330393 s, cost 39.3272; best cost 36.94059.

### G116_build_HG10_exact_windows
- In progress: 2026-10-13T09:13:31.099694-04:00
- Metadata `logs/gap_improvements/G116_build_HG10_exact_windows.json`; log `logs/gap_improvements/G116_build_HG10_exact_windows.log`; patch `logs/gap_improvements/G116_build_HG10_exact_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:13:38.704091-04:00; exit 1; runner elapsed 7.563 s.

### G117_native_HG10_relaxation_bound
- In progress: 2026-10-13T09:13:38.894414-04:00
- Metadata `logs/gap_improvements/G117_native_HG10_relaxation_bound.json`; log `logs/gap_improvements/G117_native_HG10_relaxation_bound.log`; patch `logs/gap_improvements/G117_native_HG10_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T09:13:39.862503-04:00; exit 0; runner elapsed 0.938 s.

### G118_build_HG10_exact_windows
- In progress: 2026-10-13T09:13:57.722409-04:00
- Metadata `logs/gap_improvements/G118_build_HG10_exact_windows.json`; log `logs/gap_improvements/G118_build_HG10_exact_windows.log`; patch `logs/gap_improvements/G118_build_HG10_exact_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:14:56.495412-04:00; exit 0; runner elapsed 58.735 s.

### G119_native_HG10_relaxation_bound
- In progress: 2026-10-13T09:14:56.702179-04:00
- Metadata `logs/gap_improvements/G119_native_HG10_relaxation_bound.json`; log `logs/gap_improvements/G119_native_HG10_relaxation_bound.log`; patch `logs/gap_improvements/G119_native_HG10_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T09:14:57.691107-04:00; exit 0; runner elapsed 0.953 s.

### G120_diag_HG10_45s
- In progress: 2026-10-13T09:15:03.698760-04:00
- Metadata `logs/gap_improvements/G120_diag_HG10_45s.json`; log `logs/gap_improvements/G120_diag_HG10_45s.log`; patch `logs/gap_improvements/G120_diag_HG10_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T09:15:49.050350-04:00; exit 0; runner elapsed 45.312 s.
  `[SearchEngine volume] iterations=129 elapsed=36.856862499999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737054.8319239342 log2-remaining-fraction=-3114458.331816385 eliminated-bits=3114458.331816385 remaining-nodes=129 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.219000000 budget=45.0`
  `[Program deadline] overshoot=0.0936997`
  `[Program gap] primal-dual-integral=22.198216969632377 normalized=0.49329371043627501 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=19.207622766494751 samples=44`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897323.0014646 program-deadline=1791897348.7938714`
- First incumbent: 19.207593203 s, cost 39.3272; best cost 36.94059.

### G121_build_diag_budget
- In progress: 2026-10-13T09:15:59.484057-04:00
- Metadata `logs/gap_improvements/G121_build_diag_budget.json`; log `logs/gap_improvements/G121_build_diag_budget.log`; patch `logs/gap_improvements/G121_build_diag_budget.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:16:59.251856-04:00; exit 0; runner elapsed 59.734 s.

### G122_diag_budget_3000
- In progress: 2026-10-13T09:16:59.443679-04:00
- Metadata `logs/gap_improvements/G122_diag_budget_3000.json`; log `logs/gap_improvements/G122_diag_budget_3000.log`; patch `logs/gap_improvements/G122_diag_budget_3000.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T09:17:29.716341-04:00; exit 0; runner elapsed 30.234 s.
  `[SearchEngine volume] iterations=55 elapsed=21.827615099999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737069.0365720564 log2-remaining-fraction=-3114444.1271682628 eliminated-bits=3114444.1271682628 remaining-nodes=55 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.156000000 budget=30.0`
  `[Program deadline] overshoot=0.0375953`
  `[Program gap] primal-dual-integral=14.512999189780732 normalized=0.48376663965935773 final-gap=0.15415536135183552 final-incumbent=36.94059 final-bound=31.245999999999999 first-incumbent-elapsed=11.611466646194458 samples=20`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897431.1435385 program-deadline=1791897449.5320954`
- First incumbent: 11.611443043 s, cost 39.3272; best cost 36.94059.

### G122_diag_budget_20000
- In progress: 2026-10-13T09:17:29.905803-04:00
- Metadata `logs/gap_improvements/G122_diag_budget_20000.json`; log `logs/gap_improvements/G122_diag_budget_20000.log`; patch `logs/gap_improvements/G122_diag_budget_20000.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T09:18:00.684993-04:00; exit 0; runner elapsed 30.750 s.
  `[SearchEngine volume] iterations=13 elapsed=21.0942097 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737497.5583903892 log2-remaining-fraction=-3114015.60534993 eliminated-bits=3114015.60534993 remaining-nodes=13 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.672000000 budget=30.0`
  `[Program deadline] overshoot=0.0344884`
  `[Program gap] primal-dual-integral=14.873358587546056 normalized=0.49577861958486852 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=12.853911876678467 samples=10`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897462.8382959 program-deadline=1791897479.9844134`
- First incumbent: 12.853882551 s, cost 39.3272; best cost 36.94059.

### G122_diag_budget_100000
- In progress: 2026-10-13T09:18:00.991195-04:00
- Metadata `logs/gap_improvements/G122_diag_budget_100000.json`; log `logs/gap_improvements/G122_diag_budget_100000.log`; patch `logs/gap_improvements/G122_diag_budget_100000.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T09:18:31.388526-04:00; exit 0; runner elapsed 30.344 s.
  `[SearchEngine volume] iterations=0 elapsed=17.318100999999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.235000000 budget=30.0`
  `[Program deadline] overshoot=0.0433636`
  `[Program gap] primal-dual-integral=20.67554193742772 normalized=0.68918473124759072 final-gap=0.11518681360297214 final-incumbent=37.020938999999998 final-bound=32.756614999999996 first-incumbent-elapsed=19.386396884918213 samples=4`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897500.5206473 program-deadline=1791897511.1342847`
- First incumbent: 19.386362553 s, cost 39.3272; best cost 37.020939.

### G123_build_HG10
- In progress: 2026-10-13T09:18:40.943889-04:00
- Metadata `logs/gap_improvements/G123_build_HG10.json`; log `logs/gap_improvements/G123_build_HG10.log`; patch `logs/gap_improvements/G123_build_HG10.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:19:41.293429-04:00; exit 0; runner elapsed 60.297 s.

### G124_oracle_HG10
- In progress: 2026-10-13T09:19:41.488066-04:00
- Metadata `logs/gap_improvements/G124_oracle_HG10.json`; log `logs/gap_improvements/G124_oracle_HG10.log`; patch `logs/gap_improvements/G124_oracle_HG10.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T09:19:44.620358-04:00; exit 0; runner elapsed 3.094 s.
- Oracle verdict: 34/34 cases passed.

### G125_gap_unit_tests_HG10
- In progress: 2026-10-13T09:19:44.800948-04:00
- Metadata `logs/gap_improvements/G125_gap_unit_tests_HG10.json`; log `logs/gap_improvements/G125_gap_unit_tests_HG10.log`; patch `logs/gap_improvements/G125_gap_unit_tests_HG10.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T09:19:44.881242-04:00; exit 0; runner elapsed 0.046 s.

### G126_native_HG10_private_layout
- In progress: 2026-10-13T09:19:45.051987-04:00
- Metadata `logs/gap_improvements/G126_native_HG10_private_layout.json`; log `logs/gap_improvements/G126_native_HG10_private_layout.log`; patch `logs/gap_improvements/G126_native_HG10_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T09:19:45.155395-04:00; exit 0; runner elapsed 0.078 s.

### G126_native_HG10_write_after_read_cache
- In progress: 2026-10-13T09:19:45.325825-04:00
- Metadata `logs/gap_improvements/G126_native_HG10_write_after_read_cache.json`; log `logs/gap_improvements/G126_native_HG10_write_after_read_cache.log`; patch `logs/gap_improvements/G126_native_HG10_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T09:19:45.706320-04:00; exit 0; runner elapsed 0.344 s.

### G126_native_HG10_incremental_propagators
- In progress: 2026-10-13T09:19:45.879890-04:00
- Metadata `logs/gap_improvements/G126_native_HG10_incremental_propagators.json`; log `logs/gap_improvements/G126_native_HG10_incremental_propagators.log`; patch `logs/gap_improvements/G126_native_HG10_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T09:19:45.964114-04:00; exit 0; runner elapsed 0.047 s.

### G126_native_HG10_start_rank_window
- In progress: 2026-10-13T09:19:46.131762-04:00
- Metadata `logs/gap_improvements/G126_native_HG10_start_rank_window.json`; log `logs/gap_improvements/G126_native_HG10_start_rank_window.log`; patch `logs/gap_improvements/G126_native_HG10_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T09:19:46.206862-04:00; exit 0; runner elapsed 0.047 s.

### G126_native_HG10_relaxation_bound
- In progress: 2026-10-13T09:19:46.372870-04:00
- Metadata `logs/gap_improvements/G126_native_HG10_relaxation_bound.json`; log `logs/gap_improvements/G126_native_HG10_relaxation_bound.log`; patch `logs/gap_improvements/G126_native_HG10_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T09:19:47.774741-04:00; exit 0; runner elapsed 1.359 s.

### G127_oracle_HG10_cases
- In progress: 2026-10-13T09:20:05.884892-04:00
- Metadata `logs/gap_improvements/G127_oracle_HG10_cases.json`; log `logs/gap_improvements/G127_oracle_HG10_cases.log`; patch `logs/gap_improvements/G127_oracle_HG10_cases.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T09:20:09.227510-04:00; exit 0; runner elapsed 3.297 s.
- Oracle verdict: 36/36 cases passed.

### G128_HG10_decode
- In progress: 2026-10-13T09:20:12.939177-04:00
- Metadata `logs/gap_improvements/G128_HG10_decode.json`; log `logs/gap_improvements/G128_HG10_decode.log`; patch `logs/gap_improvements/G128_HG10_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T09:30:13.718023-04:00; exit 0; runner elapsed 600.735 s.
  `[SearchEngine volume] iterations=2461 elapsed=591.39476290000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.0751441552 log2-remaining-fraction=-3114678.0885961642 eliminated-bits=3114678.0885961642 remaining-nodes=2461 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.672000000 budget=600.0`
  `[Program deadline] overshoot=0.103785`
  `[Program gap] primal-dual-integral=78.98823498901703 normalized=0.13164705831502838 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=12.361882448196411 samples=1588`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791897625.3875101 program-deadline=1791898213.0256553`
- First incumbent: 12.361854792 s, cost 39.3272; best cost 36.94059.

### G129_HG10_confirm1_decode
- In progress: 2026-10-13T09:30:39.678084-04:00
- Metadata `logs/gap_improvements/G129_HG10_confirm1_decode.json`; log `logs/gap_improvements/G129_HG10_confirm1_decode.log`; patch `logs/gap_improvements/G129_HG10_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T09:40:39.961001-04:00; exit 0; runner elapsed 600.250 s.
  `[SearchEngine volume] iterations=2387 elapsed=591.58797660000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736838.1107643454 log2-remaining-fraction=-3114675.052975974 eliminated-bits=3114675.052975974 remaining-nodes=2387 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.188000000 budget=600.0`
  `[Program deadline] overshoot=0.0581493`
  `[Program gap] primal-dual-integral=78.842137885941057 normalized=0.13140356314323509 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=12.195564031600952 samples=1542`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791898251.9613695 program-deadline=1791898839.7658358`
- First incumbent: 12.195533752 s, cost 39.3272; best cost 36.94059.

### G130_HG10_confirm2_decode
- In progress: 2026-10-13T09:40:40.146761-04:00
- Metadata `logs/gap_improvements/G130_HG10_confirm2_decode.json`; log `logs/gap_improvements/G130_HG10_confirm2_decode.log`; patch `logs/gap_improvements/G130_HG10_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T09:50:40.747004-04:00; exit 0; runner elapsed 600.563 s.
  `[SearchEngine volume] iterations=2465 elapsed=592.2494269 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736834.9188085177 log2-remaining-fraction=-3114678.2449318012 eliminated-bits=3114678.2449318012 remaining-nodes=2465 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.500000000 budget=600.0`
  `[Program deadline] overshoot=0.390558`
  `[Program gap] primal-dual-integral=78.449186189414775 normalized=0.13074864364902464 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=11.755465030670166 samples=1593`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791898851.9878857 program-deadline=1791899440.2324557`
- First incumbent: 11.755429983 s, cost 39.3272; best cost 36.94059.

- Notes G114–G130 (2026-10-13):
  - G114/G115 (diagnostic, removed): exported interval windows; offline CP-SAT (`scratchpad/window_cpsat.py`, research only): LM-head window exact optimum 12.747 (proved), a layer window best schedule 1.303 (incumbent 1.32, so the plan is near-optimal per layer); exact layer relaxation min over selections of max(critical path, loads) = 1.2747 (proved) vs our window bound ≈1.11.
  - **H-G10 exact window value**: per interval window, a depth-first enumeration of the selections below a closing-link member computes min of max(critical path, engine loads) (shared classes once; prune when a load reaches the best value), with a budget of 20000 search nodes (0 if exhausted). Valid: every plan's window is at least its chain and each engine's load. Gemma: LM-head window 11.24 → 12.75, bound 31.246 → 32.757; root +0.4 s (budget sweep G122: 3000 → no gain, 20000 → full gain in 1.18 s, 100000 → same, 2.8 s). Oracle cases 35 (shared class counted once; double counting claims 8 > 5) and 36 (discrete whole-vs-split choice: relaxation ≈1, exact 4 = optimum), both tight. G118 build, G119 brute force PASS, G124/G127 oracle 36/36, G125 gap unit tests, G126 native PASS.
  - **G128 H-G10: 0.13165** (first incumbent 12.362 s, bits 3114678.09); **G129: 0.13140** (12.196 s, 3114675.05); **G130: 0.13075** (11.755 s, 3114678.24). Patches identical (hash 39BADBB5…).
- **Gate verdict H-G10: ACCEPTED as optimization O-G4** (2026-10-13): oracle 36/36 with certified-bound validity, gap unit tests and native suites pass; median 0.13140 vs 0.17075 (spread of O-G3 runs 8.7e-4); bits ≥ 3114375.50 in every run; first incumbent median 12.196 ≤ 11.694 + 1, slowest 12.362 ≤ 11.974 + 2; costs 39.3272 / 36.94059 unchanged. Accepted source `logs/gap_improvements/G128_HG10_decode.patch` on HEAD 66e37b62.

### G131_build_HG11_dominators
- In progress: 2026-10-13T09:50:57.820522-04:00
- Metadata `logs/gap_improvements/G131_build_HG11_dominators.json`; log `logs/gap_improvements/G131_build_HG11_dominators.log`; patch `logs/gap_improvements/G131_build_HG11_dominators.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T09:51:57.698953-04:00; exit 0; runner elapsed 59.844 s.

### G132_native_HG11_relaxation_bound
- In progress: 2026-10-13T09:51:57.901846-04:00
- Metadata `logs/gap_improvements/G132_native_HG11_relaxation_bound.json`; log `logs/gap_improvements/G132_native_HG11_relaxation_bound.log`; patch `logs/gap_improvements/G132_native_HG11_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T09:51:59.057399-04:00; exit 0; runner elapsed 1.110 s.

### G133_oracle_HG11
- In progress: 2026-10-13T09:51:59.229635-04:00
- Metadata `logs/gap_improvements/G133_oracle_HG11.json`; log `logs/gap_improvements/G133_oracle_HG11.log`; patch `logs/gap_improvements/G133_oracle_HG11.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T09:52:02.783750-04:00; exit 0; runner elapsed 3.516 s.
- Oracle verdict: 37/37 cases passed.

### G134_diag_HG11_45s
- In progress: 2026-10-13T09:52:06.386600-04:00
- Metadata `logs/gap_improvements/G134_diag_HG11_45s.json`; log `logs/gap_improvements/G134_diag_HG11_45s.log`; patch `logs/gap_improvements/G134_diag_HG11_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T09:52:51.657082-04:00; exit 0; runner elapsed 45.234 s.
  `[SearchEngine volume] iterations=186 elapsed=36.9021191 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737041.7373476322 log2-remaining-fraction=-3114471.426392687 eliminated-bits=3114471.426392687 remaining-nodes=186 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.171000000 budget=45.0`
  `[Program deadline] overshoot=0.0115893`
  `[Program gap] primal-dual-integral=15.573951817576956 normalized=0.3460878181683768 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=11.736586093902588 samples=66`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791899538.2147374 program-deadline=1791899571.4781799`
- First incumbent: 11.736557484 s, cost 39.3272; best cost 36.94059.

### G135_restore_check_HG10
- In progress: 2026-10-13T09:53:53.908909-04:00
- Metadata `logs/gap_improvements/G135_restore_check_HG10.json`; log `logs/gap_improvements/G135_restore_check_HG10.log`; patch `logs/gap_improvements/G135_restore_check_HG10.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T09:53:53.990397-04:00; exit 0; runner elapsed 0.047 s.

### G136_build_HG12_window_search
- In progress: 2026-10-13T10:01:01.291571-04:00
- Metadata `logs/gap_improvements/G136_build_HG12_window_search.json`; log `logs/gap_improvements/G136_build_HG12_window_search.log`; patch `logs/gap_improvements/G136_build_HG12_window_search.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:01:59.892560-04:00; exit 0; runner elapsed 58.578 s.

### G137_native_HG12_relaxation_bound
- In progress: 2026-10-13T10:02:03.363540-04:00
- Metadata `logs/gap_improvements/G137_native_HG12_relaxation_bound.json`; log `logs/gap_improvements/G137_native_HG12_relaxation_bound.log`; patch `logs/gap_improvements/G137_native_HG12_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T10:02:07.350553-04:00; exit 0; runner elapsed 3.954 s.

### G138_diag_HG12_s78_b200
- In progress: 2026-10-13T10:02:14.141827-04:00
- Metadata `logs/gap_improvements/G138_diag_HG12_s78_b200.json`; log `logs/gap_improvements/G138_diag_HG12_s78_b200.log`; patch `logs/gap_improvements/G138_diag_HG12_s78_b200.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 240`
- Completed: 2026-10-13T10:06:14.462185-04:00; exit 0; runner elapsed 240.281 s.
  `[SearchEngine volume] iterations=316 elapsed=231.49109150000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737000.3107386853 log2-remaining-fraction=-3114512.8530016337 eliminated-bits=3114512.8530016337 remaining-nodes=316 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=240.218000000 budget=240.0`
  `[Program deadline] overshoot=0.0518756`
  `[Program gap] primal-dual-integral=188.80099467270125 normalized=0.78667081113625525 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=182.18281960487366 samples=136`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791900316.4117956 program-deadline=1791900374.2289996`
- First incumbent: 182.182796001 s, cost 39.3272; best cost 36.94059.

### G138_diag_HG12_s78_b2000
- In progress: 2026-10-13T10:06:14.684781-04:00
- Metadata `logs/gap_improvements/G138_diag_HG12_s78_b2000.json`; log `logs/gap_improvements/G138_diag_HG12_s78_b2000.log`; patch `logs/gap_improvements/G138_diag_HG12_s78_b2000.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 240`
- Completed: 2026-10-13T10:10:17.179048-04:00; exit 1; runner elapsed 242.453 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G139_build_HG12b_flip_branching
- In progress: 2026-10-13T10:11:13.395899-04:00
- Metadata `logs/gap_improvements/G139_build_HG12b_flip_branching.json`; log `logs/gap_improvements/G139_build_HG12b_flip_branching.log`; patch `logs/gap_improvements/G139_build_HG12b_flip_branching.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:12:12.268213-04:00; exit 0; runner elapsed 58.828 s.

### G140_native_HG12b
- In progress: 2026-10-13T10:12:12.459531-04:00
- Metadata `logs/gap_improvements/G140_native_HG12b.json`; log `logs/gap_improvements/G140_native_HG12b.log`; patch `logs/gap_improvements/G140_native_HG12b.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T10:12:16.449099-04:00; exit 0; runner elapsed 3.953 s.

### G141_diag_HG12b_s78_b200
- In progress: 2026-10-13T10:12:16.621596-04:00
- Metadata `logs/gap_improvements/G141_diag_HG12b_s78_b200.json`; log `logs/gap_improvements/G141_diag_HG12b_s78_b200.log`; patch `logs/gap_improvements/G141_diag_HG12b_s78_b200.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T10:14:18.766944-04:00; exit 1; runner elapsed 122.110 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G142_diag_HG12b_s78_b2000
- In progress: 2026-10-13T10:14:24.834122-04:00
- Metadata `logs/gap_improvements/G142_diag_HG12b_s78_b2000.json`; log `logs/gap_improvements/G142_diag_HG12b_s78_b2000.log`; patch `logs/gap_improvements/G142_diag_HG12b_s78_b2000.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 400`
- Completed: 2026-10-13T10:21:07.327582-04:00; exit 1; runner elapsed 402.422 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G143_restore_build_HG10
- In progress: 2026-10-13T10:21:23.309406-04:00
- Metadata `logs/gap_improvements/G143_restore_build_HG10.json`; log `logs/gap_improvements/G143_restore_build_HG10.log`; patch `logs/gap_improvements/G143_restore_build_HG10.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:22:38.363957-04:00; exit 0; runner elapsed 74.969 s.

- Notes G131–G143 (2026-10-13):
  - **H-G11 dominator-truncated windows** (rejected): owned children (one region parent) summed directly, shared classes charged by their immediate dominator under alternatives where they are necessary. Sound (brute force G132, oracle 37/37 incl. case 37), but no gain on Gemma (G134: 32.757). Python replay on the exported layer window: 144 of 281 classes are shared and few are necessary under a fixed alternative, so the truncated sum is only ≈0.49 per layer (vs 1.11). Saved: `logs/gap_improvements/HG11_dominator_trial.diff`, case `HG11_case37.json`.
  - **Where the per-layer bound loses** (offline CP-SAT, research only): at λ_c = 0.8 the exact minimum weighted load is 1.101 vs our recursion ≈1.109 (with path terms), so the support recursion is near-exact for the Lagrangian. The exact Lagrangian with a critical-path term reaches 1.185 (μ = 0.5, λ_c = 1). The true min-max (max of CP, CPU load, GPU load) is 1.2747 (CPU 1.2747, GPU 0.643, CP 1.267; the λ-argmin selection differs in 14 classes). So most of the per-layer gap (1.11 → 1.27, ≈2.9 per bucket) is a duality gap that a Lagrangian bound cannot close.
  - **H-G12 window branch and bound** (rejected): best-first over alternative restrictions, node bound = path-weighted relaxation over the remaining alternatives, upper bound = true value of the relaxation's selection, branch on classes whose relaxation choice flips under nearby multipliers (fallback: longest operation). Sound (brute force G137/G140: 54000 searched windows, never above the minimum makespan or the exact value; budget 1000 reaches the exact value in 17694/18000). Gemma, per-layer windows: longest-operation branching +0.01 (budget 200, 11 s) / +0.10 (2000, 182 s); flip branching +0.11 (200, 12 s) / +0.27 (2000, 205 s). Far too slow to close ≈2.9. Saved: `logs/gap_improvements/HG12_window_search_trial.diff`.
  - G143: working tree restored to the accepted H-G10 source (captured patch identical to G128) and rebuilt.

### G144_build_HG13_threshold
- In progress: 2026-10-13T10:27:16.732077-04:00
- Metadata `logs/gap_improvements/G144_build_HG13_threshold.json`; log `logs/gap_improvements/G144_build_HG13_threshold.log`; patch `logs/gap_improvements/G144_build_HG13_threshold.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:28:15.382570-04:00; exit 0; runner elapsed 58.609 s.

### G145_native_HG13_relaxation_bound
- In progress: 2026-10-13T10:28:15.585299-04:00
- Metadata `logs/gap_improvements/G145_native_HG13_relaxation_bound.json`; log `logs/gap_improvements/G145_native_HG13_relaxation_bound.log`; patch `logs/gap_improvements/G145_native_HG13_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T10:28:16.980995-04:00; exit 0; runner elapsed 1.360 s.

### G146_build_HG13b
- In progress: 2026-10-13T10:28:36.985901-04:00
- Metadata `logs/gap_improvements/G146_build_HG13b.json`; log `logs/gap_improvements/G146_build_HG13b.log`; patch `logs/gap_improvements/G146_build_HG13b.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:29:35.749450-04:00; exit 0; runner elapsed 58.734 s.

### G147_native_HG13b_relaxation_bound
- In progress: 2026-10-13T10:29:35.941832-04:00
- Metadata `logs/gap_improvements/G147_native_HG13b_relaxation_bound.json`; log `logs/gap_improvements/G147_native_HG13b_relaxation_bound.log`; patch `logs/gap_improvements/G147_native_HG13b_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T10:29:38.428263-04:00; exit 0; runner elapsed 2.453 s.

### G148_diag_HG13_45s
- In progress: 2026-10-13T10:29:38.606798-04:00
- Metadata `logs/gap_improvements/G148_diag_HG13_45s.json`; log `logs/gap_improvements/G148_diag_HG13_45s.log`; patch `logs/gap_improvements/G148_diag_HG13_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T10:30:23.881851-04:00; exit 0; runner elapsed 45.234 s.
  `[SearchEngine volume] iterations=181 elapsed=36.709010999999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737043.0277512658 log2-remaining-fraction=-3114470.1359890532 eliminated-bits=3114470.1359890532 remaining-nodes=181 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.172000000 budget=45.0`
  `[Program deadline] overshoot=0.00696349`
  `[Program gap] primal-dual-integral=16.371110535385405 normalized=0.3638024563418979 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=12.637348651885986 samples=64`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791901791.3330009 program-deadline=1791901823.6956794`
- First incumbent: 12.637321472 s, cost 39.3272; best cost 36.94059.

### G149_build_diag_threshold
- In progress: 2026-10-13T10:30:36.098405-04:00
- Metadata `logs/gap_improvements/G149_build_diag_threshold.json`; log `logs/gap_improvements/G149_build_diag_threshold.log`; patch `logs/gap_improvements/G149_build_diag_threshold.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:31:34.663539-04:00; exit 0; runner elapsed 58.515 s.

### G150_diag_threshold_30s
- In progress: 2026-10-13T10:31:34.857231-04:00
- Metadata `logs/gap_improvements/G150_diag_threshold_30s.json`; log `logs/gap_improvements/G150_diag_threshold_30s.log`; patch `logs/gap_improvements/G150_diag_threshold_30s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 30`
- Completed: 2026-10-13T10:32:05.128210-04:00; exit 0; runner elapsed 30.234 s.
  `[SearchEngine volume] iterations=50 elapsed=21.743022700000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737070.3066215797 log2-remaining-fraction=-3114442.8571187397 eliminated-bits=3114442.8571187397 remaining-nodes=50 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=30.156000000 budget=30.0`
  `[Program deadline] overshoot=0.00720191`
  `[Program gap] primal-dual-integral=14.636178225789839 normalized=0.48787260752632799 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=12.595427989959717 samples=18`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791901907.5414488 program-deadline=1791901924.9460473`
- First incumbent: 12.595401525 s, cost 39.3272; best cost 36.94059.

### G151_build_HG13c
- In progress: 2026-10-13T10:32:34.967164-04:00
- Metadata `logs/gap_improvements/G151_build_HG13c.json`; log `logs/gap_improvements/G151_build_HG13c.log`; patch `logs/gap_improvements/G151_build_HG13c.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:33:33.638192-04:00; exit 0; runner elapsed 58.641 s.

### G152_diag_HG13c_60s
- In progress: 2026-10-13T10:33:33.827272-04:00
- Metadata `logs/gap_improvements/G152_diag_HG13c_60s.json`; log `logs/gap_improvements/G152_diag_HG13c_60s.log`; patch `logs/gap_improvements/G152_diag_HG13c_60s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T10:34:34.170444-04:00; exit 0; runner elapsed 60.312 s.
  `[SearchEngine volume] iterations=278 elapsed=51.457748600000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737005.1314388816 log2-remaining-fraction=-3114508.0323014376 eliminated-bits=3114508.0323014376 remaining-nodes=278 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.235000000 budget=60.0`
  `[Program deadline] overshoot=0.0833743`
  `[Program gap] primal-dual-integral=19.525109793417197 normalized=0.32541849655695326 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=14.277051687240601 samples=104`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791902028.1937201 program-deadline=1791902073.9166958`
- First incumbent: 14.277024269 s, cost 39.3272; best cost 36.94059.

### G153_build_diag_stride78
- In progress: 2026-10-13T10:34:51.028781-04:00
- Metadata `logs/gap_improvements/G153_build_diag_stride78.json`; log `logs/gap_improvements/G153_build_diag_stride78.log`; patch `logs/gap_improvements/G153_build_diag_stride78.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:35:49.850070-04:00; exit 0; runner elapsed 58.781 s.

### G154_diag_threshold_s78
- In progress: 2026-10-13T10:35:50.040402-04:00
- Metadata `logs/gap_improvements/G154_diag_threshold_s78.json`; log `logs/gap_improvements/G154_diag_threshold_s78.log`; patch `logs/gap_improvements/G154_diag_threshold_s78.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T10:36:50.385113-04:00; exit 0; runner elapsed 60.297 s.
  `[SearchEngine volume] iterations=280 elapsed=51.737348400000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737004.671520497 log2-remaining-fraction=-3114508.4922198225 eliminated-bits=3114508.4922198225 remaining-nodes=280 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.235000000 budget=60.0`
  `[Program deadline] overshoot=0.0802202`
  `[Program gap] primal-dual-integral=19.548336739642647 normalized=0.32580561232737743 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=14.303471088409424 samples=105`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791902164.4322667 program-deadline=1791902210.1288257`
- First incumbent: 14.303441048 s, cost 39.3272; best cost 36.94059.

### G155_build_diag_infeasible
- In progress: 2026-10-13T10:37:04.544224-04:00
- Metadata `logs/gap_improvements/G155_build_diag_infeasible.json`; log `logs/gap_improvements/G155_build_diag_infeasible.log`; patch `logs/gap_improvements/G155_build_diag_infeasible.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:38:03.344100-04:00; exit 0; runner elapsed 58.766 s.

### G156_diag_infeasible
- In progress: 2026-10-13T10:38:03.537040-04:00
- Metadata `logs/gap_improvements/G156_diag_infeasible.json`; log `logs/gap_improvements/G156_diag_infeasible.log`; patch `logs/gap_improvements/G156_diag_infeasible.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T10:39:03.913121-04:00; exit 0; runner elapsed 60.343 s.
  `[SearchEngine volume] iterations=265 elapsed=51.303225400000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737007.7229884006 log2-remaining-fraction=-3114505.4407519186 eliminated-bits=3114505.4407519186 remaining-nodes=265 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.266000000 budget=60.0`
  `[Program deadline] overshoot=0.115202`
  `[Program gap] primal-dual-integral=20.047639924782068 normalized=0.33412733207970113 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=14.865792512893677 samples=99`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791902298.4952312 program-deadline=1791902343.6294622`
- First incumbent: 14.865768909 s, cost 39.3272; best cost 36.94059.

### G157_build_diag_infeasible2
- In progress: 2026-10-13T10:39:11.665942-04:00
- Metadata `logs/gap_improvements/G157_build_diag_infeasible2.json`; log `logs/gap_improvements/G157_build_diag_infeasible2.log`; patch `logs/gap_improvements/G157_build_diag_infeasible2.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:40:10.511929-04:00; exit 0; runner elapsed 58.812 s.

### G158_diag_infeasible2
- In progress: 2026-10-13T10:40:10.700383-04:00
- Metadata `logs/gap_improvements/G158_diag_infeasible2.json`; log `logs/gap_improvements/G158_diag_infeasible2.log`; patch `logs/gap_improvements/G158_diag_infeasible2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60`
- Completed: 2026-10-13T10:41:11.372705-04:00; exit 0; runner elapsed 60.625 s.
  `[SearchEngine volume] iterations=261 elapsed=51.973960400000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737008.6011963484 log2-remaining-fraction=-3114504.5625439705 eliminated-bits=3114504.5625439705 remaining-nodes=261 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=60.563000000 budget=60.0`
  `[Program deadline] overshoot=0.0859792`
  `[Program gap] primal-dual-integral=19.460599297516623 normalized=0.32434332162527707 final-gap=0.11326226787390249 final-incumbent=36.94059 final-bound=32.756614999999996 first-incumbent-elapsed=14.203473329544067 samples=100`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791902424.9908843 program-deadline=1791902470.7874415`
- First incumbent: 14.203442812 s, cost 39.3272; best cost 36.94059.

### G159_restore_build_HG10
- In progress: 2026-10-13T10:41:47.698988-04:00
- Metadata `logs/gap_improvements/G159_restore_build_HG10.json`; log `logs/gap_improvements/G159_restore_build_HG10.log`; patch `logs/gap_improvements/G159_restore_build_HG10.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:42:46.549173-04:00; exit 0; runner elapsed 58.812 s.

- Notes G144–G159 (2026-10-13):
  - **H-G13 threshold bound** (rejected, saved as `logs/gap_improvements/HG13_threshold_trial.diff`): to prove that no window selection has max(chain, loads) ≤ T, remove alternatives whose head (shortest chain below) plus tail (shortest chain up to a target) exceeds T, to a fixpoint, then show the λ-weighted load recursion over the remaining alternatives exceeds T (or that a restricted exact enumeration does); bisection on T. Sound: brute force G147 (18000 checks with exact enumeration disabled; never above the minimum makespan or the exact value; reaches the exact value in 16823). Offline prototype with an exact (CP-SAT) inner minimum proves 1.208 on the layer window (vs 1.109). Natively no gain (G150–G158): after pruning at T = 1.178 our support recursion gives L(λ_c = 1) = 1.087 while the exact minimum weighted load is ≈1.20 (overlapping supports drop most anchors in the pruned window), and the restricted enumeration does not finish. An LP relaxation of the extraction (GLOP, offline) is useless here (≈0.02): fractional selections halve the cost per level.
  - G159: working tree restored to the accepted H-G10 source (patch identical to G128) and rebuilt.
  - Conclusion for the goal: every route to a bound near 36 found so far needs an exact minimum-cost DAG extraction (with a hard chain threshold) for ~281-class windows inside the run; no such native solver exists in this code base, and the polynomial relaxations available (support recursion, dominator truncation, LP) are far from exact on these windows.

### G160_build_HG14_cut_dp
- In progress: 2026-10-13T10:45:02.615242-04:00
- Metadata `logs/gap_improvements/G160_build_HG14_cut_dp.json`; log `logs/gap_improvements/G160_build_HG14_cut_dp.log`; patch `logs/gap_improvements/G160_build_HG14_cut_dp.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:46:01.585236-04:00; exit 0; runner elapsed 58.937 s.

### G161_native_HG14_relaxation_bound
- In progress: 2026-10-13T10:46:01.792662-04:00
- Metadata `logs/gap_improvements/G161_native_HG14_relaxation_bound.json`; log `logs/gap_improvements/G161_native_HG14_relaxation_bound.log`; patch `logs/gap_improvements/G161_native_HG14_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T10:46:03.741957-04:00; exit 0; runner elapsed 1.922 s.

### G162_diag_HG14_45s
- In progress: 2026-10-13T10:46:03.913420-04:00
- Metadata `logs/gap_improvements/G162_diag_HG14_45s.json`; log `logs/gap_improvements/G162_diag_HG14_45s.log`; patch `logs/gap_improvements/G162_diag_HG14_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T10:46:51.413911-04:00; exit 1; runner elapsed 47.468 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G163_build_diag_cutdp
- In progress: 2026-10-13T10:47:07.318680-04:00
- Metadata `logs/gap_improvements/G163_build_diag_cutdp.json`; log `logs/gap_improvements/G163_build_diag_cutdp.log`; patch `logs/gap_improvements/G163_build_diag_cutdp.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:48:18.283632-04:00; exit 0; runner elapsed 70.891 s.

### G164_diag_cutdp_300s
- In progress: 2026-10-13T10:48:18.475972-04:00
- Metadata `logs/gap_improvements/G164_diag_cutdp_300s.json`; log `logs/gap_improvements/G164_diag_cutdp_300s.log`; patch `logs/gap_improvements/G164_diag_cutdp_300s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 300`
- Completed: 2026-10-13T10:53:18.871829-04:00; exit 0; runner elapsed 300.360 s.
  `[SearchEngine volume] iterations=127 elapsed=291.89825109999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737055.4235681961 log2-remaining-fraction=-3114457.7401721231 eliminated-bits=3114457.7401721231 remaining-nodes=127 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=300.266000000 budget=300.0`
  `[Program deadline] overshoot=0.0697389`
  `[Program gap] primal-dual-integral=268.10190543915206 normalized=0.89367301813050692 final-gap=0.058497251938856333 final-incumbent=36.94059 final-bound=34.779667000000003 first-incumbent-elapsed=266.02881217002869 samples=55`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791903164.5938668 program-deadline=1791903198.5650835`
- First incumbent: 266.028783321 s, cost 39.3272; best cost 36.94059.

### G165_build_diag_strides_cutdp
- In progress: 2026-10-13T10:53:42.250198-04:00
- Metadata `logs/gap_improvements/G165_build_diag_strides_cutdp.json`; log `logs/gap_improvements/G165_build_diag_strides_cutdp.log`; patch `logs/gap_improvements/G165_build_diag_strides_cutdp.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:54:41.624566-04:00; exit 0; runner elapsed 59.329 s.

### G166_diag_strides_cutdp
- In progress: 2026-10-13T10:54:41.816536-04:00
- Metadata `logs/gap_improvements/G166_diag_strides_cutdp.json`; log `logs/gap_improvements/G166_diag_strides_cutdp.log`; patch `logs/gap_improvements/G166_diag_strides_cutdp.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 200`
- Completed: 2026-10-13T10:58:04.194004-04:00; exit 1; runner elapsed 202.328 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### G167_build_diag_strides2
- In progress: 2026-10-13T10:58:20.590282-04:00
- Metadata `logs/gap_improvements/G167_build_diag_strides2.json`; log `logs/gap_improvements/G167_build_diag_strides2.log`; patch `logs/gap_improvements/G167_build_diag_strides2.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T10:59:19.470821-04:00; exit 0; runner elapsed 58.843 s.

### G168_diag_strides2
- In progress: 2026-10-13T10:59:19.661232-04:00
- Metadata `logs/gap_improvements/G168_diag_strides2.json`; log `logs/gap_improvements/G168_diag_strides2.log`; patch `logs/gap_improvements/G168_diag_strides2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T11:01:20.019659-04:00; exit 0; runner elapsed 120.312 s.
  `[SearchEngine volume] iterations=200 elapsed=111.74994390000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737037.2300899683 log2-remaining-fraction=-3114475.9336503511 eliminated-bits=3114475.9336503511 remaining-nodes=200 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.250000000 budget=120.0`
  `[Program deadline] overshoot=0.102274`
  `[Program gap] primal-dual-integral=75.284605108753738 normalized=0.62737170923961449 final-gap=0.05824300586428096 final-incumbent=36.94059 final-bound=34.789059000000002 first-incumbent-elapsed=72.387316465377808 samples=86`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791903632.1366651 program-deadline=1791903679.7493901`
- First incumbent: 72.387274981 s, cost 39.3272; best cost 36.94059.

### G169_build_diag_ascending
- In progress: 2026-10-13T11:01:45.073343-04:00
- Metadata `logs/gap_improvements/G169_build_diag_ascending.json`; log `logs/gap_improvements/G169_build_diag_ascending.log`; patch `logs/gap_improvements/G169_build_diag_ascending.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:02:43.814227-04:00; exit 0; runner elapsed 58.703 s.

### G170_diag_ascending
- In progress: 2026-10-13T11:02:44.005505-04:00
- Metadata `logs/gap_improvements/G170_diag_ascending.json`; log `logs/gap_improvements/G170_diag_ascending.log`; patch `logs/gap_improvements/G170_diag_ascending.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 90`
- Completed: 2026-10-13T11:04:14.377844-04:00; exit 0; runner elapsed 90.329 s.
  `[SearchEngine volume] iterations=446 elapsed=81.895360100000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736984.6788079033 log2-remaining-fraction=-3114528.4849324161 eliminated-bits=3114528.4849324161 remaining-nodes=446 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=90.266000000 budget=90.0`
  `[Program deadline] overshoot=0.106259`
  `[Program gap] primal-dual-integral=17.390237759213235 normalized=0.19322486399125818 final-gap=0.053660485660894978 final-incumbent=36.94059 final-bound=34.95834 first-incumbent-elapsed=13.19352388381958 samples=182`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791903777.2906623 program-deadline=1791903854.0971608`
- First incumbent: 13.193501472 s, cost 39.3272; best cost 36.94059.

### G171_build_HG14b_dedupe
- In progress: 2026-10-13T11:05:23.243163-04:00
- Metadata `logs/gap_improvements/G171_build_HG14b_dedupe.json`; log `logs/gap_improvements/G171_build_HG14b_dedupe.log`; patch `logs/gap_improvements/G171_build_HG14b_dedupe.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:06:22.570594-04:00; exit 0; runner elapsed 59.281 s.

### G172_diag_HG14b_45s
- In progress: 2026-10-13T11:06:22.791511-04:00
- Metadata `logs/gap_improvements/G172_diag_HG14b_45s.json`; log `logs/gap_improvements/G172_diag_HG14b_45s.log`; patch `logs/gap_improvements/G172_diag_HG14b_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:07:08.135218-04:00; exit 0; runner elapsed 45.313 s.
  `[SearchEngine volume] iterations=165 elapsed=36.897652299999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737045.8553005373 log2-remaining-fraction=-3114467.3084397819 eliminated-bits=3114467.3084397819 remaining-nodes=165 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.234000000 budget=45.0`
  `[Program deadline] overshoot=0.092515`
  `[Program gap] primal-dual-integral=17.281796096079496 normalized=0.38403991324621101 final-gap=0.053660485660894978 final-incumbent=36.94059 final-bound=34.95834 first-incumbent-elapsed=15.632553100585938 samples=57`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791903998.5156784 program-deadline=1791904027.8831551`
- First incumbent: 15.632523298 s, cost 39.3272; best cost 36.94059.

### G173_build_HG14c
- In progress: 2026-10-13T11:07:21.416538-04:00
- Metadata `logs/gap_improvements/G173_build_HG14c.json`; log `logs/gap_improvements/G173_build_HG14c.log`; patch `logs/gap_improvements/G173_build_HG14c.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:08:20.524322-04:00; exit 0; runner elapsed 59.062 s.

### G174_diag_HG14c_45s
- In progress: 2026-10-13T11:08:20.711798-04:00
- Metadata `logs/gap_improvements/G174_diag_HG14c_45s.json`; log `logs/gap_improvements/G174_diag_HG14c_45s.log`; patch `logs/gap_improvements/G174_diag_HG14c_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:09:06.073037-04:00; exit 0; runner elapsed 45.328 s.
  `[SearchEngine volume] iterations=189 elapsed=36.638800099999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737040.7897846098 log2-remaining-fraction=-3114472.3739557094 eliminated-bits=3114472.3739557094 remaining-nodes=189 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.250000000 budget=45.0`
  `[Program deadline] overshoot=0.132577`
  `[Program gap] primal-dual-integral=13.869402876261532 normalized=0.30820895280581184 final-gap=0.053660485660894978 final-incumbent=36.94059 final-bound=34.95834 first-incumbent-elapsed=12.026442289352417 samples=67`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791904112.8234041 program-deadline=1791904145.7969909`
- First incumbent: 12.026413202 s, cost 39.3272; best cost 36.94059.

### G175_build_HG14
- In progress: 2026-10-13T11:09:24.977460-04:00
- Metadata `logs/gap_improvements/G175_build_HG14.json`; log `logs/gap_improvements/G175_build_HG14.log`; patch `logs/gap_improvements/G175_build_HG14.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:10:24.071028-04:00; exit 0; runner elapsed 59.047 s.

### G176_oracle_HG14
- In progress: 2026-10-13T11:10:24.264295-04:00
- Metadata `logs/gap_improvements/G176_oracle_HG14.json`; log `logs/gap_improvements/G176_oracle_HG14.log`; patch `logs/gap_improvements/G176_oracle_HG14.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T11:10:27.598763-04:00; exit 0; runner elapsed 3.297 s.
- Oracle verdict: 36/36 cases passed.

### G177_gap_unit_tests_HG14
- In progress: 2026-10-13T11:10:27.772722-04:00
- Metadata `logs/gap_improvements/G177_gap_unit_tests_HG14.json`; log `logs/gap_improvements/G177_gap_unit_tests_HG14.log`; patch `logs/gap_improvements/G177_gap_unit_tests_HG14.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T11:10:27.857475-04:00; exit 0; runner elapsed 0.031 s.

### G178_native_HG14_private_layout
- In progress: 2026-10-13T11:10:28.029811-04:00
- Metadata `logs/gap_improvements/G178_native_HG14_private_layout.json`; log `logs/gap_improvements/G178_native_HG14_private_layout.log`; patch `logs/gap_improvements/G178_native_HG14_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T11:10:28.145879-04:00; exit 0; runner elapsed 0.078 s.

### G178_native_HG14_write_after_read_cache
- In progress: 2026-10-13T11:10:28.326161-04:00
- Metadata `logs/gap_improvements/G178_native_HG14_write_after_read_cache.json`; log `logs/gap_improvements/G178_native_HG14_write_after_read_cache.log`; patch `logs/gap_improvements/G178_native_HG14_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T11:10:28.786060-04:00; exit 0; runner elapsed 0.422 s.

### G178_native_HG14_incremental_propagators
- In progress: 2026-10-13T11:10:28.957921-04:00
- Metadata `logs/gap_improvements/G178_native_HG14_incremental_propagators.json`; log `logs/gap_improvements/G178_native_HG14_incremental_propagators.log`; patch `logs/gap_improvements/G178_native_HG14_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T11:10:29.042708-04:00; exit 0; runner elapsed 0.047 s.

### G178_native_HG14_start_rank_window
- In progress: 2026-10-13T11:10:29.216640-04:00
- Metadata `logs/gap_improvements/G178_native_HG14_start_rank_window.json`; log `logs/gap_improvements/G178_native_HG14_start_rank_window.log`; patch `logs/gap_improvements/G178_native_HG14_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T11:10:29.774916-04:00; exit 0; runner elapsed 0.531 s.

### G178_native_HG14_relaxation_bound
- In progress: 2026-10-13T11:10:29.948118-04:00
- Metadata `logs/gap_improvements/G178_native_HG14_relaxation_bound.json`; log `logs/gap_improvements/G178_native_HG14_relaxation_bound.log`; patch `logs/gap_improvements/G178_native_HG14_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T11:10:31.877784-04:00; exit 0; runner elapsed 1.891 s.

### G179_HG14_decode
- In progress: 2026-10-13T11:10:35.076344-04:00
- Metadata `logs/gap_improvements/G179_HG14_decode.json`; log `logs/gap_improvements/G179_HG14_decode.log`; patch `logs/gap_improvements/G179_HG14_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T11:20:35.461497-04:00; exit 0; runner elapsed 600.328 s.
  `[SearchEngine volume] iterations=2512 elapsed=591.85035789999995 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736832.6774712505 log2-remaining-fraction=-3114680.4862690689 eliminated-bits=3114680.4862690689 remaining-nodes=2512 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.265000000 budget=600.0`
  `[Program deadline] overshoot=0.132336`
  `[Program gap] primal-dual-integral=43.420005985654711 normalized=0.072366676642757852 final-gap=0.053660485660894978 final-incumbent=36.94059 final-bound=34.95834 first-incumbent-elapsed=11.781800031661987 samples=1626`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791904246.9452324 program-deadline=1791904835.1634564`
- First incumbent: 11.781775951 s, cost 39.3272; best cost 36.94059.

### G180_build_HG15_dyadic
- In progress: 2026-10-13T11:20:45.077547-04:00
- Metadata `logs/gap_improvements/G180_build_HG15_dyadic.json`; log `logs/gap_improvements/G180_build_HG15_dyadic.log`; patch `logs/gap_improvements/G180_build_HG15_dyadic.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:21:44.249255-04:00; exit 0; runner elapsed 59.141 s.

### G181_native_HG15_relaxation_bound
- In progress: 2026-10-13T11:21:44.457058-04:00
- Metadata `logs/gap_improvements/G181_native_HG15_relaxation_bound.json`; log `logs/gap_improvements/G181_native_HG15_relaxation_bound.log`; patch `logs/gap_improvements/G181_native_HG15_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T11:21:48.148171-04:00; exit 0; runner elapsed 3.657 s.

### G182_diag_HG15_45s
- In progress: 2026-10-13T11:21:48.321480-04:00
- Metadata `logs/gap_improvements/G182_diag_HG15_45s.json`; log `logs/gap_improvements/G182_diag_HG15_45s.log`; patch `logs/gap_improvements/G182_diag_HG15_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:22:33.644804-04:00; exit 0; runner elapsed 45.297 s.
  `[SearchEngine volume] iterations=184 elapsed=36.885099599999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737042.2259212234 log2-remaining-fraction=-3114470.9378190958 eliminated-bits=3114470.9378190958 remaining-nodes=184 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.219000000 budget=45.0`
  `[Program deadline] overshoot=0.0675364`
  `[Program gap] primal-dual-integral=13.213255314229452 normalized=0.29362789587176558 final-gap=0.043287451553968098 final-incumbent=36.94059 final-bound=35.341526000000002 first-incumbent-elapsed=11.695238828659058 samples=65`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791904920.1036496 program-deadline=1791904953.4084373`
- First incumbent: 11.695212364 s, cost 39.3272; best cost 36.94059.

### G183_build_diag_variants
- In progress: 2026-10-13T11:22:47.757508-04:00
- Metadata `logs/gap_improvements/G183_build_diag_variants.json`; log `logs/gap_improvements/G183_build_diag_variants.log`; patch `logs/gap_improvements/G183_build_diag_variants.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:23:46.916619-04:00; exit 0; runner elapsed 59.125 s.

### G184_diag_nostop
- In progress: 2026-10-13T11:23:47.109436-04:00
- Metadata `logs/gap_improvements/G184_diag_nostop.json`; log `logs/gap_improvements/G184_diag_nostop.log`; patch `logs/gap_improvements/G184_diag_nostop.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T11:25:47.564850-04:00; exit 0; runner elapsed 120.421 s.
  `[SearchEngine volume] iterations=593 elapsed=111.7793915 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736967.9487184116 log2-remaining-fraction=-3114545.2150219074 eliminated-bits=3114545.2150219074 remaining-nodes=593 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.344000000 budget=120.0`
  `[Program deadline] overshoot=0.153381`
  `[Program gap] primal-dual-integral=22.07570017654087 normalized=0.18396416813784058 final-gap=0.043287451553968098 final-incumbent=36.94059 final-bound=35.341526000000002 first-incumbent-elapsed=17.567975759506226 samples=265`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905044.7655168 program-deadline=1791905147.1975679`
- First incumbent: 17.567948818 s, cost 39.3272; best cost 36.94059.

### G184_diag_nostop_dp1500
- In progress: 2026-10-13T11:25:47.773253-04:00
- Metadata `logs/gap_improvements/G184_diag_nostop_dp1500.json`; log `logs/gap_improvements/G184_diag_nostop_dp1500.log`; patch `logs/gap_improvements/G184_diag_nostop_dp1500.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T11:27:48.222047-04:00; exit 0; runner elapsed 120.406 s.
  `[SearchEngine volume] iterations=543 elapsed=111.6962868 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736976.7850250935 log2-remaining-fraction=-3114536.3787152255 eliminated-bits=3114536.3787152255 remaining-nodes=543 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.328000000 budget=120.0`
  `[Program deadline] overshoot=0.0809102`
  `[Program gap] primal-dual-integral=28.98828896495214 normalized=0.24156907470793451 final-gap=0.043287451553968098 final-incumbent=36.94059 final-bound=35.341526000000002 first-incumbent-elapsed=24.792389392852783 samples=237`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905172.6537292 program-deadline=1791905267.8613677`
- First incumbent: 24.792361498 s, cost 39.3272; best cost 36.94059.

### G185_build_HG16_tail_dp
- In progress: 2026-10-13T11:28:38.902130-04:00
- Metadata `logs/gap_improvements/G185_build_HG16_tail_dp.json`; log `logs/gap_improvements/G185_build_HG16_tail_dp.log`; patch `logs/gap_improvements/G185_build_HG16_tail_dp.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:29:37.985302-04:00; exit 0; runner elapsed 59.047 s.

### G186_native_HG16
- In progress: 2026-10-13T11:29:38.186078-04:00
- Metadata `logs/gap_improvements/G186_native_HG16.json`; log `logs/gap_improvements/G186_native_HG16.log`; patch `logs/gap_improvements/G186_native_HG16.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T11:29:42.614711-04:00; exit 0; runner elapsed 4.391 s.

### G187_diag_HG16_45s
- In progress: 2026-10-13T11:29:42.799453-04:00
- Metadata `logs/gap_improvements/G187_diag_HG16_45s.json`; log `logs/gap_improvements/G187_diag_HG16_45s.log`; patch `logs/gap_improvements/G187_diag_HG16_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:30:28.179250-04:00; exit 0; runner elapsed 45.328 s.
  `[SearchEngine volume] iterations=179 elapsed=36.821173399999999 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737043.4271904931 log2-remaining-fraction=-3114469.7365498263 eliminated-bits=3114469.7365498263 remaining-nodes=179 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.266000000 budget=45.0`
  `[Program deadline] overshoot=0.11906`
  `[Program gap] primal-dual-integral=14.804211751729616 normalized=0.32898248337176922 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=13.620096921920776 samples=63`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905396.5065861 program-deadline=1791905427.8865151`
- First incumbent: 13.620070934 s, cost 39.3272; best cost 36.94059.

### G188_build_HG16b_parallel_windows
- In progress: 2026-10-13T11:30:54.836039-04:00
- Metadata `logs/gap_improvements/G188_build_HG16b_parallel_windows.json`; log `logs/gap_improvements/G188_build_HG16b_parallel_windows.log`; patch `logs/gap_improvements/G188_build_HG16b_parallel_windows.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:31:53.766028-04:00; exit 0; runner elapsed 58.891 s.

### G189_diag_HG16b_45s
- In progress: 2026-10-13T11:31:53.973515-04:00
- Metadata `logs/gap_improvements/G189_diag_HG16b_45s.json`; log `logs/gap_improvements/G189_diag_HG16b_45s.log`; patch `logs/gap_improvements/G189_diag_HG16b_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:32:39.274136-04:00; exit 0; runner elapsed 45.250 s.
  `[SearchEngine volume] iterations=188 elapsed=36.847321899999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737041.1962443923 log2-remaining-fraction=-3114471.9674959267 eliminated-bits=3114471.9674959267 remaining-nodes=188 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.187000000 budget=45.0`
  `[Program deadline] overshoot=0.0620923`
  `[Program gap] primal-dual-integral=12.980876304959994 normalized=0.28846391788799985 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=11.730014562606812 samples=67`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905525.7906864 program-deadline=1791905559.0606949`
- First incumbent: 11.729991436 s, cost 39.3272; best cost 36.94059.

### G190_build_diag_caps
- In progress: 2026-10-13T11:32:47.512043-04:00
- Metadata `logs/gap_improvements/G190_build_diag_caps.json`; log `logs/gap_improvements/G190_build_diag_caps.log`; patch `logs/gap_improvements/G190_build_diag_caps.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:33:46.648286-04:00; exit 0; runner elapsed 59.094 s.

### G191_diag_caps_45s
- In progress: 2026-10-13T11:33:46.838331-04:00
- Metadata `logs/gap_improvements/G191_diag_caps_45s.json`; log `logs/gap_improvements/G191_diag_caps_45s.log`; patch `logs/gap_improvements/G191_diag_caps_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T11:34:32.111732-04:00; exit 0; runner elapsed 45.235 s.
  `[SearchEngine volume] iterations=180 elapsed=36.460331400000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737043.20683986 log2-remaining-fraction=-3114469.9569004592 eliminated-bits=3114469.9569004592 remaining-nodes=180 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.187000000 budget=45.0`
  `[Program deadline] overshoot=0.0613434`
  `[Program gap] primal-dual-integral=13.954359258795069 normalized=0.3100968724176682 final-gap=0.035281677959122973 final-incumbent=36.94059 final-bound=35.637264000000002 first-incumbent-elapsed=12.7395179271698 samples=64`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905639.6550367 program-deadline=1791905671.9155481`
- First incumbent: 12.739488602 s, cost 39.3272; best cost 36.94059.

### G192_build_HG16
- In progress: 2026-10-13T11:34:41.722129-04:00
- Metadata `logs/gap_improvements/G192_build_HG16.json`; log `logs/gap_improvements/G192_build_HG16.log`; patch `logs/gap_improvements/G192_build_HG16.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:35:40.874056-04:00; exit 0; runner elapsed 59.125 s.

### G193_oracle_HG16
- In progress: 2026-10-13T11:35:41.060318-04:00
- Metadata `logs/gap_improvements/G193_oracle_HG16.json`; log `logs/gap_improvements/G193_oracle_HG16.log`; patch `logs/gap_improvements/G193_oracle_HG16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T11:35:44.410644-04:00; exit 0; runner elapsed 3.313 s.
- Oracle verdict: 36/36 cases passed.

### G194_gap_unit_tests_HG16
- In progress: 2026-10-13T11:35:44.584244-04:00
- Metadata `logs/gap_improvements/G194_gap_unit_tests_HG16.json`; log `logs/gap_improvements/G194_gap_unit_tests_HG16.log`; patch `logs/gap_improvements/G194_gap_unit_tests_HG16.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T11:35:44.664697-04:00; exit 0; runner elapsed 0.032 s.

### G195_native_HG16_private_layout
- In progress: 2026-10-13T11:35:44.837516-04:00
- Metadata `logs/gap_improvements/G195_native_HG16_private_layout.json`; log `logs/gap_improvements/G195_native_HG16_private_layout.log`; patch `logs/gap_improvements/G195_native_HG16_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T11:35:44.948191-04:00; exit 0; runner elapsed 0.078 s.

### G195_native_HG16_write_after_read_cache
- In progress: 2026-10-13T11:35:45.105846-04:00
- Metadata `logs/gap_improvements/G195_native_HG16_write_after_read_cache.json`; log `logs/gap_improvements/G195_native_HG16_write_after_read_cache.log`; patch `logs/gap_improvements/G195_native_HG16_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T11:35:45.645690-04:00; exit 0; runner elapsed 0.484 s.

### G195_native_HG16_incremental_propagators
- In progress: 2026-10-13T11:35:45.822472-04:00
- Metadata `logs/gap_improvements/G195_native_HG16_incremental_propagators.json`; log `logs/gap_improvements/G195_native_HG16_incremental_propagators.log`; patch `logs/gap_improvements/G195_native_HG16_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T11:35:45.907169-04:00; exit 0; runner elapsed 0.047 s.

### G195_native_HG16_start_rank_window
- In progress: 2026-10-13T11:35:46.073660-04:00
- Metadata `logs/gap_improvements/G195_native_HG16_start_rank_window.json`; log `logs/gap_improvements/G195_native_HG16_start_rank_window.log`; patch `logs/gap_improvements/G195_native_HG16_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T11:35:46.154363-04:00; exit 0; runner elapsed 0.031 s.

### G195_native_HG16_relaxation_bound
- In progress: 2026-10-13T11:35:46.319375-04:00
- Metadata `logs/gap_improvements/G195_native_HG16_relaxation_bound.json`; log `logs/gap_improvements/G195_native_HG16_relaxation_bound.log`; patch `logs/gap_improvements/G195_native_HG16_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T11:35:50.958108-04:00; exit 0; runner elapsed 4.609 s.

### G196_HG16_decode
- In progress: 2026-10-13T11:35:54.081431-04:00
- Metadata `logs/gap_improvements/G196_HG16_decode.json`; log `logs/gap_improvements/G196_HG16_decode.log`; patch `logs/gap_improvements/G196_HG16_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T11:45:54.562338-04:00; exit 0; runner elapsed 600.437 s.
  `[SearchEngine volume] iterations=2499 elapsed=592.12346909999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736833.3254815666 log2-remaining-fraction=-3114679.8382587526 eliminated-bits=3114679.8382587526 remaining-nodes=2499 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.359000000 budget=600.0`
  `[Program deadline] overshoot=0.239121`
  `[Program gap] primal-dual-integral=32.512315794005254 normalized=0.054187192990008758 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=11.655994415283203 samples=1617`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791905765.8237259 program-deadline=1791906354.1677637`
- First incumbent: 11.655962229 s, cost 39.3272; best cost 36.94059.

### G197_build_diag_noscreen
- In progress: 2026-10-13T11:46:10.610671-04:00
- Metadata `logs/gap_improvements/G197_build_diag_noscreen.json`; log `logs/gap_improvements/G197_build_diag_noscreen.log`; patch `logs/gap_improvements/G197_build_diag_noscreen.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:47:09.389224-04:00; exit 0; runner elapsed 58.750 s.

### G198_diag_noscreen_120s
- In progress: 2026-10-13T11:47:09.585315-04:00
- Metadata `logs/gap_improvements/G198_diag_noscreen_120s.json`; log `logs/gap_improvements/G198_diag_noscreen_120s.log`; patch `logs/gap_improvements/G198_diag_noscreen_120s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 120`
- Completed: 2026-10-13T11:49:10.066574-04:00; exit 0; runner elapsed 120.437 s.
  `[SearchEngine volume] iterations=156 elapsed=112.1037784 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737047.0573361092 log2-remaining-fraction=-3114466.10640421 eliminated-bits=3114466.10640421 remaining-nodes=156 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=120.375000000 budget=120.0`
  `[Program deadline] overshoot=0.220153`
  `[Program gap] primal-dual-integral=15.880359907132096 normalized=0.13233633255943414 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=11.705647468566895 samples=73`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791906441.3752379 program-deadline=1791906549.669615`
- First incumbent: 11.705622911 s, cost 39.3272; best cost 36.94059.

### G199_build_diag_planall2
- In progress: 2026-10-13T11:50:39.578135-04:00
- Metadata `logs/gap_improvements/G199_build_diag_planall2.json`; log `logs/gap_improvements/G199_build_diag_planall2.log`; patch `logs/gap_improvements/G199_build_diag_planall2.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:51:39.047539-04:00; exit 0; runner elapsed 59.422 s.

### G200_diag_planall2
- In progress: 2026-10-13T11:51:39.241679-04:00
- Metadata `logs/gap_improvements/G200_diag_planall2.json`; log `logs/gap_improvements/G200_diag_planall2.log`; patch `logs/gap_improvements/G200_diag_planall2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T11:52:05.003523-04:00; exit 0; runner elapsed 25.735 s.
  `[SearchEngine volume] iterations=1 elapsed=16.872855600000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.672000000 budget=25.0`
  `[Program deadline] overshoot=0.0999923`
  `[Program gap] primal-dual-integral=12.33949759545469 normalized=0.49357990381818762 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=11.79798698425293 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791906711.1240292 program-deadline=1791906724.3260734`
- First incumbent: 11.797955751 s, cost 39.3272; best cost 37.020939.

### G201_build_costmodel_atomic
- In progress: 2026-10-13T11:52:24.115717-04:00
- Metadata `logs/gap_improvements/G201_build_costmodel_atomic.json`; log `logs/gap_improvements/G201_build_costmodel_atomic.log`; patch `logs/gap_improvements/G201_build_costmodel_atomic.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:53:23.401492-04:00; exit 0; runner elapsed 59.250 s.

### G202_diag_costmodel_atomic
- In progress: 2026-10-13T11:53:23.589855-04:00
- Metadata `logs/gap_improvements/G202_diag_costmodel_atomic.json`; log `logs/gap_improvements/G202_diag_costmodel_atomic.log`; patch `logs/gap_improvements/G202_diag_costmodel_atomic.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T11:53:48.952629-04:00; exit 0; runner elapsed 25.328 s.
  `[SearchEngine volume] iterations=1 elapsed=16.9170452 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737798.8063347638 log2-remaining-fraction=-3113714.3574055554 eliminated-bits=3113714.3574055554 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.266000000 budget=25.0`
  `[Program deadline] overshoot=0.136771`
  `[Program gap] primal-dual-integral=12.32434275420745 normalized=0.492973710168298 final-gap=0.035322256628819437 final-incumbent=36.94059 final-bound=35.635764999999999 first-incumbent-elapsed=11.78266453742981 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791906815.4516706 program-deadline=1791906828.6690347`
- First incumbent: 11.782635927 s, cost 39.3272; best cost 37.020939.

### G203_build_diag_reduce_steps
- In progress: 2026-10-13T11:53:59.106988-04:00
- Metadata `logs/gap_improvements/G203_build_diag_reduce_steps.json`; log `logs/gap_improvements/G203_build_diag_reduce_steps.log`; patch `logs/gap_improvements/G203_build_diag_reduce_steps.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:54:58.583449-04:00; exit 0; runner elapsed 59.437 s.

### G204_diag_reduce_steps
- In progress: 2026-10-13T11:54:58.784728-04:00
- Metadata `logs/gap_improvements/G204_diag_reduce_steps.json`; log `logs/gap_improvements/G204_diag_reduce_steps.log`; patch `logs/gap_improvements/G204_diag_reduce_steps.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T11:55:24.501437-04:00; exit 0; runner elapsed 25.672 s.
  `[SearchEngine volume] iterations=0 elapsed=16.845902500000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.609000000 budget=25.0`
  `[Program deadline] overshoot=0.215301`
  `[Program gap] primal-dual-integral=12.523860524875042 normalized=0.50095442099500165 final-gap=0.037415960735085602 final-incumbent=37.020938999999998 final-bound=35.635764999999999 first-incumbent-elapsed=11.98895263671875 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791906910.8643165 program-deadline=1791906923.8753898`
- First incumbent: 11.988926649 s, cost 39.3272; best cost 37.020939.

### G205_build_diag_infos
- In progress: 2026-10-13T11:55:34.822329-04:00
- Metadata `logs/gap_improvements/G205_build_diag_infos.json`; log `logs/gap_improvements/G205_build_diag_infos.log`; patch `logs/gap_improvements/G205_build_diag_infos.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:56:35.288576-04:00; exit 0; runner elapsed 60.422 s.

### G206_diag_infos
- In progress: 2026-10-13T11:56:35.489008-04:00
- Metadata `logs/gap_improvements/G206_diag_infos.json`; log `logs/gap_improvements/G206_diag_infos.log`; patch `logs/gap_improvements/G206_diag_infos.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T11:57:01.384826-04:00; exit 0; runner elapsed 25.860 s.
  `[SearchEngine volume] iterations=0 elapsed=16.791884799999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.797000000 budget=25.0`
  `[Program deadline] overshoot=0.196208`
  `[Program gap] primal-dual-integral=12.510610898617806 normalized=0.5004244359447122 final-gap=0.037415960735085602 final-incumbent=37.020938999999998 final-bound=35.635764999999999 first-incumbent-elapsed=11.974874496459961 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907007.5513723 program-deadline=1791907020.5765252`
- First incumbent: 11.974847078 s, cost 39.3272; best cost 37.020939.

### G207_build_heuristic_costs_alloc
- In progress: 2026-10-13T11:57:15.765932-04:00
- Metadata `logs/gap_improvements/G207_build_heuristic_costs_alloc.json`; log `logs/gap_improvements/G207_build_heuristic_costs_alloc.log`; patch `logs/gap_improvements/G207_build_heuristic_costs_alloc.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:58:15.679221-04:00; exit 0; runner elapsed 59.875 s.

### G208_diag_heuristic_alloc
- In progress: 2026-10-13T11:58:15.881116-04:00
- Metadata `logs/gap_improvements/G208_diag_heuristic_alloc.json`; log `logs/gap_improvements/G208_diag_heuristic_alloc.log`; patch `logs/gap_improvements/G208_diag_heuristic_alloc.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T11:58:41.416512-04:00; exit 0; runner elapsed 25.500 s.
  `[SearchEngine volume] iterations=0 elapsed=16.7999112 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.422000000 budget=25.0`
  `[Program deadline] overshoot=0.316139`
  `[Program gap] primal-dual-integral=12.642070994951467 normalized=0.50568283979805873 final-gap=0.037415960735085602 final-incumbent=37.020938999999998 final-bound=35.635764999999999 first-incumbent-elapsed=12.110984563827515 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907108.0742755 program-deadline=1791907120.9633174`
- First incumbent: 12.110958099 s, cost 39.3272; best cost 37.020939.

### G209_build_diag_heuristic_phases
- In progress: 2026-10-13T11:58:56.748665-04:00
- Metadata `logs/gap_improvements/G209_build_diag_heuristic_phases.json`; log `logs/gap_improvements/G209_build_diag_heuristic_phases.log`; patch `logs/gap_improvements/G209_build_diag_heuristic_phases.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T11:59:56.851464-04:00; exit 0; runner elapsed 60.063 s.

### G210_diag_heuristic_phases
- In progress: 2026-10-13T11:59:57.044945-04:00
- Metadata `logs/gap_improvements/G210_diag_heuristic_phases.json`; log `logs/gap_improvements/G210_diag_heuristic_phases.log`; patch `logs/gap_improvements/G210_diag_heuristic_phases.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 25`
- Completed: 2026-10-13T12:00:22.603820-04:00; exit 0; runner elapsed 25.516 s.
  `[SearchEngine volume] iterations=0 elapsed=17.144378100000001 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737799.1282628586 log2-remaining-fraction=-3113714.0354774604 eliminated-bits=3113714.0354774604 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=25.453000000 budget=25.0`
  `[Program deadline] overshoot=0.294569`
  `[Program gap] primal-dual-integral=12.339682171254776 normalized=0.49358728685019104 final-gap=0.037415960735085602 final-incumbent=37.020938999999998 final-bound=35.635764999999999 first-incumbent-elapsed=11.796968460083008 samples=5`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907208.9289453 program-deadline=1791907222.1320076`
- First incumbent: 11.796937704 s, cost 39.3272; best cost 37.020939.

### G211_build_HG17_partition
- In progress: 2026-10-13T12:02:05.772219-04:00
- Metadata `logs/gap_improvements/G211_build_HG17_partition.json`; log `logs/gap_improvements/G211_build_HG17_partition.log`; patch `logs/gap_improvements/G211_build_HG17_partition.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T12:03:05.083931-04:00; exit 0; runner elapsed 59.281 s.

### G212_native_HG17
- In progress: 2026-10-13T12:03:05.295567-04:00
- Metadata `logs/gap_improvements/G212_native_HG17.json`; log `logs/gap_improvements/G212_native_HG17.log`; patch `logs/gap_improvements/G212_native_HG17.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T12:03:13.050446-04:00; exit 0; runner elapsed 7.719 s.

### G213_diag_HG17_45s
- In progress: 2026-10-13T12:03:13.231104-04:00
- Metadata `logs/gap_improvements/G213_diag_HG17_45s.json`; log `logs/gap_improvements/G213_diag_HG17_45s.log`; patch `logs/gap_improvements/G213_diag_HG17_45s.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T12:03:58.496249-04:00; exit 0; runner elapsed 45.218 s.
  `[SearchEngine volume] iterations=183 elapsed=36.930410899999998 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737042.3697075648 log2-remaining-fraction=-3114470.7940327544 eliminated-bits=3114470.7940327544 remaining-nodes=181 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.156000000 budget=45.0`
  `[Program deadline] overshoot=0.0221944`
  `[Program gap] primal-dual-integral=13.371686585036823 normalized=0.29714859077859607 final-gap=0.023881237413912498 final-incumbent=36.94059 final-bound=36.058402999999998 first-incumbent-elapsed=12.519931793212891 samples=65`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907405.8388503 program-deadline=1791907438.3189423`
- First incumbent: 12.519907951 s, cost 39.3272; best cost 36.94059.

### G214_build_diag_phases
- In progress: 2026-10-13T12:04:08.863663-04:00
- Metadata `logs/gap_improvements/G214_build_diag_phases.json`; log `logs/gap_improvements/G214_build_diag_phases.log`; patch `logs/gap_improvements/G214_build_diag_phases.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T12:05:08.092536-04:00; exit 0; runner elapsed 59.187 s.

### G215_diag_phases4
- In progress: 2026-10-13T12:05:08.285841-04:00
- Metadata `logs/gap_improvements/G215_diag_phases4.json`; log `logs/gap_improvements/G215_diag_phases4.log`; patch `logs/gap_improvements/G215_diag_phases4.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T12:05:53.727136-04:00; exit 0; runner elapsed 45.406 s.
  `[SearchEngine volume] iterations=167 elapsed=36.674216100000002 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737045.3682714526 log2-remaining-fraction=-3114467.7954688668 eliminated-bits=3114467.7954688668 remaining-nodes=165 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.344000000 budget=45.0`
  `[Program deadline] overshoot=0.0974185`
  `[Program gap] primal-dual-integral=14.913619714560552 normalized=0.33141377143467893 final-gap=0.020722841730465139 final-incumbent=36.94059 final-bound=36.175075999999997 first-incumbent-elapsed=14.198190689086914 samples=59`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907522.5713503 program-deadline=1791907553.3731859`
- First incumbent: 14.198164463 s, cost 39.3272; best cost 36.94059.

### G215_diag_phases8
- In progress: 2026-10-13T12:05:53.922404-04:00
- Metadata `logs/gap_improvements/G215_diag_phases8.json`; log `logs/gap_improvements/G215_diag_phases8.log`; patch `logs/gap_improvements/G215_diag_phases8.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 45`
- Completed: 2026-10-13T12:06:39.158110-04:00; exit 0; runner elapsed 45.188 s.
  `[SearchEngine volume] iterations=137 elapsed=36.953939400000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1737050.8550917541 log2-remaining-fraction=-3114462.3086485649 eliminated-bits=3114462.3086485649 remaining-nodes=135 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=45.125000000 budget=45.0`
  `[Program deadline] overshoot=0.00178075`
  `[Program gap] primal-dual-integral=17.283907857065412 normalized=0.38408684126812026 final-gap=0.02012731794484068 final-incumbent=36.94059 final-bound=36.197074999999998 first-incumbent-elapsed=16.635450601577759 samples=52`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907570.6442349 program-deadline=1791907599.0088141`
- First incumbent: 16.635420799 s, cost 39.3272; best cost 36.94059.

### G216_build_HG17
- In progress: 2026-10-13T12:07:12.947186-04:00
- Metadata `logs/gap_improvements/G216_build_HG17.json`; log `logs/gap_improvements/G216_build_HG17.log`; patch `logs/gap_improvements/G216_build_HG17.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test --no-lint --log-level INFO`
- Completed: 2026-10-13T12:08:12.174804-04:00; exit 0; runner elapsed 59.187 s.

### G217_oracle_HG17
- In progress: 2026-10-13T12:08:12.369306-04:00
- Metadata `logs/gap_improvements/G217_oracle_HG17.json`; log `logs/gap_improvements/G217_oracle_HG17.log`; patch `logs/gap_improvements/G217_oracle_HG17.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_planner_oracle`
- Completed: 2026-10-13T12:08:15.704248-04:00; exit 0; runner elapsed 3.297 s.
- Oracle verdict: 36/36 cases passed.

### G218_gap_unit_tests_HG17
- In progress: 2026-10-13T12:08:15.879938-04:00
- Metadata `logs/gap_improvements/G218_gap_unit_tests_HG17.json`; log `logs/gap_improvements/G218_gap_unit_tests_HG17.log`; patch `logs/gap_improvements/G218_gap_unit_tests_HG17.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_search_gap`
- Completed: 2026-10-13T12:08:15.958541-04:00; exit 0; runner elapsed 0.032 s.

### G219_native_HG17_private_layout
- In progress: 2026-10-13T12:08:16.124440-04:00
- Metadata `logs/gap_improvements/G219_native_HG17_private_layout.json`; log `logs/gap_improvements/G219_native_HG17_private_layout.log`; patch `logs/gap_improvements/G219_native_HG17_private_layout.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --private-layout`
- Completed: 2026-10-13T12:08:16.666270-04:00; exit 0; runner elapsed 0.500 s.

### G219_native_HG17_write_after_read_cache
- In progress: 2026-10-13T12:08:16.841769-04:00
- Metadata `logs/gap_improvements/G219_native_HG17_write_after_read_cache.json`; log `logs/gap_improvements/G219_native_HG17_write_after_read_cache.log`; patch `logs/gap_improvements/G219_native_HG17_write_after_read_cache.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --write-after-read-cache`
- Completed: 2026-10-13T12:08:16.921485-04:00; exit 0; runner elapsed 0.047 s.

### G219_native_HG17_incremental_propagators
- In progress: 2026-10-13T12:08:17.087965-04:00
- Metadata `logs/gap_improvements/G219_native_HG17_incremental_propagators.json`; log `logs/gap_improvements/G219_native_HG17_incremental_propagators.log`; patch `logs/gap_improvements/G219_native_HG17_incremental_propagators.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --incremental-propagators`
- Completed: 2026-10-13T12:08:17.676197-04:00; exit 0; runner elapsed 0.547 s.

### G219_native_HG17_start_rank_window
- In progress: 2026-10-13T12:08:17.856279-04:00
- Metadata `logs/gap_improvements/G219_native_HG17_start_rank_window.json`; log `logs/gap_improvements/G219_native_HG17_start_rank_window.log`; patch `logs/gap_improvements/G219_native_HG17_start_rank_window.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --start-rank-window`
- Completed: 2026-10-13T12:08:17.929704-04:00; exit 0; runner elapsed 0.047 s.

### G219_native_HG17_relaxation_bound
- In progress: 2026-10-13T12:08:18.098831-04:00
- Metadata `logs/gap_improvements/G219_native_HG17_relaxation_bound.json`; log `logs/gap_improvements/G219_native_HG17_relaxation_bound.log`; patch `logs/gap_improvements/G219_native_HG17_relaxation_bound.patch`
- Command: `C:\Users\aaron\CODING\tensor_graphs\tensor_graphs_cpp\test.exe --relaxation-bound`
- Completed: 2026-10-13T12:08:25.873342-04:00; exit 0; runner elapsed 7.734 s.

### G220_HG17_decode
- In progress: 2026-10-13T12:08:30.498192-04:00
- Metadata `logs/gap_improvements/G220_HG17_decode.json`; log `logs/gap_improvements/G220_HG17_decode.log`; patch `logs/gap_improvements/G220_HG17_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`

- Notes G160–G219 (2026-10-13):
  - **H-G14 cut DP** (`cutDpWindow`): exact dynamic program over a window's classes, parents first. A state is the set of required unprocessed classes; it keeps Pareto points of engine loads. A bisection on T first prunes alternatives with head + tail > T. Offline prototype (scratch `cut_dp.py`): exact (matches CP-SAT 1.1011 / 0.9731 / 0.6710 / 1.2743 / 1.2133) with ≤ 420 states. Natively plus bucket deduplication (`relaxationSignature`: all 16 decode buckets share one relaxation): G179 normalized 0.07237 (bound 34.958, first incumbent 11.78 s, bits 3114680.49).
  - **H-G15 dyadic windows**: windows of stride 2s are unions of two stride-s windows; tree DP over strides 1..64. Bound 35.34 (G182).
  - **H-G16 chain-coupled DP + parallel windows**: DP points also carry, for each required class, the longest chain from the target down to it, so a class may take an alternative only if that chain plus its head fits T (exact chain constraint along the selection). Exact enumeration and cut DP run in parallel over windows. Bound 35.636 at 0.93 s (G189). **G196: normalized 0.05419** (first incumbent 11.656 s, bits 3114679.84). Larger caps (20000 states / 32 points: +0.0015, slower) and no early stop (no gain) rejected.
  - **H-G17 window partitions**: each stride also evaluated shifted by half a stride; an interval DP (`bestWindowPartition`) picks the best partition of the chain into windows from any run (subsumes the dyadic tree). Bound 36.058 at 1.86 s (G213). Four phases: 36.175 at 3.17 s, eight: 36.197 at 6.0 s (too slow for the first-incumbent gate; not used).
  - Primal trials (no gain): local search without the max-load screen (G198: same 37.02 local optimum, 70k schedules); the incumbent keeps the CPU busy for the whole makespan, so dispatch order cannot help.
  - Startup diagnostics (G199–G210, instrumentation removed): per-bucket reduction is dominated by `computeENodeHeuristicCosts`' optimistic-summary pass (dense 123k-class bitsets per candidate; 0.3 s alone, ~1 s each in parallel). A semantics-preserving rewrite was deferred (its scratch summaries are reused without clearing, so changing the representation could change brancher estimates). An atomic init flag in `CostModel::predict` and hoisted vectors did not help; reverted.
  - Brute force (G212/G219): 18000 cut-DP checks (exact enumeration disabled) never above the minimum makespan or the exact value; 12000 window partitions over strides 1–4 with every phase valid. Oracle 36/36 (G217), gap unit tests G218, native G219 PASS.
- Completed: 2026-10-13T12:18:31.085279-04:00; exit 0; runner elapsed 600.547 s.
  `[SearchEngine volume] iterations=2439 elapsed=591.97418849999997 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736836.0892019644 log2-remaining-fraction=-3114677.0745383548 eliminated-bits=3114677.0745383548 remaining-nodes=2437 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.469000000 budget=600.0`
  `[Program deadline] overshoot=0.33421`
  `[Program gap] primal-dual-integral=26.744280470886419 normalized=0.044573800784810699 final-gap=0.023881237413912498 final-incumbent=36.94059 final-bound=36.058402999999998 first-incumbent-elapsed=12.635338544845581 samples=1592`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791907723.2203567 program-deadline=1791908310.585043`
- First incumbent: 12.635313749 s, cost 39.3272; best cost 36.94059.

### G221_HG17_confirm1_decode
- In progress: 2026-10-13T12:18:31.272629-04:00
- Metadata `logs/gap_improvements/G221_HG17_confirm1_decode.json`; log `logs/gap_improvements/G221_HG17_confirm1_decode.log`; patch `logs/gap_improvements/G221_HG17_confirm1_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T12:28:31.666057-04:00; exit 0; runner elapsed 600.344 s.
  `[SearchEngine volume] iterations=2459 elapsed=592.00240340000005 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.1829204143 log2-remaining-fraction=-3114677.9808199052 eliminated-bits=3114677.9808199052 remaining-nodes=2457 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.281000000 budget=600.0`
  `[Program deadline] overshoot=0.1377`
  `[Program gap] primal-dual-integral=26.513136081738342 normalized=0.04418856013623057 final-gap=0.023881237413912498 final-incumbent=36.94059 final-bound=36.058402999999998 first-incumbent-elapsed=12.405710697174072 samples=1606`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791908323.7639091 program-deadline=1791908911.358227`
- First incumbent: 12.405682087 s, cost 39.3272; best cost 36.94059.

### G222_HG17_confirm2_decode
- In progress: 2026-10-13T12:28:31.856032-04:00
- Metadata `logs/gap_improvements/G222_HG17_confirm2_decode.json`; log `logs/gap_improvements/G222_HG17_confirm2_decode.log`; patch `logs/gap_improvements/G222_HG17_confirm2_decode.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600`
- Completed: 2026-10-13T12:38:32.436351-04:00; exit 0; runner elapsed 600.531 s.
  `[SearchEngine volume] iterations=2454 elapsed=592.03528040000003 total-domain-bits=4851513.1637403192 remaining-domain-bits=1736835.4030054226 log2-remaining-fraction=-3114677.7607348966 eliminated-bits=3114677.7607348966 remaining-nodes=2452 exhausted=false incomplete-branches=0`
  `[Program evaluation] elapsed=600.469000000 budget=600.0`
  `[Program deadline] overshoot=0.315509`
  `[Program gap] primal-dual-integral=26.708362878448408 normalized=0.044513938130747346 final-gap=0.023881237413912498 final-incumbent=36.94059 final-bound=36.058402999999998 first-incumbent-elapsed=12.60619068145752 samples=1605`
  `[SearchEngine] New best cost: 39.3272 at iteration 0 source=root-heuristic event-epoch=1791908924.5472753 program-deadline=1791909511.9411089`
- First incumbent: 12.606166363 s, cost 39.3272; best cost 36.94059.

- G220/G221/G222 H-G17: 0.04457 / 0.04419 / 0.04451 (first incumbent 12.635 / 12.406 / 12.606 s; bits 3114677.07 / 3114677.98 / 3114677.76; bound 36.058403; first/deadline cost 39.3272 / 36.94059). Patches identical (98382B97…).
- **Gate verdict H-G17: ACCEPTED as optimization O-G5** (2026-10-13). (1) oracle G217 36/36 with certified-bound validity, gap unit tests G218, native G219 pass; (2) median 0.04451 vs O-G4 0.13140, improvement 0.087 ≫ spread 9.0e-4; (3) bits ≥ 3114375.50 in every run; (4) first incumbent median 12.606 s ≤ 12.196 + 1, slowest 12.635 ≤ 12.362 + 2; (5) first-incumbent cost 39.3272 and deadline cost 36.94059 unchanged. **Goal (median ≤ 0.05) met.**
