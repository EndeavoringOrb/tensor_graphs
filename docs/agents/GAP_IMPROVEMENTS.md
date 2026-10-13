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

Accepted-implementation reference for the gate (**H-G4b, accepted 2026-10-12**): median normalized 0.22517474 (runs 0.22507279 / 0.22517474 / 0.22634102, spread 1.27e-3), median first incumbent 25.997 s (slowest 26.864 s), eliminated bits ≥ 3114673.71 (minimum over the three runs; gate threshold from the original baseline 3114375.5045604468), first/deadline cost 39.3272 / 37.020939.

Original gap baseline (for reference): median normalized 0.99164973, median first incumbent 135.456 s, eliminated bits 3114375.5045604468, costs 75.0097 / 75.009705.

## Resume checkpoint (current)
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
