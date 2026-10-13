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

Accepted-implementation reference for the gate: median normalized 0.99164973 (spread 2.7e-5), median first incumbent 135.456 s (slowest 136.025 s), eliminated bits 3114375.5045604468, first/deadline cost 75.0097 / 75.009705.

## Resume checkpoint (current)
- Accepted implementation: branch `faster-search` at `20bda2dd` (O9: 3,114,375.50 eliminated bits) plus uncommitted gap-measurement infrastructure: `core/plan/selector.hpp` (minOpenLowerBound), `core/plan/search_engine.hpp` (certifiedBound, sampleSearchGap, TG_SEARCH_GAP / TG_SEARCH_GAP_INTERVAL), `bindings/planner_oracle.hpp` + `tests/test_planner_oracle.py` (certified-bound validity), `tests/search_gap.py`, `tests/test_search_gap.py`, `tests/test_gemma_output.py` (supervisor gap report), `tests/search_experiment.py`, `docs/core/search_gap.md`, `docs/agents/improve_gap_integral.md`. Exact source: `logs/gap_improvements/G008_diag_ab_sampling_off.patch`. Extension and test.exe built from it (G007).
- No running processes. Baseline measured (G004-G006); no candidate evaluated yet.
- Next concrete step: H-G1, a sound root engine-load lower bound. Today EngineWorkloadPropagator charges only mandatory work, so the root bound is 0.528 while R166 (CP-SAT) proves 30.36 per decode bucket. Any bound must be validated by the oracle (certified-bound check) with new adversarial cases (optional work, multi-engine alternatives, zero-weight bucket, caches). It must not add measurable time before the first incumbent: compute it once at the root, or incrementally.
- Further candidates: cost-aware first dive guided by the relaxation; LNS around the incumbent after the first plan; reduced-cost style fixing once the gap is small (see docs/reviews/ortools-cpsat-lessons-2026-10-12.md).

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
