# Prefill+decode first plan within 30 s: progress (box)

Task: follow `docs/agents/targets/prefill_decode_incumbent_seq_len/target.md`. Runs are appended by `tests/search_experiment.py --record prefill_decode_incumbent_seq_len:box`; logs, metadata and source patches live in `logs/targets/prefill_decode_incumbent_seq_len/`.

## Fixed evaluation contract

- Command: `.venv/bin/python tests/search_experiment.py --record prefill_decode_incumbent_seq_len:box <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len <N> --ignore-checkpoints --run-dir logs/gemma_output_runs/<run_id>`
- Build: `.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Hardware: Linux box, 16 threads, 31 GB RAM, Quadro M4000 (CUDA), OpenCL off.
- Base revision: `abde0a8bfaaba261136ad2ad1e6217957bee61b0`; initially clean worktree.

## Comparison table

Accepted implementation: **C037 scaling**, reference `I619_c037_n47` (patch under target logs), built by `I618_c037_costkey_build`. N*=**47**, first incumbent **28.194377 s**, compile **29.736 s**; all ninety-five logits graphs exact-checked (`ok=true`); confirmations I620 (28.173 s, 29.642 s compile) and I621 (28.276 s, 29.742 s). Prior N*=46 regression `I622_c037_n46_regression` passes (27.612 s, 0.5 s earlier than C036). Gates I623-I631 pass (oracle 48/48, search gap, seven native suites).

| Run | Change | N | ok | first incumbent s | final incumbent | compile s | Verdict |
|---|---|---|---|---|---|---|---|
| I003_baseline_n1 | unchanged base | 1 | true | 11.186343 | 14.694064 | 28.106 | baseline reference |
| I004_baseline_n2 | unchanged base | 2 | false | none | none | 30.112 | budget miss during root propagation |
| I005_baseline_n3 | unchanged base | 3 | false | none | none | 30.104 | budget miss during reduction |
| I011_validate_first_n1 | C001 validate before local search | 1 | true | 8.992408 | 15.287892 | 28.092 | accepted speed change |
| I012_validate_first_n1_confirm1 | C001 identical source/build | 1 | true | 9.238490 | 15.287892 | 28.056 | confirmation 1 passes |
| I013_validate_first_n1_confirm2 | C001 identical source/build | 1 | true | 9.128998 | 15.287892 | 28.104918956756592 | confirmation 2 passes |
| I027_deferred_primal_n2 | C002 deferred bound / streaming primal | 2 | false | unavailable | not verified | 30.102 | miss; arena/write incomplete |
| I028_deferred_primal_n1 | C002 | 1 | true | 6.093158 | 86.653038 | 27.864 | speed candidate; superseded |
| I033_lazy_regions_n2 | C003 lazy regions | 2 | true | 25.086478 | 91.342201 | 29.827 | passes; superseded for deadline handling |
| I034_lazy_regions_n2_confirm1 | C003 identical build | 2 | true | 25.050185 | 91.342201 | 29.848 | passes; superseded |
| I039_preparation_deadline_n2 | C004 interruptible preparation | 2 | true | 25.615177 | 91.342201 | 29.823 | passes; needs safer extraction margin |
| I040_safe_margin_n2 | C005 | 2 | true | 24.953302 | 91.342201 | 26.239 | accepted scaling change |
| I041_safe_margin_n2_confirm1 | C005 | 2 | true | 25.099681 | 91.342201 | 26.352 | confirmation 1 passes |
| I042_safe_margin_n2_confirm2 | C005 | 2 | true | 25.098190 | 91.342201 | 26.345 | confirmation 2 passes |
| I043_safe_margin_n1_regression | C005 | 1 | true | 5.224945 | 86.653038 | 25.928 | prior N*=1 no regression |
| I050_accepted_frontier_n3 | accepted C005 frontier | 3 | false | none | none | 30.128 | budget miss during reduction |
| I054_post_reduction_estimates_n3 | C006 single heuristic pass | 3 | false | none | none | 30.115 | miss, search now reached |
| I055_post_reduction_estimates_n2 | C006 | 2 | true | 22.443355 | 91.342201 | 26.451 | accepted speed change |
| I056_post_reduction_estimates_n2_confirm1 | C006 identical source/build | 2 | true | 22.392718 | 91.342201 | 26.391 | confirmation passes |
| I057_post_reduction_estimates_n2_confirm2 | C006 identical source/build | 2 | true | 22.230959 | 91.342201 | 26.379 | confirmation passes |
| I070_incremental_bucket_saturation_n3 | C008 cumulative partial state | 3 | false | none | none | 30.080 | later buckets rescanned a growing graph; discarded |
| I072_shared_partial_bucket_closure_n3 | C007 shared partial closure | 3 | false | none | none | 30.106 | search began at 28.735 s; budget miss |
| I073_shared_partial_bucket_closure_n2 | C007 | 2 | true | 18.343996 | 91.380814 | 26.162 | accepted speed change; all five logits graphs pass |
| I074_shared_partial_bucket_closure_n2_confirm1 | C007 identical source/build | 2 | true | 18.403718 | 91.380814 | 26.190 | confirmation passes |
| I075_shared_partial_bucket_closure_n2_confirm2 | C007 identical source/build | 2 | true | 18.348832 | 91.380814 | 26.179 | confirmation passes |
| I085_region_family_saturation_n3 | boundary-family union | 3 | false | none | none | 30.140 | four closures; saturation 21.42 s |
| I087_dirty_family_saturation_n3 | input-region family union | 3 | false | none | none | 30.140 | four closures; saturation 28.18 s |
| I089_dirty_origin_family_n3 | origin-only family union | 3 | false | none | none | 30.139 | two closures; saturation 25.81 s |
| I092_shared_reconciliation_n3 | C007 plus unique-state reconciliation | 3 | false | none | none | 30.116 | unchanged; search start 28.88 s |
| I094_shared_primal_n2 | memoized selection/covering/scheduling for equal bucket relaxations | 2 | true | 18.447397 | 91.380814 | 26.3 | no tie-break gain vs C007; discarded |
| I095_shared_primal_n3 | same as I094 | 3 | false | none | none | 30.105 | search start 28.834 s; miss unchanged vs C007 |
| I097_c007_bucket_timing_n3 | C007 with TG_TIME_BUCKETS | 3 | false | none | none | 30.115 | diagnostic: partial shared closure 20.21 s; saturation 24.17 s |
| I098_c007_rule_timing_n3 | C007 with TG_TIME_RULES | 3 | false | none | none | 30.115 | diagnostic: 78 rounds; rebuild 12.24 s, sweeps 6.13 s, fusion filter 1.54 s |
| I100_reuse_rebuild_hash_n3 | retain scratch hash-map buckets across EGraph rebuild rounds | 3 | false | none | none | 30.117 | saturation 24.409 s, search start 28.943 s; slower, discarded |
| I103_parallel_rebuild_n3 | shard EGraph rebuild scans across 16 threads; serial merge application | 3 | false | none | none | 30.117 | saturation 22.921 s, search start 27.579 s |
| I104_parallel_rebuild_n2 | same as I103 | 2 | true | 18.474151 | 17.500534 | 26.2 | exact pass; not earlier than C007 |
| I113_parallel_shared_state_n3 | C009: parallel rebuild plus single shared-state owner through reduction | 3 | false | none | none | 30.119 | search start 24.856 s; incumbent validation after deadline, no plan write |
| I114_parallel_shared_state_n2 | C009 before final reconciliation audit | 2 | true | 17.701276 | 17.500534 | 26.412 | exact pass; superseded by I126-I128 |
| I115_parallel_shared_state_n2_confirm1 | same pre-audit source/build | 2 | true | 17.545292 | 17.500534 | 26.438 | confirmation passes |
| I116_parallel_shared_state_n2_confirm2 | same pre-audit source/build | 2 | true | 17.538002 | 17.500534 | 26.406 | confirmation passes |
| I126_c009_n2_after_reconcile_fix | C009 final reconciliation audit | 2 | true | 17.735929 | 17.500534 | 26.535 | accepted speed change; all five graphs exact |
| I127_c009_n2_after_reconcile_fix_confirm1 | C009 identical source/build | 2 | true | 17.814731 | 17.500534 | 26.570 | confirmation passes |
| I128_c009_n2_after_reconcile_fix_confirm2 | C009 identical source/build | 2 | true | 17.755571 | 17.500534 | 26.531 | confirmation passes |
| I137_c009_n3_after_reconcile_fix | C009 final source/build | 3 | false | none in budget | not written | 30.072 | search starts 24.740 s; target miss |
| I139_c009_share_reduced_storage_n3 | freeze reduced representative EGraph storage before alias copies | 3 | false | none | not written | 26.379 | reached search at 23.828 s, then crashed during root primal; reverted |
| I141_fast_offset_completion_build / I142_fast_offset_completion_n3 / I143_fast_offset_completion_n2 | batch offset completion with one final propagation, then sequential fallback | 2/3 | pass/miss | 18.438 / none | 17.500534 / not written | 26.5 / 30.1 | N=2 slower than C009; N=3 conflicts on producer-consumer buffer overlap; reverted |
| I144_share_full_bucket_build / I145_share_full_bucket_n2 / I146_share_full_bucket_n3 | retain one full saturation graph through reconciliation, then move it into the full bucket | 2/3 | pass/miss | 17.831 / none | 17.500534 / not written | 26.328 / 30.095 | N=2 exact pass; N=3 search starts 24.870 s, no improvement over C009; reverted |
| I149_share_search_egraphs_build2 / I150_share_search_egraphs_n3 / I151_share_search_egraphs_n2 | share reduced EGraph ownership among buckets with the same reduced-state representative | 2/3 | pass/miss | 17.190 / none | 17.500534 / not written | 26.336 / 30.182 | accepted speed at N=2; exact pass, 0.545 s earlier than C009; N=3 search starts 24.202 s but misses plan write |
| I160_c011_n2_regression | C011 lazy offset completion, zero-alloc queues, shared reachable cids | 2 | true | 16.435553 | 17.500534 | 29.921 | prior N*=2 no regression (faster by 0.755 s) |
| I161_c011_n3 | C011 | 3 | true | 27.956848 | 109.448563 | 29.730 | accepted scaling change; all seven logits graphs pass |
| I162_c011_n3_confirm1 | C011 identical source/build | 3 | true | 27.624890 | 109.448563 | 29.549 | confirmation 1 passes |
| I163_c011_n3_confirm2 | C011 identical source/build | 3 | true | 27.595142 | 109.448563 | 29.471 | confirmation 2 passes |
| I164_c011_n4 | C011 frontier | 4 | false | none | none | 30.092 | saturation alone takes 28.66 s; budget miss |
| I169_c012_n4 | C012 in-sweep propagation, search deadline buffer, 4.5s margin | 4 | true | 17.712358 | 17.309278 | 28.542 | accepted scaling change; all nine logits graphs pass |
| I170_c012_n4_confirm1 | C012 identical source/build | 4 | true | 17.791255 | 17.309278 | 28.680 | confirmation 1 passes |
| I171_c012_n4_confirm2 | C012 identical source/build | 4 | true | 17.823853 | 17.309278 | 28.272 | confirmation 2 passes |
| I172_c012_n3_regression | C012 | 3 | true | 13.067457 | 109.652542 | 26.960 | prior N*=3 no regression (faster by 14.889 s) |
| I173_c012_n5 | C012 N=5 initial | 5 | true | 22.635956 | 99.901581 | 29.568 | passes, but margin tight (0.43s headroom) |
| I174_c012_n5_confirm1 | C012 identical source/build | 5 | true | 22.531723 | 99.901581 | 29.606 | passes, 0.39s headroom |
| I175_c012_n5_confirm2 | C012 identical source/build | 5 | false | 22.5 (est) | none | 30.037 | supervisor killed at 30.037 s by 37 ms; 4.5s margin too tight |
| I177_c013_n5 | pubsetbuf stack buffer trial | 5 | false | 23.029263 | none | 29.941 | stream buffer destroyed before file flush, plan.bin 0 bytes; reverted |
| I179_c013_n5 | C013 O(1) buffer index in buildCompiledGraph, 5.5s search margin | 5 | true | 22.611045 | 99.901581 | 29.085 | accepted scaling change; all eleven logits graphs pass |
| I180_c013_n5_confirm1 | C013 identical source/build | 5 | true | 22.546773 | 99.901581 | 29.069 | confirmation 1 passes |
| I181_c013_n5_confirm2 | C013 identical source/build | 5 | true | 22.771193 | 99.901581 | 29.816 | confirmation 2 passes |
| I182_c013_n4_regression | C013 | 4 | true | 17.733594 | 17.309278 | 27.604 | prior N*=4 no regression (compile faster by 0.94s) |
| I190_c014_after_prop_n6 | C014 disjoint offset proposal after propagation | 6 | false | none (after deadline) | 100.008881 | 30.359 | first tryPlan 1.8 s (was 4.1 s) but duplicate validations push search to 27.8 s; miss |
| I194_c015_n6 | C015 skip re-validating tried plans; stop validating past deadline | 6 | false | ~25.2 | 100.008881 | 30.324 | search done 24.98 s; plan write still too slow |
| I198_c015_n6 | C015 + single persistCache | 6 | true | ~25.3 | 100.008881 | 29.605 | first N=6 pass, 0.4 s margin |
| I202_c015_compact_views_n6 | C015 + compiled graphs keep only referenced views | 6 | true | ~25.0 | 100.008881 | 26.939 | plan 199 MB -> 6 MB |
| I204_c015_n7 | C015 + background free of planner state | 7 | false | ~29.8 | 24.6443 | 30.136 | miss: saturation 16.44 s, search start 24.13 s |
| I205_c015_n6 | C015 final | 6 | true | 25.320655 | 100.008881 | 26.554 | **accepted scaling change, N*=6** |
| I228_c016_n7 | C016 move compaction, background compaction free, final-rebuild hashcons skip, addOp probe, calloc arena | 7 | true | 29.027618 | 24.644283 | 29.889 | first N=7 pass, 0.11 s margin |
| I232_c016_n7 | C016 + fused-kernel scatter/cache memo + parallel propagation state | 7 | true | 27.977648 | 24.644283 | 28.834 | **accepted scaling change, N*=7** |
| I233_c016_n6_regression | C016 | 6 | true | 23.670287 | 100.008881 | 24.791 | prior N* regression passes |
| I245_c017_n8 | C017 completion bitmasks + partial-group saturation on a worker thread (one group) | 8 | true | 26.113307 | 115.861198 | 26.898 | **accepted scaling change, N*=8** |
| I246_c017_n7_regression | C017 | 7 | true | ~23.2 | — | 25.262 | prior N* regression passes |
| I260_c018_n8 | C018 parallel enode infos + parallel base-equality reconciliation + group owner fix (one group) | 8 | true | 26.095996 | 115.861198 | 26.850 | not accepted: 17 ms earlier than C017 is run noise |
| I263_c018_g9_n9_diag | C018 + I261/I262 (parallel per-bucket propagator setup, pooled group reduction), TG_PARTIAL_GROUPS=9 | 9 | false | none | none | 28.588 | diagnostic: search start 15.08 s, root propagation 7.62 s, no consistent primal plan |
| I270_c019_g9_n9_diag | C019 parallel root setup (reachability trees, WAR schedules, input-prune tables, rank windows), G=9 | 9 | false | none | none | 28.960 | diagnostic: root propagation 5.06 s; bucket 18 relaxation root infeasible |
| I273_c019_g8_n8_diag | same, G=8 | 8 | false | none | none | 25.531 | diagnostic: buckets 14/15 relaxation root infeasible; traced to compaction emptying classes (I276-I279) |
| I280_standin_g8_n8_diag | + compaction stand-in for pruned hashcons representatives, G=8 | 8 | true | 24.322883 | 235.771088 | 25.820 | diagnostic pass (env override) |
| I283_c019_n9 | **C019**: parallel root setup + compaction stand-in fix + default one saturation group per distinct output region | 9 | true | 26.304730 | 27.161423 | 27.986 | **accepted scaling change, N*=9**; all 19 graphs exact |
| I284_c019_n9_confirm1 | C019 identical source/build | 9 | true | 26.070413 | 27.161423 | 27.755 | confirmation 1 passes |
| I285_c019_n9_confirm2 | C019 identical source/build | 9 | true | 26.161172 | 27.161423 | 27.778 | confirmation 2 passes |
| I286_c019_n8_regression | C019 | 8 | true | 23.990621 | 235.771088 | 26.291 | prior N*=8 no regression (2.1 s earlier than C017) |
| I310_c019_n10 | C019 frontier | 10 | false | after deadline | none | 30.251 | miss: incumbent 4.8 s past search deadline (tryPlan 4.43 s), killed during write |
| I312_c020_n10 | C020 step 1: bound-only propagators skipped while completing a trial plan | 10 | false | after deadline | none | 30.238 | tryPlan 4.43 -> 3.71 s; still killed during write |
| I314_c020_n10 | + relaxations freed in the background, parallel signatures, dynamic bucket scheduling | 10 | true | 28.336388 | 28.233677 | 29.286 | first N=10 pass, 0.7 s headroom |
| I317_c020_n10 | **C020** + heuristic DP by Knuth's algorithm (reduction 6.2 -> 3.0 s) | 10 | true | 25.032133 | 28.233677 | 26.025 | passes |
| I318_c020_n12 | C020 | 12 | false | after deadline | none | 30.286 | miss: graphs built at 29.3 s, killed during write |
| I319_c020_n11 | C020 | 11 | true | 27.437513 | 27.965914 | 28.520 | **accepted scaling change, N*=11**; all 23 graphs exact |
| I320_c020_n11_confirm1 | C020 identical source/build | 11 | true | 27.379168 | 27.965914 | 28.448 | confirmation 1 passes |
| I321_c020_n11_confirm2 | C020 identical source/build | 11 | true | 27.195079 | 27.965914 | 28.493 | confirmation 2 passes |
| I322_c020_n9_regression | C020 | 9 | true | 22.922123 | 27.161423 | 24.868 | prior N*=9 no regression (3.4 s earlier than C019) |
| I333_c021_n12 | C021 step 1: one shared ancestor walk per class in PearceKelly, two tasks per bucket in ensurePropagationState | 12 | false | after deadline | none | 30.265 | root propagation 6.11 -> 5.81 s; graphs built 29.33 s, killed during write |
| I335_c021_n12 | **C021** + parallel view-user maps + glibc malloc tuning (`malloc_tuning.hpp`: mmap threshold 32 MiB, no trim, 256 MiB top pad) | 12 | true | 25.744363 | 28.246435 | 26.737 | **accepted scaling change, N*=12**; all 25 graphs exact; peak RSS 8.0 GB |
| I336_c021_n14 | C021 | 14 | false | none | none | 30.057 | miss: search start 17.06 s, root propagation 5.38 s, first trial plan rejected |
| I337_c021_n13 | C021 | 13 | true | 28.497580 | 158.328857 | 29.622 | frontier pass, 0.4 s headroom (first trial plan rejected: ViewSelectOffset conflict, second validated); not claimed |
| I338_c021_n12_confirm1 | C021 identical source/build | 12 | true | 25.919753 | 28.246435 | 27.041 | confirmation 1 passes |
| I339_c021_n12_confirm2 | C021 identical source/build | 12 | true | 26.147549 | 28.246435 | 27.162 | confirmation 2 passes |
| I340_c021_n11_regression | C021 | 11 | true | 24.162520 | 27.965914 | 25.521 | prior N*=11 no regression (3.3 s earlier than C020) |
| I351_c022_n13 | C022 step 1: parallel_for callers block after a 50 us spin instead of yielding | 13 | true | 28.271269 | 158.328857 | 29.389 | partial saturation 9.12 -> 8.98 s |
| I353_c022_n13 | + skip inf-predicted trial plans, ParentRemoval skipped on fixed complete plans | 13 | true | 27.980051 | 158.328857 | 29.053 | |
| I355_c022_n13 | + reconciliation rebuilds only changed graphs after round 1 | 13 | true | 27.573285 | 158.328857 | 28.721 | reconcile 1.49 -> 1.12 s |
| I357_c022_n13 | **C022** + bucket variables built in parallel (two-phase, same ids) | 13 | true | 26.424315 | 158.328857 | 27.681 | search setup 1.0 -> 0.48 s |
| I358_c022_n14 | C022 | 14 | true | 27.899543 | 649.000000 | 29.097 | **accepted scaling change, N*=14**; all 29 graphs exact |
| I359_c022_n14_confirm1 | C022 identical source/build | 14 | true | 27.839048 | 649.000000 | 28.979 | confirmation 1 passes |
| I360_c022_n14_confirm2 | C022 identical source/build | 14 | true | 28.071328 | 649.000000 | 29.350 | confirmation 2 passes |
| I361_c022_n12_regression | C022 | 12 | true | 24.112437 | 28.246435 | 25.548 | prior N*=12 no regression (1.6 s earlier than C021) |
| I372_c023_n16 | **C023**: incremental congruence closure in `EGraph::rebuild` (only e-nodes with a non-canonical child are re-hashed) | 16 | true | 26.240398 | 32.487766 | 27.640 | first N=16 pass |
| I373_c023_n17 | C023 | 17 | true | 28.296102 | 210.009293 | 29.765 | **accepted scaling change, N*=17**; all 35 graphs exact |
| I374_c023_n17_confirm1 | C023 identical source/build | 17 | true | 28.133551 | 210.009293 | 29.428 | confirmation 1 passes |
| I375_c023_n17_confirm2 | C023 identical source/build | 17 | true | 28.486986 | 210.009293 | 29.850 | confirmation 2 passes |
| I376_c023_n14_regression | C023 | 14 | true | 23.418888 | — | 25.369 | prior N*=14 no regression (4.5 s earlier than C022) |
| I387_c024_n18 | C024 step 1: propagator work on unselected classes trimmed (see C024 notes) | 18 | false | after deadline | none | 30.056 | miss: 18 regions on 16 threads merged two groups, partial saturation 9.2 s |
| I389_c024_n18 | **C024** + one saturation group per output region up to 4x hardware threads | 18 | true | 27.985398 | 223.352 | 29.511 | **accepted scaling change, N*=18**; all 37 graphs exact |
| I390_c024_n18_confirm1 | C024 identical source/build | 18 | true | 28.083753 | 223.352 | 29.606 | confirmation 1 passes |
| I391_c024_n18_confirm2 | C024 identical source/build | 18 | true | 27.887367 | 223.352 | 29.439 | confirmation 2 passes |
| I392_c024_n17_regression | C024 | 17 | true | 26.135702 | — | 27.531 | prior N*=17 no regression (2.2 s earlier than C023) |
| I403_c025_n19 | C025 step 1: StartPrecedence reschedules starts only after a real selection change; PearceKelly dense parent lists | 19 | true | 28.484578 | — | 30.001 | passes with no margin; not claimed |
| I405_c025_n19 | + primal heuristic forbids cached views of differently cached bases (first trial plan no longer rejected) | 19 | true | 28.168934 | 348.603943 | 29.759 | passes, 0.24 s margin |
| I407_c025_n19 | **C025** + lazy relaxation signatures, parallel alias check | 19 | true | 27.796348 | 348.603943 | 29.222 | **accepted scaling change, N*=19**; all 39 graphs exact |
| I408_c025_n19_confirm1 | C025 identical source/build | 19 | true | 27.833079 | 348.603943 | 29.265 | confirmation 1 passes |
| I409_c025_n19_confirm2 | C025 identical source/build | 19 | true | 27.543056 | 348.603943 | 29.019 | confirmation 2 passes |
| I410_c025_n18_regression | C025 | 18 | true | 26.051405 | — | 27.708 | prior N*=18 no regression (1.9 s earlier than C024) |
| I420_c026_alloc_build | C026 build (first attempt) | — | — | — | — | — | build failed the repository lint (ENode reference held across e-graph mutation in materializeFusedCandidates); I421_c026_n20 ran on an unrecorded build of that source and is superseded |
| I4223_c026_n20 | **C026**: fewer allocations (interned ENode names/origins/shapes/strides/engines, reused buffers; see C026 notes), built by I422 | 20 | true | 26.637883 | 443.285889 | 28.184 | first N=20 pass (run id typo for I423) |
| I4224_c026_n21 | C026 | 21 | true | 27.701424 | 420.521912 | 29.425 | **accepted scaling change, N*=21**; all 43 graphs exact (run id typo for I424) |
| I425_c026_n21_confirm1 | C026 identical source/build | 21 | true | 27.577033 | 420.521912 | 29.235 | confirmation 1 passes |
| I426_c026_n21_confirm2 | C026 identical source/build | 21 | true | 27.586768 | 420.521912 | 29.209 | confirmation 2 passes |
| I427_c026_n19_regression | C026 | 19 | true | 25.460740 | — | 26.941 | prior N*=19 no regression (2.3 s earlier than C025) |
| I438_c027_n22 | **C027**: EClass shape/strides interned (`InternedVector`) | 22 | true | 27.330481 | 446.980042 | 29.064 | **accepted scaling change, N*=22**; all 45 graphs exact |
| I439_c027_n23 | C027 frontier | 23 | false | none | none | 30.060 | budget miss |
| I440_c027_n22_confirm1 | C027 identical source/build | 22 | true | 27.754192 | 446.980042 | 29.465 | confirmation 1 passes |
| I441_c027_n22_confirm2 | C027 identical source/build | 22 | true | 27.790077 | 446.980042 | 29.576 | confirmation 2 passes |
| I442_c027_n21_regression | C027 | 21 | true | 26.444695 | — | 28.137 | prior N*=21 no regression (1.3 s earlier than C026) |
| I453_c028_n26 | C028 step 1 (I452 build): first incumbent from the root heuristic on unpropagated domains, validated in a plan-restricted state | 26 | true | 23.364295 | 652.285339 | 24.654 | passes |
| I456_c028_n30 | same | 30 | true | 28.108505 | 45.403740 | 29.769 | passes (restricted propagation 4.9 s) |
| I458_c028_n32 | **C028** + StartUnique/WriteAfterReadStart batched per bucket while completing a plan (I457 build) | 32 | true | 26.034607 | 43.419163 | 27.698 | passes |
| I459_c028_n36 | C028 | 36 | false | none | none | 30.089 | miss: search done 27.45 s, killed while writing |
| I460_c028_n34 | C028 | 34 | true | 27.794994 | 605.776917 | 29.575 | **accepted scaling change, N*=34**; all 69 graphs exact |
| I461_c028_n34_confirm1 | C028 identical source/build | 34 | true | 27.899348 | 605.776917 | 29.594 | confirmation 1 passes |
| I462_c028_n34_confirm2 | C028 identical source/build | 34 | true | 28.038950 | 605.776917 | 29.762 | confirmation 2 passes |
| I463_c028_n22_regression | C028 | 22 | true | 19.329610 | — | 20.692 | prior N*=22 no regression (8.0 s earlier than C027) |
| I474_c029_n36 | C029 step 1 (I473 build): session skips reloading an unchanged records file | 36 | true | 28.063625 | 687.363281 | 29.731 | passes, 0.27 s margin |
| I476_c029_n37 | **C029** + twin relaxation buckets shared, RequiredReachability idle while completing a plan, per-thread cache-alternative memo (I475 build) | 37 | true | 27.549184 | 693.133911 | 29.333 | **accepted scaling change, N*=37**; all 75 graphs exact |
| I477_c029_n38 | C029 | 38 | false | none | none | 30.135 | budget miss |
| I478_c029_n37_confirm1 | C029 identical source/build | 37 | true | 27.768410 | 693.133911 | 29.677 | confirmation 1 passes |
| I479_c029_n37_confirm2 | C029 identical source/build | 37 | true | 27.896376 | 693.133911 | 29.705 | confirmation 2 passes |
| I480_c029_n34_regression | C029 | 34 | true | 25.970638 | — | 27.918 | prior N*=34 no regression (1.8 s earlier than C028) |
| I491_c030_n38 | C030 step 1 (I490 build): congruence closure follows parent lists after the first round | 38 | false | none | none | 30.055 | miss |
| I493_c030_n38 | + parallel base indexes, cache e-nodes and compiled graphs (I492 build) | 38 | true | 27.860941 | 73.399780 | 29.942 | passes, 0.06 s margin |
| I495_c030_n38 | + parallel plan serialization (I494 build) | 38 | true | 27.693157 | 73.399780 | 29.649 | passes |
| I497_c030_n38 | **C030** + twin buckets share heuristic rounds (I496 build) | 38 | true | 26.879646 | 73.399780 | 28.386 | passes |
| I498_c030_n39 | C030 | 39 | true | 27.543094 | 77.747307 | 29.268 | **accepted scaling change, N*=39**; all 79 graphs exact |
| I499_c030_n39_confirm1 | C030 identical source/build | 39 | true | 27.564896 | 77.747307 | 29.485 | confirmation 1 passes |
| I500_c030_n39_confirm2 | C030 identical source/build | 39 | true | 27.590890 | 77.747307 | 29.127 | confirmation 2 passes |
| I501_c030_n37_regression | C030 | 37 | true | 25.123486 | — | 26.466 | prior N*=37 no regression (2.4 s earlier than C029) |
| I512_c031_n40 | C031 step 1 (I511 build): e-node debug origins only with TG_ENODE_ORIGINS, probes skip re-interning unchanged vectors | 40 | true | 27.162854 | 76.352119 | 28.656 | passes |
| I513_c031_n41 | same | 41 | false | — | — | 28.988 | **crash**: double free after the plan was written (exit -6); parallel buildCompiledGraph race (see C030 defect) |
| I515_c031_n40 | **C031** + compiled graphs built serially again (I514 build) | 40 | true | 27.103334 | — | 29.229 | passes |
| I516_c031_n41 | C031 | 41 | true | 27.549989 | — | 29.598 | **accepted scaling change, N*=41**; all 83 graphs exact |
| I517_c031_n41_confirm1 | C031 identical source/build | 41 | true | 27.477222 | — | 29.489 | confirmation 1 passes |
| I518_c031_n41_confirm2 | C031 identical source/build | 41 | true | 27.537306 | — | 29.525 | confirmation 2 passes |
| I519_c031_n39_regression | C031 | 39 | true | 27.051803 | — | 29.382 | prior N*=39 no regression (0.5 s earlier than C030) |
| I530_c032_n42 | **C032**: per-chunk cost-memo answers, shared fusion selections, restricted engine freed in the background | 42 | false | none | none | 30.145 | miss |
| I531_c032_n41 | C032 | 41 | true | 27.005217 | — | 28.944 | **accepted speed change at N*=41** (0.54 s earlier than C031) |
| I532_c032_n41_confirm1 | C032 identical source/build | 41 | true | 27.078229 | — | 29.152 | confirmation 1 passes |
| I533_c032_n41_confirm2 | C032 identical source/build | 41 | true | 27.037847 | — | 29.090 | confirmation 2 passes |
| I544_c033_n42 | **C033**: sweep-filter parent lists kept across the sweeps of a saturation call | 42 | true | 27.534729 | — | 29.535 | **accepted scaling change, N*=42**; all 85 graphs exact |
| I545_c033_n42_confirm1 | C033 identical source/build | 42 | true | 27.563193 | — | 29.582 | confirmation 1 passes |
| I546_c033_n42_confirm2 | C033 identical source/build | 42 | true | 27.654769 | — | 29.865 | confirmation 2 passes |
| I547_c033_n41_regression | C033 | 41 | true | 26.599438 | — | 28.530 | prior N*=41 no regression (0.4 s earlier than C032) |
| I558_c034_n43 | C034 candidate (I557 build): cost model loaded concurrently with the model graph build; pre-planAll timeline markers | 43 | false | none | none | 30.093 | miss (pre-planAll 2.25 -> 2.11 s at N=42: the graph build takes only 0.14 s, the 1.1 s records parse is the fixed loader) |
| I560_c034_n43 | + compiled graphs built in parallel again, Debug::BaseRefVerifier::registerBase locked (I559 build) | 43 | false | none | none | 30.054 | miss; graph build 0.96 -> 0.51 s |
| I562_c034_n43 | + no lock around kernel view inference (I561 build) | 43 | false | none | none | 30.055 | miss |
| I564_c034_n43 | + VarInfo names formatted on demand (I563 build) | 43 | true | 28.057661 | — | 29.959 | passes, 0.04 s margin |
| I566_c034_n43 | **C034** + plan file written while arenas and constants are set up (I565 build) | 43 | true | 27.899814 | — | 29.686 | **accepted scaling change, N*=43**; all 87 graphs exact |
| I567_c034_n43_confirm1 | C034 identical source/build | 43 | true | 28.047107 | — | 29.785 | confirmation 1 passes |
| I568_c034_n43_confirm2 | C034 identical source/build | 43 | true | 28.008164 | — | 29.853 | confirmation 2 passes |
| I569_c034_n42_regression | C034 | 42 | true | 26.920283 | — | 28.550 | prior N*=42 no regression (0.6 s earlier than C033) |
| I580_c035_n44 | C035 step 1 (I579 build): StartPrecedence / InputPruneStartPrecedence run one pass over the bucket per flush while completing a plan | 44 | false | none | none | 30.109 | miss |
| I582_c035_n44 | + ConsumerStartPrecedence bucket pass, extractCovering reuses single-child row values (I581 build) | 44 | true | 28.184389 | — | 30.000 | passes with no margin; not claimed |
| I584_c035_n44 | **C035** + cheaper relaxation build (debug flag read once, dense class index), base registrations batched per graph (I583 build) | 44 | true | 28.037186 | — | 29.635 | **accepted scaling change, N*=44**; all 89 graphs exact |
| I585_c035_n44_confirm1 | C035 identical source/build | 44 | true | 28.032700 | — | 29.518 | confirmation 1 passes |
| I586_c035_n44_confirm2 | C035 identical source/build | 44 | true | 28.016230 | — | 29.521 | confirmation 2 passes |
| I587_c035_n43_regression | C035 | 43 | true | 27.430076 | — | 28.863 | prior N*=43 no regression (0.5 s earlier than C034) |
| I598_c036_n46 | **C036**: owned-parent index kept between congruence closures (rebuild visits only parents of merged-away classes) | 46 | true | 28.112762 | — | 29.608 | **accepted scaling change, N*=46**; all 93 graphs exact |
| I599_c036_n47 | C036 | 47 | false | none | none | 30.073 | miss |
| I600_c036_n46_confirm1 | C036 identical source/build | 46 | true | 28.176634 | — | 29.742 | confirmation 1 passes |
| I601_c036_n46_confirm2 | C036 identical source/build | 46 | true | 28.105761 | — | 29.573 | confirmation 2 passes |
| I602_c036_n44_regression | C036 | 44 | true | 27.505821 | — | 29.082 | prior N*=44 no regression (0.5 s earlier than C035) |
| I612_c037_cycles_build | C036 + cycle-component counts only after the last pass (others under TG_CYCLE_DIAGNOSTICS) + valid empty owned-parent index when no owned e-nodes | — | — | — | — | — | build ok; N=46 diagnostic reduction 0.23 s shorter, same cost |
| I613_c037_n47 | I612 build | 47 | false | none | none | 30.207 | miss |
| I614_c037_rounds_build | I612 + primal rounds reuse a bucket's selection/final plan when allowed, used and forbidden sets are unchanged (const deterministic inputs; identical plans) | — | — | — | — | — | build ok; N=46 early primal 5.01 -> 4.91 s, same cost 80.8953 |
| I615_c037_n47 | I614 build | 47 | false | none | none | 30.030 | miss by 0.03 s |
| I616_c037_selptr_build | I614 + FusionRule selection memo hands out plain pointers (shard entries are never removed or replaced) instead of copying shared pointers | — | — | — | — | — | build ok; N=46 diagnostic: identical group sizes and cost, partial saturation -0.07 s (noise) |
| I617_c037_n47 | I616 build | 47 | false | none | none | 30.205 | miss |
| I618_c037_costkey_build | I616 + planner cost memo keyed by two 64-bit hashes of the arguments instead of a byte-string key | — | — | — | — | — | build ok; N=46 diagnostic reduction -0.29 s, identical search buckets and cost |
| I619_c037_n47 | **C037** (I618 build) | 47 | true | 28.194377 | — | 29.736 | **accepted scaling change, N*=47**; all 95 graphs exact |
| I620_c037_n47_confirm1 | C037 identical source/build | 47 | true | 28.172919 | — | 29.642 | confirmation 1 passes |
| I621_c037_n47_confirm2 | C037 identical source/build | 47 | true | 28.276296 | — | 29.742 | confirmation 2 passes |
| I622_c037_n46_regression | C037 | 46 | true | 27.611915 | — | 29.099 | prior N*=46 no regression (0.5 s earlier than C036) |
| I632_c038_standin_build | C037 + compaction finds a listed duplicate's representative among its own class's alive e-nodes before a hashcons lookup (congruent e-nodes share a class; same stand-ins and the same no-representative error) | — | — | — | — | — | build ok; N=46 diagnostic: identical search buckets and cost, compaction stand-in step 4.0 -> ~0 CPU-s of lookups, reduction wall -0.05 s |

I094/I095 tested sharing the root primal's selection, cache covering, and list scheduling among buckets only when relaxation signatures, base IDs, lambdas, and weighted status all matched. The N=2 exact check passed, but the first incumbent was 0.10 s slower than C007's reference (within run variance); N=3 still reached search at 28.834 s, essentially the C007 28.735 s result. That candidate was reverted.

I144-I146 tested removing the duplicate full-bucket e-graph copy: reconcile the full saturation graph once, keep it as the sole full-bucket state through cache-candidate discovery, then move it into the bucket before reduction. N=2 passed exact validation at 17.831 s, but N=3 search began at 24.870 s versus C009's 24.740 s and still missed plan writing. The change was reverted because it did not improve the measured objective.

I141-I143 tested proposing every remaining offset with one final propagation and falling back to sequential completion on conflict. N=3 conflicted on producer-consumer buffer overlap and needed the sequential fallback; N=2 passed exact validation but its first incumbent was 18.438 s, slower than C009. The change was reverted.

I149-I151 share the reduced EGraph owner across bucket aliases in `SearchState`, avoiding deep copies of the large graph while retaining per-bucket enode-info arrays and all bucket-specific variables. The graphs are no longer rewritten after reduction; SearchState and its propagators read them. N=2 exact-checks all five graphs with a 17.190435 s first incumbent, 0.5455 s earlier than C009. At N=3, search starts at 24.2016 s (C009 24.7404 s) and root search finishes at 28.7651 s, but the incumbent is validated after the search deadline and plan output is killed by the 30 s supervisor. Keep C010 as the new speed point; keep working on N=3 plan extraction/write time.

I097/I098 profile C007's shared closure: bucket-path injection and postprocessing are small; saturation grows the union state to 571k e-nodes. I098 attributes about 12.24 s of its 20.21 s partial saturation to rebuilding, 6.13 s to rule sweeps, and 1.54 s to the fusion filter over 78 rounds. I100 attempted to reuse the rebuild hash map's scratch allocation between rounds without changing merge semantics; it did not improve the N=3 timing and was reverted. A useful next saturation optimization must avoid rescanning/rebuilding the entire growing shared graph on each round while preserving every rule alternative required by the union of buckets.

I103 parallelized the canonicalization and signature-grouping phases of `EGraph::rebuild`, with class merges still applied serially. I112/I113 then removed pre-reduction copies of the shared partial graph: one representative remains live through reconciliation and reduction, and bucket aliases are copied only after the smaller reduced state is ready. On N=3 this moved search start from C007's 28.735 s to 24.856 s; the primal was validated, but the run missed plan writing at 30 s. On N=2, three identical-source runs exact-checked successfully and improved first-incumbent time by about 0.8 s against C007. I117 and I118-I124 passed the full oracle, all six native suites, and search-gap tests. Keep looking for another few seconds before the root heuristic while preserving the accepted N=2 behavior.

The final reconciliation audit includes both the standalone full-saturation graph and its full-bucket copy, plus one graph per shared-state representative. I125 rebuilt that source; I126-I128 exact-checked all five N=2 graphs with matching source/build. I129, I130-I135, and I136 passed the oracle, six native suites, and search-gap tests. I137 confirms the N=3 miss on that source: search starts at 24.740 s and validates the root plan too late for plan output. For C010, I149 builds; the full 48-case planner oracle, six required native suites, and search-gap tests pass. I151 exact-checks all five N=2 logits graphs; I150 confirms N=3 still misses, now with search start at 24.202 s.

I139 attempted to share the reduced EGraph's frozen class, enode, hash, and logical-index storage across bucket aliases. It reached SearchEngine sooner, but crashed during root primal construction, so the change was reverted. I140 rebuilt C009; its source patch hash matches I126 exactly. Keep the earlier C009 speed result and continue profiling the remaining root preparation and validation cost.

I149-I151 share the reduced EGraph owner across bucket aliases in `SearchState`, avoiding deep copies of the large graph while retaining per-bucket enode-info arrays and all bucket-specific variables. The graphs are no longer rewritten after reduction; SearchState and its propagators read them. N=2 exact-checks all five graphs with a 17.190435 s first incumbent, 0.5455 s earlier than C009.

C011 (I159-I163) addresses the remaining post-search bottleneck: in `SearchEngine::tryPlan`, only `preferredOffset` is initially evaluated, computing `offsetPastAllocations` and domain extremes lazily only when a conflict occurs. `deferred_calls_` and `pending_deferred_slots_` are statically sized in `SearchEngine` to eliminate dynamic allocations in propagator queues. Reachable e-classes are computed once per reduced-state representative and reused across identical bucket aliases. Inner deadline checks are added to `RelaxationBoundComputer::flexibleIntervalBound`, and `SEARCH_MARGIN_SECONDS` is set to 3.5 s. This achieves N*=3! Reference run `I161_c011_n3` finds its first incumbent in 27.956848 s (compile 29.730 s, best cost 109.448563), confirmed by `I162_c011_n3_confirm1` (27.624890 s) and `I163_c011_n3_confirm2` (27.595142 s), all passing exact logits checks across all 7 graphs. Prior N*=2 regression check `I160_c011_n2_regression` passes all 5 graphs with first incumbent at 16.435553 s (faster than C010's 17.190 s).

C012 (I168-I172) resolves the core saturation scaling bottleneck: in `Planner::saturateRules`, round loops previously bounded iteration to `eNodeIdx < sweep_enodes_bound` (the node count before the sweep began). Because slice pushdown along the 75-op critical path of Gemma-3 only advanced by one node per round, saturation required ~78 full rebuild/sweep rounds (spending 12.24 s in rebuild and 6.13 s in sweeps). By evaluating dynamically up to `egraph.getNumENodes()`, new slice pushdown nodes are processed within the current sweep, allowing pushdowns to cascade through the graph in a single round. Saturation time at N=4 plunges from 28.66 s to 8.63 s (>70% reduction). With a 0.5 s search deadline buffer and a 4.5 s search margin to ensure extraction and writing of all 9 graphs, this achieves N*=4.

C013 (I178-I182) addresses graph assembly and serialization overhead as bucket counts grow: in `Planner::buildCompiledGraph`, instruction buffer lookups previously ran an inner loop over all `extraction.buffers` for every instruction child, performing ~44 million inner loop comparisons and hash lookups across the 11 buckets of N=5. C013 replaces this with an $O(1)$ buffer index map, eliminating that quadratic overhead and shaving ~1 s off total compile time (N=4 compile drops from 28.542 s to 27.604 s). Furthermore, `SEARCH_MARGIN_SECONDS` is adjusted to 5.5 s to grant adequate margin for multi-graph extraction and cache persistence. This achieves N*=5! Reference run `I179_c013_n5` finds its first incumbent in 22.611045 s (compile 29.085 s, best cost 99.901581), confirmed by `I180_c013_n5_confirm1` (22.546773 s, compile 29.069 s) and `I181_c013_n5_confirm2` (22.771193 s, compile 29.816 s), all eleven logits graphs passing exact checks (`ok=true`). Prior N*=4 regression check `I182_c013_n4_regression` compiles cleanly in 27.604 s (first incumbent 17.733594 s, all nine ok). All gates pass: full planner oracle (48/48 cases), search gap, and all six native test suites.

## Phase scaling

| N | buckets | saturation s | reduction s | root propagation s | heuristic build s | first incumbent s |
|---|---|---|---|---|---|---|
| 1 | 3 | 1.60763 | 0.82131 (includes search setup) | 1.60842 | 2.10855 | 11.186343 |
| 2 | 5 | 12.526 | 7.2813 (includes search setup) | killed while propagating | not reached | none |
| 3 | 7 | 20.882 | killed during reduction | not reached | not reached | none |
| 3 (C009) | 7 | 20.528 | 4.21 (includes search setup) | 1.286 | 3.202 (includes validation) | validated after search deadline; budget miss |
| 3 (C005) | 7 | 20.4999 | killed during reduction | not reached | not reached | none |
| 2 (C005) | 5 | 12.2099 | 7.2177 (includes setup) | 1.66584 | 3.64232 (includes early validation) | 24.953302 |
| 2 (C011) | 5 | 12.18 | 2.65 (includes search setup) | 0.95 | 1.82 (includes early validation) | 16.435553 |
| 3 (C011) | 7 | 20.578 | 3.65 (includes search setup) | 1.258 | 2.34 (includes validation) | 27.956848 |
| 4 (C011) | 9 | 28.662 | killed during reduction | not reached | not reached | none |
| 3 (C012) | 7 | 6.22 | 3.51 (includes search setup) | 1.21 | 2.12 (includes validation) | 13.067457 |
| 4 (C012) | 9 | 8.63 | 4.28 (includes search setup) | 1.82 | 2.98 (includes validation) | 17.712358 |
| 4 (C013) | 9 | 8.61 | 4.25 (includes search setup) | 1.80 | 2.95 (includes validation) | 17.733594 |
| 5 (C013) | 11 | 11.02 | 4.85 (includes search setup) | 2.45 | 4.12 (includes validation) | 22.611045 |
| 6 (C015) | 13 | 13.85 | 6.57 (includes search setup) | 2.46 | 2.13 (includes validation) | 25.320655 |
| 7 (C015) | 15 | 16.44 | 7.69 (includes search setup) | 2.91 | 2.44 (includes validation) | ~29.8 (after budget) |
| 7 (C016) | 15 | 16.20 | 6.74 (includes search setup) | 2.68 | 1.92 (includes validation) | 27.977648 |
| 8 (C017, one group) | 17 | 15.59 (partial) | 3.89 | 2.60 | 2.49 (includes validation) | 26.113307 |
| 8 (C019, 8 groups) | 17 | 6.0 (partial) +1.2 reconcile | 6.4 | ~4.0 | ~4.5 (includes validation) | 23.990621 |
| 9 (C019, 9 groups) | 19 | 6.4 (partial) +1.2 reconcile | 6.3 | 4.51 | ~5.7 (validation tryPlan 3.67) | 26.304730 |
| 10 (C019, 10 groups) | 21 | 7.26 (partial) +1.3 reconcile | 6.2 | 5.21 | 6.56 (validation tryPlan 4.43) | after deadline (miss) |
| 10 (C020) | 21 | 7.2 (partial) +1.3 reconcile | 3.03 | 5.19 | 5.52 (validation tryPlan 3.72) | 25.032133 |
| 11 (C020) | 23 | 7.9 (partial) +1.4 reconcile | ~3.6 | 5.79 | ~5.9 (validation tryPlan 4.08) | 27.437513 |
| 12 (C020) | 25 | 8.62 (partial) +1.5 reconcile | 3.56 | 6.11 | 6.49 (validation tryPlan 4.43) | after deadline (miss) |
| 12 (C021) | 25 | 7.4 (partial) +1.4 reconcile | 3.26 | 4.69 | ~6.1 (validation tryPlan 4.12) | 25.744363 |
| 13 (C021) | 27 | 8.1 (partial) +1.5 reconcile | 3.54 | 5.05 | ~7.1 (two trial plans: 0.47 rejected + 4.38) | 28.497580 |
| 14 (C021) | 29 | 8.8 (partial) +1.6 reconcile | 3.87 | 5.38 | not reached in budget | none |
| 13 (C022) | 27 | 7.98 (partial) +1.12 reconcile | 4.29 (+0.48 setup) | 4.78 | ~6.2 (0.36 rejected + tryPlan 3.84) | 26.424315 |
| 14 (C022) | 29 | 8.6 (partial) +1.17 reconcile | 4.56 (+0.53 setup) | 4.97 | ~6.6 (tryPlan 4.15) | 27.899543 |
| 14 (C023, diag) | 29 | 5.4 (partial) +0.8 reconcile | 3.1 (+0.5 setup) | ~5.0 | ~6.5 | (diagnostic compile 25.3 s) |
| 17 (C023) | 35 | 6.8 (partial) +1.0 reconcile | 3.8 (+0.56 setup) | 5.96 | ~8.1 (rejected 0.44 + tryPlan 4.79) | 28.296102 |
| 18 (C024) | 37 | 7.4 (partial, 18 groups) +1.1 reconcile | ~4.3 (+0.6 setup) | 6.0 | ~6.0 (rejected 0.54 + tryPlan 3.4) | 27.985398 |
| 19 (C025) | 39 | 6.8 (partial, 19 groups) +1.1 reconcile | 4.3 (+0.75 setup) | 5.49 | 2.6 construction + 4.26 tryPlan | 27.796348 |
| 19 (C026, diag) | 39 | 6.4 (partial) +0.75 reconcile | 3.2 (+0.76 setup) | 5.4 | ~2.4 construction + ~4.2 tryPlan | (diag compile 26.9 s; search start 12.82 s vs 14.96 s C025) |
| 34 (C028) | 69 | ~11.8 (partial, 34 groups) +0.8 reconcile | ~4.7 (+1.1 setup) | skipped (early incumbent) | early primal ~5.9 (restricted validation ~2.2) | 27.794994 |

## Resume checkpoint

- Accepted implementation: **C037 scaling**, N*=47; I618 build, I619 exact N=47 reference, I620/I621 confirmations, I622 N=46 regression; gates I623-I631 pass.
- C037 contents (on top of C036): (1) reduction computes cyclic-component counts only after the last cycle pass (the one that decides DAG buckets; the others under TG_CYCLE_DIAGNOSTICS); (2) a graph with no owned e-nodes starts with a valid empty owned-parent index; (3) primal rounds reuse a bucket's selection/final plan when the allowed, used and forbidden sets are unchanged (extractSelection/extractCovering are deterministic in const inputs); (4) the FusionRule selection memo hands out plain pointers (shard entries are never removed or replaced); (5) the planner cost memo is keyed by two independent 64-bit hashes of every argument (as the selection and op-match memos are) instead of a byte-string key (reduction -0.29 s at N=46); (6) saturation groups log their owned e-node count.
- C036 contents (on top of C035): the EGraph keeps its owned-e-node parent index (`OwnedParentIndex`: per class, the owned e-nodes listing it as a stored child) between congruence closures, and `merge` records each merged-away class (`merged_since_closure`). A closure indexes the e-nodes added since, then visits only the parents of classes merged away since the last one (appending each such class's list to its root's), instead of scanning every owned e-node for a non-canonical child (the scan was ~3.5% of all CPU, run at every rebuild); only merges make an e-node stale, so the stale set is the same. The index is reset by copying, freezing, clearing, materializing (owned positions shift; missing this crashed compaction in a diagnostic) and compaction. N=40: identical group e-node counts, search buckets and cost; compile 28.28 -> 25.93 s (with C035).
- C035 contents (on top of C034): (1) while completing a plan, StartPrecedence, InputPruneStartPrecedence and ConsumerStartPrecedence run one pass over the bucket's classes per flush (PropagationBatch::BUCKET; each class's pass reads only its own selection/start and its children's or consumers' domains, so passing every class reaches the same fixpoint); restricted validation at N=43 3.00 -> 2.6 s; (2) extractCovering reads a kept row value for single-child alternatives and builds the winner's row once (row values summed exactly as valueOf; identical choices); (3) buildRelaxationBucket reads TG_DEBUG_BOUND once and maps classes through a dense array; (4) buildCompiledGraph sends its base registrations to Debug::BaseRefVerifier in one locked batch per graph (graph build 0.49 -> 0.14 s at N=43). N=43 diagnostic compile 28.9 s.
- C034 contents (on top of C033): (1) **root cause of the C030 heap corruption**: `buildCompiledGraph` registers every class's base in the global `Debug::BaseRefVerifier` map, unsynchronized; `registerBase` now locks, and compiled graphs are built in parallel again (gdb stress at N=41: crashed on the first run twice before, then 9 clean runs; kernel view inference works on a local graph and needs no lock); graph build 0.96 -> 0.49 s at N=43; (2) `VarInfo::name()` formats the diagnostic name on demand instead of storing one string per variable (search setup 0.80 -> 0.56 s at N=43); (3) `SearchState::commitAllBucketVariables` commits buckets' variables in parallel (same ids, domains, dirty order, empty/fixed-offset indexes; revision counters advanced as addVar's would); (4) `Session::compile` writes the plan file on a thread (`plan(doSaturate, &persisted)`) while arenas and constants are set up (both only read the plan), joining before `compiled`; (5) the cost model is loaded on a thread while the model graph is built (LLMSession; Session owns its CostModel through a pointer; the stamp is taken before the parse), and pre-planAll timeline markers.
- C033 contents (on top of C032): saturateRules' sweep-filter parent lists (child class -> parent e-nodes) persist across the sweeps of one call: each sweep indexes only newly added e-nodes and appends the list of every class that stopped being canonical to its root's list (same canonical (class, parent) pairs as re-indexing every e-node, so identical filter depths); was a full rebuild over all ~265k group e-nodes per sweep (~3% of all CPU). N=40: partial saturation 12.09 -> 11.71 s, identical group e-node counts and cost.
- C032 contents (on top of C031): (1) computeENodeInfos chunks keep their own argument buffers, memo key buffer and cost answers (`memoizedCost` key_scratch/local_costs: repeated layer queries skip the shared lock; thread_local accesses cost __tls_get_addr calls in the module); (2) FusionRule's fused-candidate selection memo is process-wide (64 locked shards behind a per-thread cache) instead of per thread, so saturation groups share selections (same 128-bit key as before); (3) validateRestricted frees its restricted engine on a background thread (~0.3 s at N=40). N=40 diagnostic: reduction 5.40 -> 5.15 s, early primal 5.51 -> 5.23 s; same group e-node counts and cost.
- C031 contents (on top of C030): (1) compiled graphs are built serially again (the C030 parallel loop raced, see below); (2) addOpToEGraph records e-node debug origins only with TG_ENODE_ORIGINS=1 (diagnostic-only field; formatting/interning it per added node was ~2% of saturation); (3) ENode::assignFields compares the probe's interned shape/strides/engines with the new values before re-interning (probe reuse in kernel loops); partial saturation at N=38 12.29 -> 11.70 s with identical group e-node counts and cost.
- C030 contents (on top of C029): (1) `closeCongruenceIncrementally` scans every owned e-node only in its first round; later rounds visit only the owned parents of the classes the previous round merged away (per-class parent lists built once when a second round is needed; a merged-away class's list is appended to the survivor's), then canonicalizes nodeToEClass once; same classes/e-nodes (verified at N=36: identical group e-node counts, search variables and cost); the per-round full rescans had been ~10% of all CPU; (2) bucket base-id indexes are built concurrently before the cache-candidate loop, CACHE e-nodes are added to bucket graphs in parallel, compiled graphs are built in parallel (each reads only its const e-graph/extraction/infos), and plan blocks are serialized per graph in parallel then written in order (same bytes); (3) SearchState::reserveVariables before committing bucket variables, VarInfo moved into addVar; (4) in buildRootPrimalPlans a bucket whose relaxation is a copy of an earlier one with the same weight takes that bucket's estimates, selection, covering, alias decisions and schedule instead of recomputing them (deterministic in those inputs); early primal at N=38 6.49 -> 5.70 s, same cost.
- C030 defect found later: building compiled graphs in parallel was not thread-safe (kernel view inference/matching through temporary Graphs); at N=41 (I513) it corrupted the heap (`double free or corruption` after the plan was written, exit -6; a gdb run crashed in Planner::buildCompiledGraph on a pool thread). C031 restores the serial loop; four N=41 gdb runs were then clean. C030's accepted runs passed their exact checks, but C031 replaces it as the reference.
- C029 contents (on top of C028): (1) `Session::plan` reloads the cost records only if the file's path, size or write time changed since the constructor loaded it (the same file was parsed twice, ~1.1 s each at 167k records; pre-planAll time at N=34 2.58 -> 1.49 s; the cost model and records are unchanged); (2) buildRootPrimalPlans builds one relaxation per group of buckets with the same shared e-graph owner, root, reachable classes, e-node costs/views and selection domains (what buildRelaxationBucket and markCacheAlternatives read) and copies it to the others with their selection variables (prefill/decode twins before root propagation); early primal at N=36 6.21 -> 5.33 s; (3) RequiredReachability returns at once while completing a plan (it only narrows an unfixed root selection; every selection is fixed first), so the restricted state never builds its index; (4) markCacheAlternatives keeps a per-thread memo in front of its locked one.
- Where the time goes at N=41/42 (C032): pre-planAll ~1.9 s (one 1.1 s records parse is the fixed cost model's loader), full saturation 0.8 s, partial saturation ~11.4 s (41 groups on 16 cores: CPU-bound, so only less CPU helps, not more inner parallelism), reconcile+cache ~1.7 s, reduction ~5.1 s (computeENodeInfos ~45%, compaction ~35% of its CPU), setup ~0.8 s, early primal ~5.2 s (construction ~2.5 s, restricted validation ~2.5 s serial), extraction+write ~1.4 s. CPU split (N=40, all threads): saturation ~50% (fusion materialization 30% of it, sweep loop/filter 14%, fusion matching 8%, region computation 6%, copy-chain rules 6%), reduction ~25%.
- Next ideas: incremental sweep-filter parent index in saturateRules (rebuilt over all ~265k group e-nodes each sweep, ~3% of all CPU); avoid materializing fused candidates already present; one compaction instead of two in reduction; MemoryNoOverlap bucket batching while completing a plan (~25% of restricted validation); share reduced graphs / heuristic work across near-identical middle groups (the only route to large N; needs a checkable equivalence argument).
- C028 contents (on top of C027): (1) **early incumbent before root propagation**: `SearchEngine::solve` (when the planner sets `add_validation_propagators`) pushes the root node open at bound 0, samples the gap, then `runEarlyPrimal` builds the root heuristic's plans on the unpropagated domains (a diagnostic showed the heuristic picks the same plan with or without root propagation: predicted 420.521 vs 420.522 at N=21) and validates each with `validateRestricted` until one holds; root propagation and everything after it run only if about `kRootSecondsPerVar` (2.5 us) per variable fits before the deadline, else the early incumbent is kept (gap and volume reported as at a deadline). A root contradiction after a validated incumbent throws. The post-root heuristic validates only starts predicted to beat the incumbent. (2) **restricted validation**: `SearchState::restrictedTo(classes)` builds a state over the same buckets, shared e-graphs, enode infos and cache candidates (cache variables keep their domains) with variables only for the plan's active classes plus every class of a cached base (all in reachable order); selections are fixed to the plan (0 for inactive cached-base classes, so CacheFreshness rejects a plan that leaves a cached base unselected), ranks follow dispatch order, and `completeFixedPlan` (the propagate/disjoint-offset/complete half of tryPlan, now shared) completes it with `addValidationPropagators`: every constraint propagator except StartRankWindow (root symmetry breaking: it restricts which of equivalent schedules search visits, not validity) and the bound-only propagators (CriticalPath, EngineWorkload, StartRunMakespan, RelaxationBound). A selected alternative whose child is not active is rejected explicitly (the restricted state has no variable for that child). Soundness: every constraint of the full state over the plan's classes is present and checked on a fully fixed assignment; constraints of unselected classes in the full state only canonicalize their irrelevant coordinates (UnselectedStartOffset) or exclude them from consideration (consumers skip fixed-0 children), except the cached-base requirement, which the restricted state keeps; the incumbent's extraction comes from the restricted state over the same e-graphs, and the exact check runs every graph. When root propagation does run (small N), it must not contradict the incumbent. (3) While completing a plan, StartUnique (bucket pass: fixed active starts distinct, their values removed from unfixed ones) and WriteAfterReadStart run once per bucket per flush (PropagationBatch::BUCKET) instead of per start; restricted propagation at N=21 2.65 -> 1.16 s. (4) Structural propagator indexes (`Propagator::prepare`, ensurePropagationState) are built concurrently before root propagation; ensurePropagationState resizes its per-variable arrays and deduplicates lists in parallel. At N=34 the first incumbent arrives ~6 s after search start with no root propagation.
- C027 contents (on top of C026): `EClass::shape`/`strides` are `InternedVector` values (a pointer into the intern table with the read-only vector interface, implicit conversion to `const std::vector<T>&`, assignment interns; `toString` overload), so class copies (copy-on-write materialization in reconcile, getMutableEClass, compaction) copy only the e-node list. N=21 diagnostic: search start 13.92 -> 12.75 s (reconcile 0.84 -> 0.51 s, reduction 3.62 -> 3.11 s); same plan cost.
- C026 contents (on top of C025): allocation reduction (an LD_PRELOAD jemalloc diagnostic at N=19 was 3.7 s faster, 2.9 s of it before search, so glibc malloc time was the target): (1) ENode stores its op name and debug origin as pointers into a global intern table (`internString`, 64 locked shards behind a per-thread cache, never freed) and its shape, strides and engine list as interned vectors (`internVector`; engines compared including their supported memory spaces), so copying or building an e-node allocates only its children (and a non-empty content hash); equality compares interned pointers (engine lists fall back to the old index/type comparison); signatures still hash contents; the constructor takes these by const reference; (2) Tarjan in `findCyclicComponents` keeps only cyclic components (one reused scratch vector, no per-singleton allocation); (3) `memoizedCost` builds its key in a thread-local buffer; `computeENodeInfos` reuses thread-local argument vectors; (4) `materializeFusedCandidates` reads the old node's interned shape/strides/origin instead of copying them and reuses one children buffer. Search start at N=19: 14.96 -> 12.82 s (reduction 4.41 -> 3.20 s, reconcile 1.09 -> 0.75 s); same plan cost.
- C025 contents (on top of C024): (1) `SearchState::domainEverChanged` (per-variable flag set by setDomain, never cleared); StartPrecedence reschedules a class's own and its children's start variables after a SELECTED event only if that selection domain ever changed: every variable is dirty from addVar, so on its initial event the starts have seen only this selection (root START events at N=14: 1.74M -> ~1.0M); root propagation 6.0 -> 5.2 s at N=18; (2) PearceKellyCycle keeps per-class parent lists (vector indexed by class) that carry the parent's selection variable (no hash lookups in the ancestor walk); (3) primal heuristic: a cached class selected as a view of a class with another cached base cannot be placed (each cached base gets its own offset; a view shares its base's), so such a view is forbidden in that bucket and selection repeats (`ForbiddenAlternatives`, extractSelection/extractCovering skip them); if the bucket's root could then not be produced, the base leaves the cache set instead. The first trial plan (the [0.5,0.5] weighting) was always rejected by ViewSelectOffset for this reason (KV-cache [4,N,256] as a RESHAPE of a cached [1,4,N,256] in bucket 0); now it validates, so the first incumbent is that plan (higher cost than the [0.6,0.4] plan C024 happened to validate second); (4) relaxation signatures are computed only for region-estimate starts (cheap estimates cost less than the signatures); the alias check runs per bucket in parallel.
- C024 contents (on top of C023): (1) InputPruneStartPrecedence and StartPrecedence skip their consumer loops on SELECTED events and skip START events of classes fixed unselected: a consumer's pruning reads only a child's start minimum and whether it is fixed to 0, which only removes it from consideration, so these calls could never change a domain; consumer loops pass the consumer entry's stored selection/start variables instead of two hash lookups; (2) ViewToBaseOffset returns at once for an unselected view (it has no confirmed base and filterUnviableViewEnodes keeps a fixed-0 selection); (3) EngineWorkload recomputes a bucket's bound only when the variable had or gains a costed selected alternative (every bound component writer calls updateLowerBoundBucket itself); (4) CacheFreshness's selection->cached index is a dense VarId array; (5) partial saturation uses one group per distinct output region up to 4x hardware threads (was capped at hardware threads: at N=18 two merged groups took 9.2 s vs 7.4 s with 18 groups). tryPlan at N=17: 4.79 -> 3.32 s; N=14 trial-plan events are 96% unselected classes (2.97M of 3.1M).
- C023 contents (on top of C022): `EGraph::rebuild` keeps the hashcons between calls (`hashcons_tracks_owned`, maintained by addENode, freeze/clear and compaction) and closes congruence incrementally (`closeCongruenceIncrementally`): only owned e-nodes with a non-canonical child leave the hashcons, are re-canonicalized and looked up again, repeated while merges happen. Same classes and distinct e-nodes as the full re-hash (verified at N=14: identical per-group e-node counts and per-bucket variable counts); only which of two congruent e-nodes is the hashcons representative can differ. Middle-group rebuild 3.4 s -> 0.43 s; partial saturation 8.9 -> 5.4 s at N=14.
- C022 contents (on top of C021): (1) `ThreadPool::parallel_for` callers spin 50 us, then block on a per-call condition variable signalled by the worker that completes the last task (was: `yield` loop, ~400 samples of `sched_yield` in saturation rebuilds); (2) trial plans the heuristic prices at infinity are not validated; ParentRemoval returns early while completing a trial plan whose bucket selections are all fixed (checked once per selection revision): a fixed parent alternative with an unselected child is then a conflict SelectionChildren reports; (3) `reconcileBaseEqualities` rebuilds a graph after round 1 only if that round merged classes or retagged a scatter node; (4) `SearchState::addAllBucketVariables`: reachable classes and variable descriptions per bucket in parallel, `addVar` in bucket order (identical ids; START selection index translated by the bucket's first id), per-bucket maps filled in parallel; `addBucketVariables` keeps the single-bucket path for tests.
- C021/C020/C019 contents as in their notes (glibc malloc tuning, Knuth heuristic DP, trial-plan bound skip, parallel root setup, compaction stand-in fix, one saturation group per output region).
- N=14 phases (I358): partial saturation 8.6 s (groups for middle t ~7-8 s each under contention), reconcile 1.17 s, cache 0.6 s, reduction ~4.0 s, setup 0.53 s, root propagation 4.97 s, primal build ~2.4 s, tryPlan 4.15 s, extraction+write ~1.3 s; compile 29.1 s.
- Working tree / build: C037 + compaction same-class representative (I632 build, not yet evaluated). Tried and reverted: pre-forbidding aliasing views before the primal rounds (leaves buckets with only infinite-cost plans); computing the root-maintenance check for all buckets in parallel (the serial loop stops at the first unmaintained bucket; parallel was 0.9 s slower); hashing computeENodeInfos cost queries straight from the e-graph to skip argument copies (identical, no measurable gain: reduction is ~76 CPU-s on 16 threads and bandwidth-bound). Measured (N=46, TG_DEBUG_COMPACT diagnostic, removed): 94 compactions take ~22 CPU-s (moving e-nodes 11.5, hashcons rebuild 3-4, alive/stand-in 6.8 of which hashcons lookups 4.0); the restricted plan validation propagates 576k variables serially in 1.95 s. No running processes. Next run id: I633.
- Scaling bound (measured, N=20 and N=46 diagnostics): a middle saturation group is ~266k e-nodes of which ~184k are owned (shared base ~82k), the same at N=20 and N=46; each bucket's reduced e-graph averages ~82k e-nodes and the search has ~112k variables per bucket (10.4M at N=46, 93 buckets). At N=32768 (65,537 buckets) per-bucket state would be ~5.4e9 e-nodes and ~7.3e9 search variables, against 31 GB RAM on box (<5 bytes per variable) and the 30 s budget. Reaching it therefore needs every per-bucket stage (saturation, reduction, search state, extraction, graph build, persistence) to be shared across buckets through a checkable equivalence (e.g. compilation parametric in the bucket's row offset), not further constant-factor work; buckets may not be merged or dropped. The owned e-node count is now logged per saturation group.
- Shared saturation across middle groups (investigated, not implemented): prefill bucket t has dirty tokens [0, t) (prefillDecodeBuckets), so a group's partial-path shapes grow with t rather than shifting, and saturation's fixed equivalent-candidate cost filter decides on concrete shapes and costs. A group's saturated graph can therefore legitimately differ with t, and reusing one group's graph for another is exact only if re-checked per group, which costs about what sharing saves. Even with saturation shared, per-bucket reduction (~1.6 CPU-s per representative) and search state (~112k variables per bucket) stay linear in N; the estimated ceiling is in the low hundreds, not 32768.
- Where the time goes (N=46): the 44 middle saturation groups are each ~265k e-nodes and ~4 CPU-s, independent of N; saturation is CPU-bound (~180 CPU-s on 16 threads), FusionRule::apply 39% of it (addOpToEGraph during fused-candidate materialization the largest part), computeBucketRegions 8%. Per-N cost ~0.9 s across all phases, so reaching large N needs the middle groups to share one saturation through a checkable equivalence (not implemented).
- Scaling slope at C022: ~1.3-1.5 s of compile per unit of N; every phase is per group or per bucket and contention-bound (at N=14 a middle-t representative's reduction steps take rebuild 1.27 s, infos+prune 1.05 s, cycle passes 0.77 s, final infos 0.54 s with ~15 running concurrently). Micro-optimizations now buy about one N each.
- Structural next step (recommended): groups for t=2..N-1 are near-identical (same enode counts within 0.1%, same reduced class counts within a few classes); saturating and reducing one representative and instantiating the others by substituting t would make saturation/reduction cost independent of N. It must be checkable by the planner (e.g. saturate two values of t, verify the results are isomorphic up to the t-dependent shape/slice constants, then instantiate), and the instantiated graph must equal what the bucket computes itself or be a sound superset.
- Next for N=15/16 otherwise: hardware threads (16) cap the default group count, so N>16 merges regions into shared groups (slower closures); saturation contention (jemalloc diagnostic showed allocator-bound closures) and the search phase (root propagation + primal + tryPlan ~11.5 s, all per bucket) are the scaling terms. Per-bucket work over prefill/decode pairs sharing one graph is duplicated (different domains).

## Runs

<!-- tests/search_experiment.py appends one entry per run below; add a short note after each. -->

### I001_build
- In progress: 2026-10-21T12:31:44.254550-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I001_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I001_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I001_build.patch`
- Command: `.venv/Scripts/python.exe .venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T12:31:44.283224-04:00; exit 1; runner elapsed 0.016 s.

### I002_build
- In progress: 2026-10-21T12:31:50.309261-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I002_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I002_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I002_build.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test bench --log-level INFO`

### Initial setup (2026-10-21)

GPU verified by `nvidia-smi`: Quadro M4000, driver 535.261.03. Commands run with unrestricted filesystem/device access. I001_build is an invocation error, not a source/build failure: runner already inserts Python, so passing `.venv/bin/python` made Python read its ELF binary. Corrected I002_build passes `build.py` directly. No evaluation or candidate measured yet.
- Completed: 2026-10-21T12:37:43.178478-04:00; exit 0; runner elapsed 352.867 s.

### I003_baseline_n1
- In progress: 2026-10-21T12:37:46.817265-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I003_baseline_n1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I003_baseline_n1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I003_baseline_n1.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I003_baseline_n1`
- Completed: 2026-10-21T12:38:33.559578-04:00; exit 0; runner elapsed 46.740 s.
  `[SearchEngine volume] iterations=18 elapsed=24.400473108 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604348.348796603 log2-remaining-fraction=-139.5907143810764 eliminated-bits=139.5907143810764 remaining-nodes=13 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=13.483684531911059 normalized=0.44945615106370196 final-gap=0.10454534565794722 final-incumbent=14.694063999999999 final-bound=13.157868000000001 first-incumbent-elapsed=11.18634295463562 samples=11`
  `[SearchEngine] New best cost: 14.6941 at iteration 0 source=root-heuristic event-epoch=1792600681.1675508 program-deadline=1792600696.9812207`
- First incumbent: 14.186330080 s, cost 14.6941; best cost 14.694064.

### I004_baseline_n2
- In progress: 2026-10-21T12:38:52.221489-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I004_baseline_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I004_baseline_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I004_baseline_n2.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I004_baseline_n2`

I003_baseline_n1: baseline hypothesis confirmed: N=1 passes. All 3 graphs exact-check ok; maximum absolute error 2.289e-05. First incumbent uses the required `[Program gap]` elapsed (11.186343 s), not the runner top-level first_incumbent.program_elapsed (14.186330 s). Build I002 passed. No source changes or optimization gate yet. User additionally authorizes tooling/build improvements to shorten the experiment loop; preserve fixed build options and evaluator.
- Completed: 2026-10-21T12:39:42.522841-04:00; exit 1; runner elapsed 50.299 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I005_baseline_n3
- In progress: 2026-10-21T12:40:10.973266-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I005_baseline_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I005_baseline_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I005_baseline_n3.patch`
- Command: `.venv/Scripts/python.exe -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I005_baseline_n3`

I004_baseline_n2: honest budget miss, supervisor exit=-9/timed_out=true, no incumbent. Search started at 19.8073 s of planner time, root relaxation still executing at kill. I005_baseline_n3 calibration requires new records (bench active); do not restart or treat lack of supervisor output as a stopped job.
- Completed: 2026-10-21T12:44:13.439508-04:00; exit 1; runner elapsed 242.464 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I006_noop_build
- In progress: 2026-10-21T12:44:22.773405-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I006_noop_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I006_noop_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I006_noop_build.patch`
- Command: `.venv/Scripts/python.exe build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T12:44:23.348417-04:00; exit 0; runner elapsed 0.573 s.

### I007_tool_tests
- In progress: 2026-10-21T12:44:43.760698-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I007_tool_tests.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I007_tool_tests.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I007_tool_tests.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_experiment`
- Completed: 2026-10-21T12:44:43.827368-04:00; exit 0; runner elapsed 0.064 s.

### I008_validate_first_build
- In progress: 2026-10-21T12:44:49.655261-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I008_validate_first_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I008_validate_first_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I008_validate_first_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

### I009_tool_tests
- In progress: 2026-10-21T12:45:17.822400-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I009_tool_tests.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I009_tool_tests.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I009_tool_tests.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_experiment`
- Completed: 2026-10-21T12:45:17.884980-04:00; exit 0; runner elapsed 0.061 s.

### I010_search_gap
- In progress: 2026-10-21T12:45:22.741378-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I010_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I010_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I010_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-21T12:45:22.769702-04:00; exit 0; runner elapsed 0.026 s.

### Candidate C001: validate initial plan before local search

Hypothesis: validating the best constructed initial plan before optimizing starts reduces time to first incumbent. No bounds, pruning, candidate set, bucket definitions or cost model change. Soundness follows from the existing `tryPlan` path: fixes the whole plan, runs normal propagators and evaluates the normal leaf before recording an incumbent. Failed validation leaves the later local-search and validation path available. New adversarial bound/pruning cases are not required because neither changes.

I005 baseline N=3 completed calibration in 3 rounds (1429, 4, 0 pending; no failures) and missed 30 s during reduction. Saturation 20.882 s; no search/first incumbent. Existing caught shape-inference error messages appear in its log; supervisor outcome is timeout, not a verified correctness result.

I006 no-op default build passed in 0.48 s wall clock (runner metadata has exact time), reusing kernels and main objects; no build-system optimization warranted yet. I007/I009 runner reporting tests and I010 search-gap tests passed. Runner now prefers supervisor gap first-incumbent elapsed, records actual Python executable, and snapshots build.py/kernel_linter.py alongside existing source paths. Baseline raw logs/metadata remain unchanged for provenance; comparison table uses authoritative gap value.
- Completed: 2026-10-21T12:47:06.660026-04:00; exit 0; runner elapsed 137.003 s.

### I011_validate_first_n1
- In progress: 2026-10-21T12:47:10.183150-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I011_validate_first_n1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I011_validate_first_n1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I011_validate_first_n1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I011_validate_first_n1`
- Completed: 2026-10-21T12:47:56.763383-04:00; exit 0; runner elapsed 46.577 s.
  `[SearchEngine volume] iterations=25 elapsed=24.460498413 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604305.6306149275 log2-remaining-fraction=-182.30889605660923 eliminated-bits=182.30889605660923 remaining-nodes=18 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.532792702490354 normalized=0.41775975674967847 final-gap=0.13932751487255396 final-incumbent=15.287891999999999 final-bound=13.157868000000001 first-incumbent-elapsed=8.9924080371856689 samples=12`
  `[SearchEngine] New best cost: 15.2879 at iteration 0 source=root-heuristic event-epoch=1792601242.3074231 program-deadline=1792601260.3150268`
- First incumbent: 8.992408037 s, cost 15.2879; best cost 15.287892.

### I012_validate_first_n1_confirm1
- In progress: 2026-10-21T12:48:11.554282-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I012_validate_first_n1_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I012_validate_first_n1_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I012_validate_first_n1_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I012_validate_first_n1_confirm1`
- Completed: 2026-10-21T12:48:58.096404-04:00; exit 0; runner elapsed 46.540 s.
  `[SearchEngine volume] iterations=21 elapsed=24.378814326000001 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604322.9990325295 log2-remaining-fraction=-164.94047845457681 eliminated-bits=164.94047845457681 remaining-nodes=16 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.772253115667791 normalized=0.4257417705222597 final-gap=0.13932751487255396 final-incumbent=15.287891999999999 final-bound=13.157868000000001 first-incumbent-elapsed=9.2384898662567139 samples=12`
  `[SearchEngine] New best cost: 15.2879 at iteration 0 source=root-heuristic event-epoch=1792601303.9311128 program-deadline=1792601321.6926353`
- First incumbent: 9.238489866 s, cost 15.2879; best cost 15.287892.

### I013_validate_first_n1_confirm2
- In progress: 2026-10-21T12:49:12.032011-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I013_validate_first_n1_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I013_validate_first_n1_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I013_validate_first_n1_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I013_validate_first_n1_confirm2`

I008 build passed; I011 candidate and I012 first confirmation passed all 3 exact graph checks with first incumbent 8.992408 s / 9.238490 s vs baseline 11.186343 s. C001 final plan cost 15.287892 (quality ungated). Improvement gate looks positive; correctness gate remains pending. No bounds or pruning changed.
- Completed: 2026-10-21T12:49:58.754286-04:00; exit 0; runner elapsed 46.720 s.
  `[SearchEngine volume] iterations=27 elapsed=24.431200989000001 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604281.0922409841 log2-remaining-fraction=-206.84727000002749 eliminated-bits=206.84727000002749 remaining-nodes=18 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=12.661002244124784 normalized=0.42203340813749279 final-gap=0.13932751487255396 final-incumbent=15.287891999999999 final-bound=13.157868000000001 first-incumbent-elapsed=9.1289975643157959 samples=13`
  `[SearchEngine] New best cost: 15.2879 at iteration 0 source=root-heuristic event-epoch=1792601364.2808719 program-deadline=1792601382.1518867`
- First incumbent: 9.128997564 s, cost 15.2879; best cost 15.287892.

### I014_validate_first_oracle
- In progress: 2026-10-21T12:50:03.374278-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I014_validate_first_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I014_validate_first_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I014_validate_first_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T12:50:09.480364-04:00; exit 0; runner elapsed 6.104 s.
- Oracle verdict: 46/46 cases passed.

### I018_start_rank_window
- In progress: 2026-10-21T12:50:11.232452-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I018_start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I018_start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I018_start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`

### I015_private_layout
- In progress: 2026-10-21T12:50:11.232428-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I015_private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I015_private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I015_private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`

### I020_relaxation_bound
- In progress: 2026-10-21T12:50:11.232266-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I020_relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I020_relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I020_relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`

### I017_incremental_propagators
- In progress: 2026-10-21T12:50:11.232676-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I017_incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I017_incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I017_incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`

### I019_selection_reachability
- In progress: 2026-10-21T12:50:11.236230-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I019_selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I019_selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I019_selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`

### I016_write_after_read_cache
- In progress: 2026-10-21T12:50:11.240804-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I016_write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I016_write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I016_write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-21T12:50:11.814622-04:00; exit 0; runner elapsed 0.580 s.
- Completed: 2026-10-21T12:50:11.855199-04:00; exit 0; runner elapsed 0.621 s.
- Completed: 2026-10-21T12:50:11.861920-04:00; exit 0; runner elapsed 0.618 s.
- Completed: 2026-10-21T12:50:11.869306-04:00; exit 0; runner elapsed 0.635 s.
- Completed: 2026-10-21T12:50:11.898180-04:00; exit 0; runner elapsed 0.660 s.
- Completed: 2026-10-21T12:52:01.975202-04:00; exit 0; runner elapsed 110.741 s.

### C001 acceptance

Accepted **speed** change. N*=1 unchanged, first incumbent 8.992408 s vs original 11.186343 s (19.6% earlier); independent confirmations 9.238490 and 9.128998 s, all exact logits checks passed. Correctness gates: I010 search gap; I014 oracle 46/46; I015 private layout; I016 write-after-read cache; I017 incremental propagators; I018 start-rank window; I019 selection reachability; I020 relaxation bound, all exit 0. I020 includes 104361 weighted checks and 23666 exact pruning checks. No evaluation was run concurrently with build/tests.

### I021_deferred_primal_build
- In progress: 2026-10-21T12:54:48.055248-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I021_deferred_primal_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I021_deferred_primal_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I021_deferred_primal_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

### Candidate C002: defer root relaxation and validate starts lazily

Build starts one at a time; every proposed plan goes through normal tryPlan/leaf evaluation. Stop generating heuristic weightings after the first accepted plan; all saturation/search alternatives remain available. Root RelaxationBoundPropagator is explicitly deferred during both structural root propagation and trial validation. Compute it only from restored root (assert trial-completion=false and trail marker equals root marker), then refresh root bounds. A time-limited maximization keeps max of completed B(lambda); each B(lambda) <= min over all root plans of weighted engine work <= min root makespan. Empty max is zero. Therefore truncation weakens the bound without losing any optimal plan. Existing float safety margin remains. New adversarial oracle case 36 has initial uniform-weighted choice cost 8 but optimum 5, exposing a bound incorrectly retained from fixed trial domains. New native expired-deadline check expects zero/no weighting. Deferred load/split passes check the deadline; each completed sub-bound remains sound.

I021 is a failed/intermediate build: new native test initially used an unqualified RelaxationBucket and treated GraphSpec as RelaxationBucket. Fixed by constructing a SearchState via existing makeState helper and buildRelaxationBucket. A search_engine.hpp error-path edit occurred during its binding compilation; after I021 exits, touch that header and run a new recorded default build to guarantee final source is compiled. No evaluation may use I021 binaries.
- Completed: 2026-10-21T12:57:02.042459-04:00; exit 1; runner elapsed 133.984 s.

### I022_deferred_primal_build
- In progress: 2026-10-21T12:57:05.195625-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I022_deferred_primal_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I022_deferred_primal_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I022_deferred_primal_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

### I023_deferred_case_reference
- In progress: 2026-10-21T12:57:45.519113-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I023_deferred_case_reference.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I023_deferred_case_reference.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I023_deferred_case_reference.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle --case 36_deferred_root_bound_after_primal --reference-only`
- Completed: 2026-10-21T12:57:45.632907-04:00; exit 0; runner elapsed 0.112 s.
- Oracle verdict: 1/1 cases passed.
- Completed: 2026-10-21T12:59:23.086097-04:00; exit 0; runner elapsed 137.887 s.

### I024_stale_root_diagnostic
- In progress: 2026-10-21T12:59:41.587315-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I024_stale_root_diagnostic.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I024_stale_root_diagnostic.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I024_stale_root_diagnostic.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle --case 37_deferred_root_bound_shared_local_minimum`
- Completed: 2026-10-21T12:59:41.761295-04:00; exit 1; runner elapsed 0.170 s.
- Oracle verdict: 0/1 cases passed.

### I025_deferred_primal_build
- In progress: 2026-10-21T12:59:52.174770-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I025_deferred_primal_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I025_deferred_primal_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I025_deferred_primal_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I022 intermediate build passed, now through installed ccache with unchanged compiler flags. I023 independent reference for new case 36 passed. I024 **diagnostic rejected**: new case 37 proves stale trial bound aggregation unsound (root_lower_bound=8 > independently enumerated optimum=5). No timed Gemma evaluation used this build. Fixed computeDeferredRootBound to combine saved root_bucket_lower_bounds_ with newly computed global relaxation bounds, never state.bucket_lower_bounds/state.lower_bound after trial rollback. I025 rebuild in progress on final source; next run both adversarial cases/full oracle before timed N=2.

Tooling: build.py now transparently uses installed ccache for non-Windows clang/gcc object compilation. It retains flags, object/link semantic keys, and existing dependency checks. No-op build already 0.48 s; expected benefit is reuse on source rollback/repeated experiments. I022 log confirms `/usr/bin/ccache` invocation. Optional launcher absence uses the ordinary compiler.
- Completed: 2026-10-21T13:02:09.773960-04:00; exit 0; runner elapsed 137.597 s.

### I026_deferred_primal_oracle
- In progress: 2026-10-21T13:02:16.405929-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I026_deferred_primal_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I026_deferred_primal_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I026_deferred_primal_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T13:02:20.807959-04:00; exit 0; runner elapsed 4.400 s.
- Oracle verdict: 48/48 cases passed.

### I027_deferred_primal_n2
- In progress: 2026-10-21T13:02:24.505931-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I027_deferred_primal_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I027_deferred_primal_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I027_deferred_primal_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I027_deferred_primal_n2`
- Completed: 2026-10-21T13:03:14.950322-04:00; exit 1; runner elapsed 50.442 s.
  `[SearchEngine volume] iterations=0 elapsed=9.0957864770000008 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792602193.7393134 program-deadline=1792602191.825146`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: 31.914167404 s, cost 91.3422; best cost 91.3422.

### I028_deferred_primal_n1
- In progress: 2026-10-21T13:03:40.226956-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I028_deferred_primal_n1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I028_deferred_primal_n1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I028_deferred_primal_n1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I028_deferred_primal_n1`
- Completed: 2026-10-21T13:04:25.675158-04:00; exit 0; runner elapsed 45.445 s.
  `[SearchEngine volume] iterations=27 elapsed=24.358063419 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604281.0922409841 log2-remaining-fraction=-206.84727000002749 eliminated-bits=206.84727000002749 remaining-nodes=28 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=27.005365529753561 normalized=0.90017885099178541 final-gap=0.84815456787562382 final-incumbent=86.653037999999995 final-bound=13.157868000000001 first-incumbent-elapsed=6.0931582450866699 samples=15`
  `[SearchEngine] New best cost: 87.3138 at iteration 0 source=root-heuristic event-epoch=1792602229.5030651 program-deadline=1792602250.4099183`
- First incumbent: 6.093158245 s, cost 87.3138; best cost 87.3138.

### I029_lazy_regions_build
- In progress: 2026-10-21T13:05:59.877970-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I029_lazy_regions_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I029_lazy_regions_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I029_lazy_regions_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I025 corrected build passed; I026 oracle **48/48 passed**, including new cases 36/37 (I024 demonstrated case 37 rejects stale bound 8 against optimum 5).

I027 C002 N=2 **miss**: timed compile killed 30.102 s during arena materialization. Saturation=12.4572 s, search starts=19.7625 s, structural root propagation=1.66414 s, heuristic construction+early validation=7.2632 s; first uniform start rejected by ViewSelectOffset, next weighting accepted after 2.88814 s validation (4030 completion choices/76 fallbacks). Search ended at planner time28.8583 s, after the search deadline; plan writing unfinished. No authoritative gap summary/check. Runner fallback first_incumbent=31.914167 is derived from a search deadline, hence overstates wall elapsed by3 s; raw event epoch plus known margin imply about28.914167 s, diagnostic only. Do not count a passing N=2. Need save at least ~2 s before plan writing.

I028 C002 N=1 passed exact graph check; first incumbent=6.093158 s vs C0018.992408 s. Compile27.864 s, final cost86.653038. C002 not accepted yet: continuing toward larger N before confirmations/gates.

### Candidate C003: cheap estimates without region preparation

Same per-class topological cheap estimate recurrence, same engine-weight multiplication/child-addition order, computed directly from RelaxationBucket without constructing RelaxationBoundComputer regions. Region computers are created lazily if a region-based start or window improvement needs one; existing bound computers can still supply window partitions. This is heuristic-only, with exact-vector comparisons against existing prepared cheapEstimates in every randomized suite weighting. No bound/pruning/saturation alternatives change. The streaming callback already validated each unique start; remove the redundant immediate revalidation after construction. Aim: save enough preparation time for N=2 extraction/write to fit30 s.

### I030_tool_tests
- In progress: 2026-10-21T13:07:02.114374-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I030_tool_tests.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I030_tool_tests.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I030_tool_tests.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_experiment`
- Completed: 2026-10-21T13:07:02.177376-04:00; exit 1; runner elapsed 0.061 s.

### I031_tool_tests
- In progress: 2026-10-21T13:07:13.554981-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I031_tool_tests.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I031_tool_tests.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I031_tool_tests.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_experiment`
- Completed: 2026-10-21T13:07:13.618394-04:00; exit 0; runner elapsed 0.061 s.
- Completed: 2026-10-21T13:08:18.186010-04:00; exit 0; runner elapsed 138.306 s.

### I032_lazy_regions_oracle
- In progress: 2026-10-21T13:08:23.690426-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I032_lazy_regions_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I032_lazy_regions_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I032_lazy_regions_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T13:08:28.117801-04:00; exit 0; runner elapsed 4.425 s.
- Oracle verdict: 48/48 cases passed.

### I033_lazy_regions_n2
- In progress: 2026-10-21T13:08:31.886328-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I033_lazy_regions_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I033_lazy_regions_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I033_lazy_regions_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I033_lazy_regions_n2`
- Completed: 2026-10-21T13:09:36.573652-04:00; exit 0; runner elapsed 64.685 s.
  `[SearchEngine volume] iterations=0 elapsed=8.7710632299999993 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.996706018183424 normalized=0.99989020060611411 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=25.086477518081665 samples=4`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792602556.8434038 program-deadline=1792602558.7569392`
- First incumbent: 25.086477518 s, cost 91.3422; best cost 91.3422.

### I034_lazy_regions_n2_confirm1
- In progress: 2026-10-21T13:09:53.423618-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I034_lazy_regions_n2_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I034_lazy_regions_n2_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I034_lazy_regions_n2_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I034_lazy_regions_n2_confirm1`
- Completed: 2026-10-21T13:10:58.305902-04:00; exit 0; runner elapsed 64.880 s.
  `[SearchEngine volume] iterations=0 elapsed=8.8486635860000007 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99668168781854 normalized=0.99988938959395135 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=25.050184726715088 samples=4`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792602638.3757684 program-deadline=1792602640.3256061`
- First incumbent: 25.050184727 s, cost 91.3422; best cost 91.3422.

### I035_preparation_deadline_build
- In progress: 2026-10-21T13:11:55.772440-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I035_preparation_deadline_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I035_preparation_deadline_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I035_preparation_deadline_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I029 C003 build passed; I032 oracle48/48 passed. I030 runner regression initially exposed a pre-existing regex bug: gemma_lines did not match `[Gemma compile] supervisor` (expected an extra closing bracket). Fixed regex; I031 all3 runner tests pass. On killed compile without authoritative gap, incumbent retains cost/raw event but elapsed=None; progress printing explicitly states unavailable.

I033 C003 **N=2 passes** exact5-graph logits check: compile29.827 s, first incumbent25.086478 s, finalcost91.342201, bound0.061235, max_abs_err8.106e-05. Saturation12.2453 s, searchstart19.5128 s, rootprop1.67841 s, heuristicbuild+validation3.66186 s (first validation0.334364 rejected, second2.88254 accepted). I034 identical-build confirmation also passes: compile29.848 s, firstincumbent25.050185 s. **Do not accept yet**: both reveal deferred relaxation preparation takes~3.1s after only~1s remained before its deadline, crossing searchdeadline without producing a useful bound (searchdone28.2838 s). Fixed contract asks deferred work to respect deadline; need interruptible preparation even though30s eval passed. Supersede C003 with C004 before second confirmation/gates.

### Candidate C004: interruptible relaxation preparation

RelaxationBoundComputer accepts optional steady-clock deadline (default unlimited). prepare checks at major work-unit loops and returns false on exhaustion; constructor leaves ready=false and partially built regions are never evaluated/exposed as a bound. Root-bound preparation logs expiry and leaves its boundzero; later refinement skips unready computers. Window heuristic preparation also receives its deadline and skips unready computers. Existing fully prepared path remains identical. An expired-constructor native regression asserts unready, alongside expiredmaximizationzero and exact cheap-estimate comparisons. Soundness: no partial decomposition is used, and zero remains <= every feasible plan cost. Goal: preserve first-incumbent improvement and finish N=2 well inside30s.

### I036_tool_and_gap_tests
- In progress: 2026-10-21T13:12:35.171118-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I036_tool_and_gap_tests.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I036_tool_and_gap_tests.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I036_tool_and_gap_tests.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m unittest tests.test_search_experiment tests.test_search_gap`
- Completed: 2026-10-21T13:12:35.241142-04:00; exit 0; runner elapsed 0.068 s.
- Completed: 2026-10-21T13:14:15.355166-04:00; exit 0; runner elapsed 139.580 s.

### I037_ccache_reuse_build
- In progress: 2026-10-21T13:14:43.785992-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I037_ccache_reuse_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I037_ccache_reuse_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I037_ccache_reuse_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T13:14:45.063746-04:00; exit 0; runner elapsed 1.276 s.

### I038_preparation_deadline_oracle
- In progress: 2026-10-21T13:14:53.647564-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I038_preparation_deadline_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I038_preparation_deadline_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I038_preparation_deadline_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T13:14:58.016164-04:00; exit 0; runner elapsed 4.365 s.
- Oracle verdict: 48/48 cases passed.

### I039_preparation_deadline_n2
- In progress: 2026-10-21T13:15:02.985713-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I039_preparation_deadline_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I039_preparation_deadline_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I039_preparation_deadline_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I039_preparation_deadline_n2`
- Completed: 2026-10-21T13:16:08.012938-04:00; exit 0; runner elapsed 65.025 s.
  `[SearchEngine volume] iterations=1 elapsed=8.345491913 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.1336149713 log2-remaining-fraction=-0.22270139586180449 eliminated-bits=0.22270139586180449 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.997060453882185 normalized=0.99990201512940613 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=25.615177392959595 samples=5`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792602948.9236293 program-deadline=1792602950.308465`
- First incumbent: 25.615177393 s, cost 91.3422; best cost 91.342201.

### I040_safe_margin_n2
- In progress: 2026-10-21T13:16:40.905753-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I040_safe_margin_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I040_safe_margin_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I040_safe_margin_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I040_safe_margin_n2`

I035 final C004 build passed in139.580s. I037 exact-header-content cache replay (touch without editing) rebuilt in1.276s; SHA256 of binding/test/bench binaries identical before/after (saved I037_binary_hashes_before.json). This verifies ccache accelerates repeated source states without changing binaries. I036 all4 reporting regressions/search-gap tests passed; Linux runner metadata now includes child_peak_rss_kib (largest single child-process RSS across calibration/compile/check, not concurrent-process sum), and finalcost uses authoritative gap summary. I038 oracle48/48 passed on identical final C++ binary.

I039 C004 N=2 passed all5 logits graphs, compile29.823s, firstincumbent25.615177s, finalcost91.342201. Region preparation expired cleanly in0.66858s, producing zerobound (no partial decomposition used). Subsequent ordinary search iteration took~2.25s, crossing deadline before next boundary check; planner searchdone28.3013s. No bound/pruning correctness issue, but only0.18s total-budget headroom.

### Candidate C005: measured extraction margin

Keep C004 C++ implementation; change only explicitly editable SEARCH_MARGIN_SECONDS from3.0 to5.0. Evidence: one indivisible search iteration~2.25s plus extraction/arena/write~1.5s needs at least3.75s margin.5.0s reserves headroom and avoids starting unhelpful proof work immediately before extraction. Total compile budget remains30s, all2N+1 buckets and exact check unchanged; no evaluator edits except permitted constant. I040 evaluation in progress.
- Completed: 2026-10-21T13:17:42.020547-04:00; exit 0; runner elapsed 61.113 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4096717439999997 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.996616738436703 normalized=0.99988722461455681 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=24.953301906585693 samples=4`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792603045.8669848 program-deadline=1792603045.9136961`
- First incumbent: 24.953301907 s, cost 91.3422; best cost 91.342201.

### I041_safe_margin_n2_confirm1
- In progress: 2026-10-21T13:17:47.162736-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I041_safe_margin_n2_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I041_safe_margin_n2_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I041_safe_margin_n2_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I041_safe_margin_n2_confirm1`
- Completed: 2026-10-21T13:18:48.234778-04:00; exit 0; runner elapsed 61.070 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4258410880000003 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.996714869453903 normalized=0.99989049564846344 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=25.099680662155151 samples=4`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792603111.9538581 program-deadline=1792603111.8541903`
- First incumbent: 25.099680662 s, cost 91.3422; best cost 91.342201.

### I042_safe_margin_n2_confirm2
- In progress: 2026-10-21T13:19:00.893495-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I042_safe_margin_n2_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I042_safe_margin_n2_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I042_safe_margin_n2_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I042_safe_margin_n2_confirm2`
- Completed: 2026-10-21T13:20:02.101388-04:00; exit 0; runner elapsed 61.206 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4118483380000004 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.996713869853963 normalized=0.99989046232846546 final-gap=0.99932960888472577 final-incumbent=91.342201000000003 final-bound=0.061234999999999998 first-incumbent-elapsed=25.09818959236145 samples=4`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792603185.8998241 program-deadline=1792603185.8016489`
- First incumbent: 25.098189592 s, cost 91.3422; best cost 91.342201.

### I043_safe_margin_n1_regression
- In progress: 2026-10-21T13:20:06.723082-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I043_safe_margin_n1_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I043_safe_margin_n1_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I043_safe_margin_n1_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 1 --ignore-checkpoints --run-dir logs/gemma_output_runs/I043_safe_margin_n1_regression`
- Completed: 2026-10-21T13:20:50.126815-04:00; exit 0; runner elapsed 43.402 s.
  `[SearchEngine volume] iterations=21 elapsed=22.473117592000001 total-domain-bits=1604487.9395109841 remaining-domain-bits=1604322.9990325295 log2-remaining-fraction=-164.94047845457681 eliminated-bits=164.94047845457681 remaining-nodes=22 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=26.880455357798279 normalized=0.89601517859327595 final-gap=0.84815456787562382 final-incumbent=86.653037999999995 final-bound=13.157868000000001 first-incumbent-elapsed=5.2249445915222168 samples=13`
  `[SearchEngine] New best cost: 87.3138 at iteration 0 source=root-heuristic event-epoch=1792603215.1335232 program-deadline=1792603234.908592`
- First incumbent: 5.224944592 s, cost 87.3138; best cost 86.653038.

### I045_write_after_read_cache_gate
- In progress: 2026-10-21T13:20:59.272918-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I045_write_after_read_cache_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I045_write_after_read_cache_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I045_write_after_read_cache_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`

### I044_private_layout_gate
- In progress: 2026-10-21T13:20:59.273021-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I044_private_layout_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I044_private_layout_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I044_private_layout_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`

### I049_relaxation_bound_gate
- In progress: 2026-10-21T13:20:59.274639-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I049_relaxation_bound_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I049_relaxation_bound_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I049_relaxation_bound_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`

### I047_start_rank_window_gate
- In progress: 2026-10-21T13:20:59.274274-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I047_start_rank_window_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I047_start_rank_window_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I047_start_rank_window_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`

### I046_incremental_propagators_gate
- In progress: 2026-10-21T13:20:59.278009-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I046_incremental_propagators_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I046_incremental_propagators_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I046_incremental_propagators_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`

### I048_selection_reachability_gate
- In progress: 2026-10-21T13:20:59.291446-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I048_selection_reachability_gate.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I048_selection_reachability_gate.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I048_selection_reachability_gate.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T13:20:59.908002-04:00; exit 0; runner elapsed 0.631 s.
- Completed: 2026-10-21T13:20:59.931652-04:00; exit 0; runner elapsed 0.652 s.
- Completed: 2026-10-21T13:20:59.942460-04:00; exit 0; runner elapsed 0.668 s.
- Completed: 2026-10-21T13:20:59.958594-04:00; exit 0; runner elapsed 0.684 s.
- Completed: 2026-10-21T13:21:00.000802-04:00; exit 0; runner elapsed 0.706 s.
- Completed: 2026-10-21T13:22:50.013605-04:00; exit 0; runner elapsed 110.737 s.

C005 evaluations I040/I041/I042 all passed all5 exact logits graphs, compile26.239/26.352/26.345s; firstincumbent24.953302/25.099681/25.098190s. I043 prior-N* regression passed all3 graphs, compile25.928s, firstincumbent5.224945s. All4 recorded source patch SHA256 values equal `f7c0b50992248d9bf9af2113c82b1adef839f85002a3244c8314dee58b9fe508`; no intervening build. Linux largest-child peak RSS3094.906MiB and3108.336MiB for I040/I041. I044-I048 native gates all passed. I049 randomized bound/cheap-estimate/deadline suite still active; acceptance pending it. Comparison-table intermediate metrics corrected from runner gap summaries (results.json uses a different gap-key schema).

### C005 acceptance (2026-10-21)

Accepted **scaling** change, N*1->2, not completion of the N32768 goal. I040 reference: firstincumbent24.953302s, compile26.239s, cost91.342201; I041/I042 identical-source/build confirmations first25.099681/25.098190s compile26.352/26.345s. Every5-graph logits check passed, max_abs_err8.106e-05. I043 accepted-old-N* regression: N1 first5.224945s compile25.928s, all3 graphs passed. Oracle48/48 and all6 native suites passed (I049104361 weighted checks, plus exact cheap-estimate vector comparisons and expired preparation/maximization checks); search-gap and4 runner tests passed. Bound argument and adversarial cases recorded under C002-C004. Saturation alternatives, buckets, cost model/old records, evaluator checks/tolerances remain unchanged. New cost records came only from baseline calibration. Final C005 snapshot is sufficient to resume/rebuild without conversation.

### I050_accepted_frontier_n3
- In progress: 2026-10-21T13:23:54.429058-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I050_accepted_frontier_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I050_accepted_frontier_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I050_accepted_frontier_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I050_accepted_frontier_n3`
- Completed: 2026-10-21T13:24:57.412540-04:00; exit 1; runner elapsed 62.981 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

I050 accepted-C005 frontier N=3: calibration pending0; saturation20.4999s (includes base-equality reconciliation2.16972s), reduction not finished by30.128s supervisor timeout/exit-9. No search, incumbent, plan or exact graph check. Honest miss; N*=2 remains. Instrumentation-free evaluation, same C005 source/build as accepted runs. Next separate timing diagnostic I051 uses pre-existing TG_TIME_BUCKETS only and cannot count as an evaluation improvement.

### I051_diagnostic_bucket_timing_n3
- In progress: 2026-10-21T13:26:00.889045-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I051_diagnostic_bucket_timing_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I051_diagnostic_bucket_timing_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I051_diagnostic_bucket_timing_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I051_diagnostic_bucket_timing_n3`
- Completed: 2026-10-21T13:27:03.536207-04:00; exit 1; runner elapsed 62.645 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I052_post_reduction_estimates_build
- In progress: 2026-10-21T13:28:03.685154-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I052_post_reduction_estimates_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I052_post_reduction_estimates_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I052_post_reduction_estimates_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I051 phase diagnostic completed: compile30.125s timeout, saturation20.5258s. Calibration phase timing: full rewrite1.905s; partial rewrite durations4.312/8.897/9.547/14.846/16.215s in parallel. Partial postprocessing20-31ms, copying/injection~2ms. Confirms rewrite saturation dominates; no value in optimizing clean-class postprocessing first. Diagnostic-only; no N* credit.

### Candidate C006: branch estimates after reduction

computeENodeInfos optionally omits computeENodeHeuristicCosts. Initial reduction requests costs+existing domination only; pruneEGraph uses cost==INF, cycle passes take only the graph. Build the estimates once after final graph reduction (existing full computeENodeInfos if rebuilt, computeENodeHeuristicCosts otherwise). No change to cost estimates, dominance checks, cycle passes or saturation alternatives. Branch-order estimates are heuristics, not certified bounds. This removes work on alternatives that reduction later drops and avoids computing the same heuristic stage twice per bucket. Aim: reduce reduction wall time enough for N=3, or improve first incumbent at accepted N=2. Source I052 snapshot includes candidate; accepted C005 snapshot remains I040.
- Completed: 2026-10-21T13:30:21.089696-04:00; exit 0; runner elapsed 137.402 s.

### I053_post_reduction_estimates_oracle
- In progress: 2026-10-21T13:30:24.094997-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I053_post_reduction_estimates_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I053_post_reduction_estimates_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I053_post_reduction_estimates_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T13:30:28.516157-04:00; exit 0; runner elapsed 4.419 s.
- Oracle verdict: 48/48 cases passed.

### I054_post_reduction_estimates_n3
- In progress: 2026-10-21T13:30:32.114077-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I054_post_reduction_estimates_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I054_post_reduction_estimates_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I054_post_reduction_estimates_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I054_post_reduction_estimates_n3`
- Completed: 2026-10-21T13:31:29.615981-04:00; exit 1; runner elapsed 57.498 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I055_post_reduction_estimates_n2
- In progress: 2026-10-21T13:31:59.333676-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I055_post_reduction_estimates_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I055_post_reduction_estimates_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I055_post_reduction_estimates_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I055_post_reduction_estimates_n2`

I052 C006 default build passed; I053 oracle48/48 passed. I054 N=3 **miss** (30.115s supervisor kill, no incumbent), but saturation20.4487s, reduction+setup6.3477s, searchstart26.7964s, rootprop2.7857s; N=3 now reaches search instead of dying during reduction. This is evidence of saved reduction work, not accepted metric improvement yet. I055 evaluates accepted N=2 speed tie-breaker. No native gates/confirmations run for C006 yet.
- Completed: 2026-10-21T13:32:57.937575-04:00; exit 0; runner elapsed 58.601 s.
  `[SearchEngine volume] iterations=0 elapsed=8.2254109240000002 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.906994021232681 normalized=0.99689980070775597 final-gap=0.98448353570985214 final-incumbent=91.342201000000003 final-bound=1.417308 first-incumbent-elapsed=22.443355083465576 samples=5`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792603958.9116168 program-deadline=1792603961.4682739`
- First incumbent: 22.443355083 s, cost 91.3422; best cost 91.342201.

### I056_post_reduction_estimates_n2_confirm1
- In progress: 2026-10-21T13:33:37.954481-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I056_post_reduction_estimates_n2_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I056_post_reduction_estimates_n2_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I056_post_reduction_estimates_n2_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I056_post_reduction_estimates_n2_confirm1`

I055 C006 N=2 passes all5 exact logits graphs, max_abs_err8.106e-05; firstincumbent22.443355s vs C00524.953302s (2.509947s /10.06% earlier), finalcost91.342201 unchanged, compile26.451s. Saturation12.1746s, reduction+setup4.5780s vs C005~7.3s, rootprop1.6838s. I056 first identical-source/build confirmation running. Candidate not accepted until confirmation2/native/search-gap gates; oracle48/48 already passed I053.
- Completed: 2026-10-21T13:34:36.689051-04:00; exit 0; runner elapsed 58.733 s.
  `[SearchEngine volume] iterations=0 elapsed=8.1024744710000007 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.906285951225883 normalized=0.99687619837419605 final-gap=0.98448353570985214 final-incumbent=91.342201000000003 final-bound=1.417308 first-incumbent-elapsed=22.392718315124512 samples=5`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792604057.6840959 program-deadline=1792604060.2913909`
- First incumbent: 22.392718315 s, cost 91.3422; best cost 91.342201.

### I057_post_reduction_estimates_n2_confirm2
- In progress: 2026-10-21T13:34:39.859721-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I057_post_reduction_estimates_n2_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I057_post_reduction_estimates_n2_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I057_post_reduction_estimates_n2_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I057_post_reduction_estimates_n2_confirm2`
- Completed: 2026-10-21T13:35:38.753925-04:00; exit 0; runner elapsed 58.891 s.
  `[SearchEngine volume] iterations=0 elapsed=8.2259811840000001 total-domain-bits=5380634.3563163672 remaining-domain-bits=5380634.3560073934 log2-remaining-fraction=-0.00030897371470928192 eliminated-bits=0.00030897371470928192 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.904873411379736 normalized=0.99682911371265781 final-gap=0.98448353570985214 final-incumbent=91.342201000000003 final-bound=1.417308 first-incumbent-elapsed=22.230959177017212 samples=5`
  `[SearchEngine] New best cost: 91.3422 at iteration 0 source=root-heuristic event-epoch=1792604119.6609395 program-deadline=1792604122.4299955`
- First incumbent: 22.230959177 s, cost 91.3422; best cost 91.342201.

### I058_c006_private_layout
- In progress: 2026-10-21T13:35:55.695772-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I058_c006_private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I058_c006_private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I058_c006_private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`

### I059_c006_write_after_read_cache
- In progress: 2026-10-21T13:35:55.698643-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I059_c006_write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I059_c006_write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I059_c006_write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`

### I060_c006_incremental_propagators
- In progress: 2026-10-21T13:35:55.698967-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I060_c006_incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I060_c006_incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I060_c006_incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`

### I063_c006_relaxation_bound
- In progress: 2026-10-21T13:35:55.699368-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I063_c006_relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I063_c006_relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I063_c006_relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`

### I061_c006_start_rank_window
- In progress: 2026-10-21T13:35:55.703665-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I061_c006_start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I061_c006_start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I061_c006_start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`

### I062_c006_selection_reachability
- In progress: 2026-10-21T13:35:55.708970-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I062_c006_selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I062_c006_selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I062_c006_selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T13:35:56.292139-04:00; exit 0; runner elapsed 0.587 s.
- Completed: 2026-10-21T13:35:56.319116-04:00; exit 0; runner elapsed 0.618 s.
- Completed: 2026-10-21T13:35:56.342921-04:00; exit 0; runner elapsed 0.642 s.
- Completed: 2026-10-21T13:35:56.350571-04:00; exit 0; runner elapsed 0.653 s.
- Completed: 2026-10-21T13:35:56.393912-04:00; exit 0; runner elapsed 0.681 s.

### I064_c006_gap_tools
- In progress: 2026-10-21T13:35:56.864069-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I064_c006_gap_tools.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I064_c006_gap_tools.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I064_c006_gap_tools.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m unittest tests.test_search_gap tests.test_search_experiment`
- Completed: 2026-10-21T13:35:56.924013-04:00; exit 0; runner elapsed 0.058 s.

I056/I057 identical-source/build confirmations passed all5 logits graphs. Firstincumbent22.392718/22.230959s, compile26.391/26.379s. All3 C006 evaluation patch SHA256 identical: 3e36805e57e85190f032d7163d393735a089c2f68ca58517a78f6e3eae017b0b. I058-I0625 native gates and I064 search-gap/reporting tests passed; I063 randomized suite still running. No timed workload overlapped tests/builds. C006 acceptance pending only I063.
- Completed: 2026-10-21T13:37:46.534894-04:00; exit 0; runner elapsed 110.833 s.

### C006 acceptance

Accepted **speed** change with N*=2 unchanged: firstincumbent22.443355s vs C00524.953302s (2.509947s earlier). Confirmations22.392718/22.230959s, all5 graph checks pass; finalcost91.342201 unchanged. Compile26.451/26.391/26.379s. All correctness gates complete: I05348/48 oracle, I058-I0636 native suites (I063104361 weighted/23666 exact pruning checks plus deadline/estimate tests), I064 search-gap/reporting. No bounds/pruning rules changed; source argument under C006. N=3 miss I054 remains; continue scaling, do not mark full goal complete.

### I065_diagnostic_rule_timing_n3
- In progress: 2026-10-21T13:38:26.947679-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I065_diagnostic_rule_timing_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I065_diagnostic_rule_timing_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I065_diagnostic_rule_timing_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I065_diagnostic_rule_timing_n3`
- Completed: 2026-10-21T13:39:25.163383-04:00; exit 1; runner elapsed 58.213 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I065 rule-timing diagnostic

Diagnostic TG_TIME_RULES=1 run completed, N=3 budget miss (not an evaluation improvement). Calibration profiling (full plus5 distinct partials): sweeps17/67/66/66/76/77; rebuild times0.518/1.877/3.446/3.417/6.386/7.642s; sweeps1.345/1.430/4.690/5.376/7.598/7.232s; filter0.074/0.625/0.883/0.910/1.237/1.341s. Slowest FusionRule total3.730s (match2.255/apply1.475), another3.796s (match2.452/apply1.344); RemoveCopyChains<=0.927s, RemoveContiguous<=0.744s. Rebuild is a major independent hot spot, not just rule matching. Current EGraph::rebuild line913 allocates a fresh unordered_map<uint64_t,vector<ENodeId>> every congruence iteration, reserves2*enodes, and each unique signature pushes one entry into a heap vector. Potential allocation optimization must keep equality checks for64-bit signature collisions and all existing enode alternatives; no pruning. No new candidate edits yet; accepted source/build C006 unchanged.

### I066_inline_hashcons_build
- In progress: 2026-10-21T13:42:52.611207-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I066_inline_hashcons_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I066_inline_hashcons_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I066_inline_hashcons_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I066 was interrupted before completing a build after the optimization direction shifted to sharing work between buckets. The incomplete inline hash-cons experiment was discarded; it produced no evaluated candidate.

### I067_incremental_bucket_saturation_build
- In progress: 2026-10-21T13:49:44.433681-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I067_incremental_bucket_saturation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I067_incremental_bucket_saturation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I067_incremental_bucket_saturation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T13:55:43.967693-04:00; exit 0; runner elapsed 359.532 s.

### I068_incremental_bucket_saturation_n3
- In progress: 2026-10-21T13:55:47.585935-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I068_incremental_bucket_saturation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I068_incremental_bucket_saturation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I068_incremental_bucket_saturation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python --env TG_TIME_BUCKETS=1 -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I068_incremental_bucket_saturation_n3`
- Completed: 2026-10-21T13:55:47.590380-04:00; exit 2; runner elapsed 0.003 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I069_incremental_bucket_saturation_n3
- In progress: 2026-10-21T13:55:51.643252-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I069_incremental_bucket_saturation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I069_incremental_bucket_saturation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I069_incremental_bucket_saturation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -- -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I069_incremental_bucket_saturation_n3`
- Completed: 2026-10-21T13:55:51.661787-04:00; exit 2; runner elapsed 0.015 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I070_incremental_bucket_saturation_n3
- In progress: 2026-10-21T13:55:55.311103-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I070_incremental_bucket_saturation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I070_incremental_bucket_saturation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I070_incremental_bucket_saturation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I070_incremental_bucket_saturation_n3`
- Completed: 2026-10-21T13:58:36.482654-04:00; exit 1; runner elapsed 161.169 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I071_shared_partial_bucket_closure_build
- In progress: 2026-10-21T14:00:08.178169-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I071_shared_partial_bucket_closure_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I071_shared_partial_bucket_closure_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I071_shared_partial_bucket_closure_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:02:25.077880-04:00; exit 0; runner elapsed 136.898 s.

### I072_shared_partial_bucket_closure_n3
- In progress: 2026-10-21T14:02:28.389694-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I072_shared_partial_bucket_closure_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I072_shared_partial_bucket_closure_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I072_shared_partial_bucket_closure_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I072_shared_partial_bucket_closure_n3`
- Completed: 2026-10-21T14:03:57.289956-04:00; exit 1; runner elapsed 88.898 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I073_shared_partial_bucket_closure_n2
- In progress: 2026-10-21T14:04:25.081989-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I073_shared_partial_bucket_closure_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I073_shared_partial_bucket_closure_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I073_shared_partial_bucket_closure_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I073_shared_partial_bucket_closure_n2`
- Completed: 2026-10-21T14:05:37.160337-04:00; exit 0; runner elapsed 72.076 s.
  `[SearchEngine volume] iterations=0 elapsed=9.8012756319999994 total-domain-bits=3110937.5748411044 remaining-domain-bits=3110937.5743563026 log2-remaining-fraction=-0.00048480182886123657 eliminated-bits=0.00048480182886123657 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.967638605072384 normalized=0.96558795350241278 final-gap=0.90061105168093591 final-incumbent=91.380814000000001 final-bound=9.0822430000000001 first-incumbent-elapsed=18.343996047973633 samples=5`
  `[SearchEngine] New best cost: 91.3808 at iteration 0 source=root-heuristic event-epoch=1792605914.4217339 program-deadline=1792605921.0777528`
- First incumbent: 18.343996048 s, cost 91.3808; best cost 91.380814.

### I074_shared_partial_bucket_closure_n2_confirm1
- In progress: 2026-10-21T14:05:44.342799-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I074_shared_partial_bucket_closure_n2_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I074_shared_partial_bucket_closure_n2_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I074_shared_partial_bucket_closure_n2_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I074_shared_partial_bucket_closure_n2_confirm1`
- Completed: 2026-10-21T14:06:41.266523-04:00; exit 0; runner elapsed 56.922 s.
  `[SearchEngine volume] iterations=0 elapsed=9.8264413099999999 total-domain-bits=3110937.5748411044 remaining-domain-bits=3110937.5743563026 log2-remaining-fraction=-0.00048480182886123657 eliminated-bits=0.00048480182886123657 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.972605102111348 normalized=0.96575350340371158 final-gap=0.90061105168093591 final-incumbent=91.380814000000001 final-bound=9.0822430000000001 first-incumbent-elapsed=18.403717756271362 samples=5`
  `[SearchEngine] New best cost: 91.3808 at iteration 0 source=root-heuristic event-epoch=1792605978.4518809 program-deadline=1792605985.048177`
- First incumbent: 18.403717756 s, cost 91.3808; best cost 91.380814.

### I075_shared_partial_bucket_closure_n2_confirm2
- In progress: 2026-10-21T14:06:43.677966-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I075_shared_partial_bucket_closure_n2_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I075_shared_partial_bucket_closure_n2_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I075_shared_partial_bucket_closure_n2_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I075_shared_partial_bucket_closure_n2_confirm2`
- Completed: 2026-10-21T14:07:40.577203-04:00; exit 0; runner elapsed 56.897 s.
  `[SearchEngine volume] iterations=0 elapsed=9.7833476939999997 total-domain-bits=3110937.5748411044 remaining-domain-bits=3110937.5743563026 log2-remaining-fraction=-0.00048480182886123657 eliminated-bits=0.00048480182886123657 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.966758582492993 normalized=0.96555861941643306 final-gap=0.90061105168093591 final-incumbent=91.380814000000001 final-bound=9.0822430000000001 first-incumbent-elapsed=18.348831653594971 samples=5`
  `[SearchEngine] New best cost: 91.3808 at iteration 0 source=root-heuristic event-epoch=1792606037.6466546 program-deadline=1792606044.2978361`
- First incumbent: 18.348831654 s, cost 91.3808; best cost 91.380814.

### I076_shared_closure_oracle
- In progress: 2026-10-21T14:07:48.600696-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I076_shared_closure_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I076_shared_closure_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I076_shared_closure_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`

### I078_shared_closure_write_after_read
- In progress: 2026-10-21T14:07:48.603213-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I078_shared_closure_write_after_read.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I078_shared_closure_write_after_read.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I078_shared_closure_write_after_read.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`

### I077_shared_closure_private_layout
- In progress: 2026-10-21T14:07:48.603980-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I077_shared_closure_private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I077_shared_closure_private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I077_shared_closure_private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`

### I079_shared_closure_incremental
- In progress: 2026-10-21T14:07:48.607967-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I079_shared_closure_incremental.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I079_shared_closure_incremental.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I079_shared_closure_incremental.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`

### I082_shared_closure_relaxation_bound
- In progress: 2026-10-21T14:07:48.638464-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I082_shared_closure_relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I082_shared_closure_relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I082_shared_closure_relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`

### I080_shared_closure_start_rank
- In progress: 2026-10-21T14:07:48.644056-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I080_shared_closure_start_rank.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I080_shared_closure_start_rank.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I080_shared_closure_start_rank.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`

### I083_shared_closure_gap_tools
- In progress: 2026-10-21T14:07:48.646696-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I083_shared_closure_gap_tools.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I083_shared_closure_gap_tools.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I083_shared_closure_gap_tools.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m unittest tests.test_search_gap tests.test_search_experiment`

### I081_shared_closure_reachability
- In progress: 2026-10-21T14:07:48.646762-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I081_shared_closure_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I081_shared_closure_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I081_shared_closure_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T14:07:48.712234-04:00; exit 0; runner elapsed 0.063 s.
- Completed: 2026-10-21T14:07:49.179842-04:00; exit 0; runner elapsed 0.570 s.
- Completed: 2026-10-21T14:07:49.187355-04:00; exit 0; runner elapsed 0.581 s.
- Completed: 2026-10-21T14:07:49.194978-04:00; exit 0; runner elapsed 0.590 s.
- Completed: 2026-10-21T14:07:49.224347-04:00; exit 0; runner elapsed 0.578 s.
- Completed: 2026-10-21T14:07:49.263046-04:00; exit 0; runner elapsed 0.614 s.
- Completed: 2026-10-21T14:07:53.007421-04:00; exit 0; runner elapsed 4.405 s.
- Oracle verdict: 48/48 cases passed.
- Completed: 2026-10-21T14:09:38.372826-04:00; exit 0; runner elapsed 109.732 s.

### I084_region_family_saturation_build
- In progress: 2026-10-21T14:11:52.567134-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I084_region_family_saturation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I084_region_family_saturation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I084_region_family_saturation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:14:09.655049-04:00; exit 0; runner elapsed 137.074 s.

### I085_region_family_saturation_n3
- In progress: 2026-10-21T14:14:12.551289-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I085_region_family_saturation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I085_region_family_saturation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I085_region_family_saturation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I085_region_family_saturation_n3`
- Completed: 2026-10-21T14:15:10.336308-04:00; exit 1; runner elapsed 57.783 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I086_dirty_family_saturation_build
- In progress: 2026-10-21T14:15:27.311065-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I086_dirty_family_saturation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I086_dirty_family_saturation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I086_dirty_family_saturation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:17:44.373905-04:00; exit 0; runner elapsed 137.061 s.

### I087_dirty_family_saturation_n3
- In progress: 2026-10-21T14:17:47.227208-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I087_dirty_family_saturation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I087_dirty_family_saturation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I087_dirty_family_saturation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I087_dirty_family_saturation_n3`
- Completed: 2026-10-21T14:19:29.507392-04:00; exit 1; runner elapsed 102.278 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I088_dirty_origin_family_build
- In progress: 2026-10-21T14:19:50.654307-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I088_dirty_origin_family_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I088_dirty_origin_family_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I088_dirty_origin_family_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:22:08.113837-04:00; exit 0; runner elapsed 137.458 s.

### I089_dirty_origin_family_n3
- In progress: 2026-10-21T14:22:10.905283-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I089_dirty_origin_family_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I089_dirty_origin_family_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I089_dirty_origin_family_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I089_dirty_origin_family_n3`
- Completed: 2026-10-21T14:23:11.475786-04:00; exit 1; runner elapsed 60.568 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I090_shared_reconciliation_build
- In progress: 2026-10-21T14:24:03.705232-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I090_shared_reconciliation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I090_shared_reconciliation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I090_shared_reconciliation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:26:20.118937-04:00; exit 0; runner elapsed 136.411 s.

### I091_shared_reconciliation_bucket_independence
- In progress: 2026-10-21T14:26:22.163176-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I091_shared_reconciliation_bucket_independence.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I091_shared_reconciliation_bucket_independence.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I091_shared_reconciliation_bucket_independence.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --bucket-independence`
- Completed: 2026-10-21T14:26:22.626084-04:00; exit 1; runner elapsed 0.459 s.

### I092_shared_reconciliation_n3
- In progress: 2026-10-21T14:26:31.270647-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I092_shared_reconciliation_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I092_shared_reconciliation_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I092_shared_reconciliation_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I092_shared_reconciliation_n3`
- Completed: 2026-10-21T14:27:30.656424-04:00; exit 1; runner elapsed 59.384 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I093_shared_primal_build
- In progress: 2026-10-21T14:31:33.226051-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I093_shared_primal_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I093_shared_primal_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I093_shared_primal_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:33:49.958052-04:00; exit 0; runner elapsed 136.730 s.

### I094_shared_primal_n2
- In progress: 2026-10-21T14:33:52.814460-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I094_shared_primal_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I094_shared_primal_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I094_shared_primal_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I094_shared_primal_n2`
- Completed: 2026-10-21T14:34:49.645150-04:00; exit 0; runner elapsed 56.829 s.
  `[SearchEngine volume] iterations=0 elapsed=9.7098727199999999 total-domain-bits=3110937.5748411044 remaining-domain-bits=3110937.5743563026 log2-remaining-fraction=-0.00048480182886123657 eliminated-bits=0.00048480182886123657 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.978924344112286 normalized=0.96596414480374293 final-gap=0.90061105168093591 final-incumbent=91.380814000000001 final-bound=9.0822430000000001 first-incumbent-elapsed=18.447396516799927 samples=5`
  `[SearchEngine] New best cost: 91.3808 at iteration 0 source=root-heuristic event-epoch=1792607666.8250899 program-deadline=1792607673.377708`
- First incumbent: 18.447396517 s, cost 91.3808; best cost 91.380814.

### I095_shared_primal_n3
- In progress: 2026-10-21T14:34:52.742990-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I095_shared_primal_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I095_shared_primal_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I095_shared_primal_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I095_shared_primal_n3`
- Completed: 2026-10-21T14:35:52.243943-04:00; exit 1; runner elapsed 59.499 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I096_c007_rebuild
- In progress: 2026-10-21T14:37:29.498680-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I096_c007_rebuild.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I096_c007_rebuild.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I096_c007_rebuild.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:37:30.718829-04:00; exit 0; runner elapsed 1.217 s.

### I097_c007_bucket_timing_n3
- In progress: 2026-10-21T14:37:36.067260-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I097_c007_bucket_timing_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I097_c007_bucket_timing_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I097_c007_bucket_timing_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I097_c007_bucket_timing_n3`
- Completed: 2026-10-21T14:38:35.494930-04:00; exit 1; runner elapsed 59.425 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I098_c007_rule_timing_n3
- In progress: 2026-10-21T14:38:48.141858-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I098_c007_rule_timing_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I098_c007_rule_timing_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I098_c007_rule_timing_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I098_c007_rule_timing_n3`
- Completed: 2026-10-21T14:39:48.747931-04:00; exit 1; runner elapsed 60.604 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I099_reuse_rebuild_hash_build
- In progress: 2026-10-21T14:40:40.665595-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I099_reuse_rebuild_hash_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I099_reuse_rebuild_hash_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I099_reuse_rebuild_hash_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:46:39.718463-04:00; exit 0; runner elapsed 359.050 s.

### I100_reuse_rebuild_hash_n3
- In progress: 2026-10-21T14:46:41.727958-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I100_reuse_rebuild_hash_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I100_reuse_rebuild_hash_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I100_reuse_rebuild_hash_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I100_reuse_rebuild_hash_n3`
- Completed: 2026-10-21T14:47:41.481967-04:00; exit 1; runner elapsed 59.752 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I101_c007_rebuild_after_discard
- In progress: 2026-10-21T14:48:19.313224-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I101_c007_rebuild_after_discard.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I101_c007_rebuild_after_discard.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I101_c007_rebuild_after_discard.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:49:56.438185-04:00; exit 0; runner elapsed 97.123 s.

### I102_parallel_rebuild_build
- In progress: 2026-10-21T14:51:33.245585-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I102_parallel_rebuild_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I102_parallel_rebuild_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I102_parallel_rebuild_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T14:57:34.562921-04:00; exit 0; runner elapsed 361.315 s.

### I103_parallel_rebuild_n3
- In progress: 2026-10-21T14:57:42.893758-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I103_parallel_rebuild_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I103_parallel_rebuild_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I103_parallel_rebuild_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I103_parallel_rebuild_n3`
- Completed: 2026-10-21T14:58:40.762172-04:00; exit 1; runner elapsed 57.866 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I104_parallel_rebuild_n2
- In progress: 2026-10-21T14:58:46.609330-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I104_parallel_rebuild_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I104_parallel_rebuild_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I104_parallel_rebuild_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I104_parallel_rebuild_n2`
- Completed: 2026-10-21T14:59:44.148394-04:00; exit 0; runner elapsed 57.537 s.
  `[SearchEngine volume] iterations=0 elapsed=10.189802180999999 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=25.263638453038826 normalized=0.84212128176796086 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=18.474151372909546 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792609160.6359591 program-deadline=1792609167.1618211`
- First incumbent: 18.474151373 s, cost 17.5005; best cost 17.500534.

### I105_parallel_rebuild_oracle
- In progress: 2026-10-21T14:59:51.499998-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I105_parallel_rebuild_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I105_parallel_rebuild_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I105_parallel_rebuild_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T14:59:55.872547-04:00; exit 0; runner elapsed 4.369 s.
- Oracle verdict: 48/48 cases passed.

### I106_parallel_private_layout
- In progress: 2026-10-21T14:59:57.921326-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I106_parallel_private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I106_parallel_private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I106_parallel_private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-21T14:59:58.139155-04:00; exit 0; runner elapsed 0.216 s.

### I107_parallel_write_after_read
- In progress: 2026-10-21T15:00:00.669165-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I107_parallel_write_after_read.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I107_parallel_write_after_read.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I107_parallel_write_after_read.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-21T15:00:00.878398-04:00; exit 0; runner elapsed 0.207 s.

### I108_parallel_incremental_propagators
- In progress: 2026-10-21T15:00:02.423324-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I108_parallel_incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I108_parallel_incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I108_parallel_incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-21T15:00:02.625572-04:00; exit 0; runner elapsed 0.198 s.

### I109_parallel_start_rank
- In progress: 2026-10-21T15:00:04.401854-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I109_parallel_start_rank.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I109_parallel_start_rank.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I109_parallel_start_rank.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-21T15:00:04.615814-04:00; exit 0; runner elapsed 0.212 s.

### I110_parallel_selection_reachability
- In progress: 2026-10-21T15:00:06.367881-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I110_parallel_selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I110_parallel_selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I110_parallel_selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T15:00:06.680097-04:00; exit 0; runner elapsed 0.310 s.

### I111_parallel_relaxation_bound
- In progress: 2026-10-21T15:00:08.714471-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I111_parallel_relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I111_parallel_relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I111_parallel_relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-21T15:01:59.027548-04:00; exit 0; runner elapsed 110.309 s.

### I112_parallel_shared_state_build
- In progress: 2026-10-21T15:03:57.942159-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I112_parallel_shared_state_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I112_parallel_shared_state_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I112_parallel_shared_state_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:06:14.790285-04:00; exit 0; runner elapsed 136.846 s.

### I113_parallel_shared_state_n3
- In progress: 2026-10-21T15:06:16.959089-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I113_parallel_shared_state_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I113_parallel_shared_state_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I113_parallel_shared_state_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I113_parallel_shared_state_n3`
- Completed: 2026-10-21T15:07:12.392926-04:00; exit 1; runner elapsed 55.430 s.
  `[SearchEngine volume] iterations=0 elapsed=4.5364800299999999 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792609631.8677084 program-deadline=1792609627.2552867`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 109.449 (no supervisor gap summary); best cost 109.449.

### I114_parallel_shared_state_n2
- In progress: 2026-10-21T15:07:21.065910-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I114_parallel_shared_state_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I114_parallel_shared_state_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I114_parallel_shared_state_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I114_parallel_shared_state_n2`
- Completed: 2026-10-21T15:08:17.748650-04:00; exit 0; runner elapsed 56.681 s.
  `[SearchEngine volume] iterations=0 elapsed=10.695983298 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.878955866987663 normalized=0.82929852889958877 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.701275825500488 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792609673.5448947 program-deadline=1792609680.8436306`
- First incumbent: 17.701275826 s, cost 17.5005; best cost 17.500534.

### I115_parallel_shared_state_n2_confirm1
- In progress: 2026-10-21T15:08:21.705101-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I115_parallel_shared_state_n2_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I115_parallel_shared_state_n2_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I115_parallel_shared_state_n2_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I115_parallel_shared_state_n2_confirm1`
- Completed: 2026-10-21T15:09:18.319709-04:00; exit 0; runner elapsed 56.612 s.
  `[SearchEngine volume] iterations=0 elapsed=10.766595991999999 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.828968236102611 normalized=0.8276322745367537 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.545292139053345 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792609734.1276929 program-deadline=1792609741.5824137`
- First incumbent: 17.545292139 s, cost 17.5005; best cost 17.500534.

### I116_parallel_shared_state_n2_confirm2
- In progress: 2026-10-21T15:09:20.933557-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I116_parallel_shared_state_n2_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I116_parallel_shared_state_n2_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I116_parallel_shared_state_n2_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I116_parallel_shared_state_n2_confirm2`
- Completed: 2026-10-21T15:10:17.475213-04:00; exit 0; runner elapsed 56.540 s.
  `[SearchEngine volume] iterations=0 elapsed=10.877748508 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.794773562835847 normalized=0.82649245209452826 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.538002014160156 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792609793.1948369 program-deadline=1792609800.656846`
- First incumbent: 17.538002014 s, cost 17.5005; best cost 17.500534.

### I117_parallel_shared_oracle
- In progress: 2026-10-21T15:10:20.165760-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I117_parallel_shared_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I117_parallel_shared_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I117_parallel_shared_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T15:10:24.653612-04:00; exit 0; runner elapsed 4.486 s.
- Oracle verdict: 48/48 cases passed.

### I118_parallel_shared_private_layout
- In progress: 2026-10-21T15:10:26.421666-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I118_parallel_shared_private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I118_parallel_shared_private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I118_parallel_shared_private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-21T15:10:26.614567-04:00; exit 0; runner elapsed 0.191 s.

### I119_parallel_shared_write_after_read
- In progress: 2026-10-21T15:10:28.303293-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I119_parallel_shared_write_after_read.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I119_parallel_shared_write_after_read.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I119_parallel_shared_write_after_read.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-21T15:10:28.524948-04:00; exit 0; runner elapsed 0.220 s.

### I120_parallel_shared_incremental_propagators
- In progress: 2026-10-21T15:10:30.838305-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I120_parallel_shared_incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I120_parallel_shared_incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I120_parallel_shared_incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-21T15:10:31.036802-04:00; exit 0; runner elapsed 0.197 s.

### I121_parallel_shared_start_rank
- In progress: 2026-10-21T15:10:32.998695-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I121_parallel_shared_start_rank.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I121_parallel_shared_start_rank.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I121_parallel_shared_start_rank.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-21T15:10:33.219894-04:00; exit 0; runner elapsed 0.219 s.

### I122_parallel_shared_selection_reachability
- In progress: 2026-10-21T15:10:35.386829-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I122_parallel_shared_selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I122_parallel_shared_selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I122_parallel_shared_selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T15:10:35.663775-04:00; exit 0; runner elapsed 0.275 s.

### I123_parallel_shared_relaxation_bound
- In progress: 2026-10-21T15:10:37.709846-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I123_parallel_shared_relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I123_parallel_shared_relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I123_parallel_shared_relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-21T15:12:27.359282-04:00; exit 0; runner elapsed 109.648 s.

### I124_parallel_shared_search_gap
- In progress: 2026-10-21T15:13:05.113365-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I124_parallel_shared_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I124_parallel_shared_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I124_parallel_shared_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-21T15:13:05.141933-04:00; exit 0; runner elapsed 0.027 s.

### I125_c009_full_bucket_reconcile_build
- In progress: 2026-10-21T15:15:23.743010-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I125_c009_full_bucket_reconcile_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I125_c009_full_bucket_reconcile_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I125_c009_full_bucket_reconcile_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:17:40.978538-04:00; exit 0; runner elapsed 137.234 s.

### I126_c009_n2_after_reconcile_fix
- In progress: 2026-10-21T15:17:47.119368-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I126_c009_n2_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I126_c009_n2_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I126_c009_n2_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I126_c009_n2_after_reconcile_fix`
- Completed: 2026-10-21T15:18:43.775198-04:00; exit 0; runner elapsed 56.654 s.
  `[SearchEngine volume] iterations=0 elapsed=10.794344361 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.901694898977301 normalized=0.83005649663257663 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.735929012298584 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792610299.620641 program-deadline=1792610306.884726`
- First incumbent: 17.735929012 s, cost 17.5005; best cost 17.500534.

### I127_c009_n2_after_reconcile_fix_confirm1
- In progress: 2026-10-21T15:18:46.309784-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I127_c009_n2_after_reconcile_fix_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I127_c009_n2_after_reconcile_fix_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I127_c009_n2_after_reconcile_fix_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I127_c009_n2_after_reconcile_fix_confirm1`
- Completed: 2026-10-21T15:19:42.813750-04:00; exit 0; runner elapsed 56.502 s.
  `[SearchEngine volume] iterations=0 elapsed=10.628706466000001 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.948027403718015 normalized=0.83160091345726717 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.814730882644653 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792610358.7941799 program-deadline=1792610365.9794619`
- First incumbent: 17.814730883 s, cost 17.5005; best cost 17.500534.

### I128_c009_n2_after_reconcile_fix_confirm2
- In progress: 2026-10-21T15:19:44.822097-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I128_c009_n2_after_reconcile_fix_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I128_c009_n2_after_reconcile_fix_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I128_c009_n2_after_reconcile_fix_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I128_c009_n2_after_reconcile_fix_confirm2`
- Completed: 2026-10-21T15:20:41.339521-04:00; exit 0; runner elapsed 56.514 s.
  `[SearchEngine volume] iterations=0 elapsed=10.677490358 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.911376799358827 normalized=0.83037922664529418 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.755571365356445 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792610417.3419981 program-deadline=1792610424.5864396`
- First incumbent: 17.755571365 s, cost 17.5005; best cost 17.500534.

### I129_c009_oracle_after_reconcile_fix
- In progress: 2026-10-21T15:20:43.706275-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I129_c009_oracle_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I129_c009_oracle_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I129_c009_oracle_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-21T15:20:48.170781-04:00; exit 0; runner elapsed 4.463 s.
- Oracle verdict: 48/48 cases passed.

### I130_c009_private_layout_after_reconcile_fix
- In progress: 2026-10-21T15:20:50.887204-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I130_c009_private_layout_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I130_c009_private_layout_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I130_c009_private_layout_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-21T15:20:51.111471-04:00; exit 0; runner elapsed 0.222 s.

### I131_c009_write_after_read_after_reconcile_fix
- In progress: 2026-10-21T15:20:53.927473-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I131_c009_write_after_read_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I131_c009_write_after_read_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I131_c009_write_after_read_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-21T15:20:54.133746-04:00; exit 0; runner elapsed 0.203 s.

### I132_c009_incremental_propagators_after_reconcile_fix
- In progress: 2026-10-21T15:20:56.230129-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I132_c009_incremental_propagators_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I132_c009_incremental_propagators_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I132_c009_incremental_propagators_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-21T15:20:56.442233-04:00; exit 0; runner elapsed 0.210 s.

### I133_c009_start_rank_after_reconcile_fix
- In progress: 2026-10-21T15:20:58.553932-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I133_c009_start_rank_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I133_c009_start_rank_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I133_c009_start_rank_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-21T15:20:58.757604-04:00; exit 0; runner elapsed 0.202 s.

### I134_c009_selection_reachability_after_reconcile_fix
- In progress: 2026-10-21T15:21:00.689750-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I134_c009_selection_reachability_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I134_c009_selection_reachability_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I134_c009_selection_reachability_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-21T15:21:00.990704-04:00; exit 0; runner elapsed 0.299 s.

### I135_c009_relaxation_bound_after_reconcile_fix
- In progress: 2026-10-21T15:21:02.904462-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I135_c009_relaxation_bound_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I135_c009_relaxation_bound_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I135_c009_relaxation_bound_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-21T15:22:53.031088-04:00; exit 0; runner elapsed 110.124 s.

### I136_c009_search_gap_after_reconcile_fix
- In progress: 2026-10-21T15:22:57.183275-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I136_c009_search_gap_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I136_c009_search_gap_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I136_c009_search_gap_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-21T15:22:57.213519-04:00; exit 0; runner elapsed 0.027 s.

### I137_c009_n3_after_reconcile_fix
- In progress: 2026-10-21T15:23:01.113763-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I137_c009_n3_after_reconcile_fix.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I137_c009_n3_after_reconcile_fix.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I137_c009_n3_after_reconcile_fix.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I137_c009_n3_after_reconcile_fix`
- Completed: 2026-10-21T15:23:56.336269-04:00; exit 1; runner elapsed 55.220 s.
  `[SearchEngine volume] iterations=0 elapsed=4.5722494100000004 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792610635.7251885 program-deadline=1792610631.2350342`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 109.449 (no supervisor gap summary); best cost 109.449.

### I138_c009_share_reduced_storage_build
- In progress: 2026-10-21T15:24:52.737788-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I138_c009_share_reduced_storage_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I138_c009_share_reduced_storage_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I138_c009_share_reduced_storage_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:27:09.836621-04:00; exit 0; runner elapsed 137.097 s.

### I139_c009_share_reduced_storage_n3
- In progress: 2026-10-21T15:27:15.053611-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I139_c009_share_reduced_storage_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I139_c009_share_reduced_storage_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I139_c009_share_reduced_storage_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I139_c009_share_reduced_storage_n3`
- Completed: 2026-10-21T15:28:06.498377-04:00; exit 1; runner elapsed 51.442 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I140_c009_restore_build
- In progress: 2026-10-21T15:28:24.553141-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I140_c009_restore_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I140_c009_restore_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I140_c009_restore_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:28:25.810245-04:00; exit 0; runner elapsed 1.255 s.

### I141_fast_offset_completion_build
- In progress: 2026-10-21T15:30:20.841323-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I141_fast_offset_completion_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I141_fast_offset_completion_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I141_fast_offset_completion_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:32:37.200003-04:00; exit 0; runner elapsed 136.357 s.

### I142_fast_offset_completion_n3
- In progress: 2026-10-21T15:32:39.286071-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I142_fast_offset_completion_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I142_fast_offset_completion_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I142_fast_offset_completion_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I142_fast_offset_completion_n3`
- Completed: 2026-10-21T15:33:34.750329-04:00; exit 1; runner elapsed 55.462 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I143_fast_offset_completion_n2
- In progress: 2026-10-21T15:33:42.282206-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I143_fast_offset_completion_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I143_fast_offset_completion_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I143_fast_offset_completion_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I143_fast_offset_completion_n2`
- Completed: 2026-10-21T15:34:39.067217-04:00; exit 0; runner elapsed 56.783 s.
  `[SearchEngine volume] iterations=0 elapsed=10.988227695999999 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=25.240477980560147 normalized=0.84134926601867155 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=18.438538551330566 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792611255.5218492 program-deadline=1792611262.083322`
- First incumbent: 18.438538551 s, cost 17.5005; best cost 17.500534.

### I144_share_full_bucket_build
- In progress: 2026-10-21T15:36:06.065959-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I144_share_full_bucket_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I144_share_full_bucket_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I144_share_full_bucket_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test --log-level INFO`
- Completed: 2026-10-21T15:38:23.683295-04:00; exit 0; runner elapsed 137.615 s.

### I145_share_full_bucket_n2
- In progress: 2026-10-21T15:38:26.470503-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I145_share_full_bucket_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I145_share_full_bucket_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I145_share_full_bucket_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I145_share_full_bucket_n2`
- Completed: 2026-10-21T15:39:22.849590-04:00; exit 0; runner elapsed 56.377 s.
  `[SearchEngine volume] iterations=0 elapsed=10.53950427 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.940308065432369 normalized=0.83134360218107894 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.83130955696106 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792611539.102968 program-deadline=1792611546.2716713`
- First incumbent: 17.831309557 s, cost 17.5005; best cost 17.500534.

### I146_share_full_bucket_n3
- In progress: 2026-10-21T15:39:25.649856-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I146_share_full_bucket_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I146_share_full_bucket_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I146_share_full_bucket_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I146_share_full_bucket_n3`
- Completed: 2026-10-21T15:40:21.084038-04:00; exit 1; runner elapsed 55.432 s.
  `[SearchEngine volume] iterations=0 elapsed=4.5452250909999998 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792611620.5538852 program-deadline=1792611615.9600847`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 109.449 (no supervisor gap summary); best cost 109.449.

### I147_restore_c009_after_share_experiment
- In progress: 2026-10-21T15:41:29.662482-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I147_restore_c009_after_share_experiment.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I147_restore_c009_after_share_experiment.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I147_restore_c009_after_share_experiment.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test --log-level INFO`
- Completed: 2026-10-21T15:41:30.918988-04:00; exit 0; runner elapsed 1.255 s.

### I148_share_search_egraphs_build
- In progress: 2026-10-21T15:42:52.522225-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I148_share_search_egraphs_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I148_share_search_egraphs_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I148_share_search_egraphs_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test --log-level INFO`
- Completed: 2026-10-21T15:43:13.093543-04:00; exit 1; runner elapsed 20.569 s.

### I149_share_search_egraphs_build2
- In progress: 2026-10-21T15:43:21.055331-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I149_share_search_egraphs_build2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I149_share_search_egraphs_build2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I149_share_search_egraphs_build2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test --log-level INFO`
- Completed: 2026-10-21T15:45:38.250267-04:00; exit 0; runner elapsed 137.193 s.

### I150_share_search_egraphs_n3
- In progress: 2026-10-21T15:47:48.122335-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I150_share_search_egraphs_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I150_share_search_egraphs_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I150_share_search_egraphs_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I150_share_search_egraphs_n3`
- Completed: 2026-10-21T15:48:43.153720-04:00; exit 1; runner elapsed 55.029 s.
  `[SearchEngine volume] iterations=0 elapsed=4.563471872 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792612121.8962917 program-deadline=1792612117.9562194`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 109.449 (no supervisor gap summary); best cost 109.449.

### I151_share_search_egraphs_n2
- In progress: 2026-10-21T15:48:53.518639-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I151_share_search_egraphs_n2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I151_share_search_egraphs_n2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I151_share_search_egraphs_n2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I151_share_search_egraphs_n2`
- Completed: 2026-10-21T15:49:49.935687-04:00; exit 0; runner elapsed 56.413 s.
  `[SearchEngine volume] iterations=0 elapsed=11.064469827 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.613769309654895 normalized=0.82045897698849646 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=17.19043493270874 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792612165.4877212 program-deadline=1792612173.2972984`
- First incumbent: 17.190434933 s, cost 17.5005; best cost 17.500534.

### I152_share_search_egraphs_full_build
- In progress: 2026-10-21T15:51:16.045719-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I152_share_search_egraphs_full_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I152_share_search_egraphs_full_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I152_share_search_egraphs_full_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T15:51:16.615568-04:00; exit 0; runner elapsed 0.568 s.

### I153_share_search_egraphs_n2_full_build
- In progress: 2026-10-21T15:51:18.930595-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I153_share_search_egraphs_n2_full_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I153_share_search_egraphs_n2_full_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I153_share_search_egraphs_n2_full_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I153_share_search_egraphs_n2_full_build`
- Completed: 2026-10-21T15:52:16.006951-04:00; exit 0; runner elapsed 57.073 s.
  `[SearchEngine volume] iterations=1 elapsed=12.095771503 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.5680817794 log2-remaining-fraction=-0.32241289736703038 eliminated-bits=0.32241289736703038 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.505686634228116 normalized=0.81685622114093726 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=16.979290246963501 samples=6`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792612310.5736465 program-deadline=1792612318.5943701`
- First incumbent: 16.979290247 s, cost 17.5005; best cost 17.500534.
- Note: N=2 full build confirms C010 speed point; all five logits graphs exact-checked.

### I154_c011_lazy_past_offset_build
- In progress: 2026-10-21T16:07:33.447125-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I154_c011_lazy_past_offset_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I154_c011_lazy_past_offset_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I154_c011_lazy_past_offset_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T16:09:51.040861-04:00; exit 0; runner elapsed 137.590 s.
- Note: Built C011 prototype with lazy fallback offsets in tryPlan, static queue sizing in SearchEngine, shared reachable cids BFS across bucket aliases, and SEARCH_MARGIN=3.5s.

### I155_c011_n3
- In progress: 2026-10-21T16:09:59.225890-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I155_c011_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I155_c011_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I155_c011_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I155_c011_n3`
- Completed: 2026-10-21T16:11:10.041913-04:00; exit 0; runner elapsed 70.814 s.
  `[SearchEngine volume] iterations=0 elapsed=3.558786698 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.995222313234162 normalized=0.99984074377447207 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.705725908279419 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792613451.7507665 program-deadline=1792613451.045053`
- First incumbent: 27.705725908 s, cost 109.449; best cost 109.448563.
- Note: First N=3 success; all seven logits graphs pass exact verification.

### I156_c011_n3_confirm1
- In progress: 2026-10-21T16:11:44.953972-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I156_c011_n3_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I156_c011_n3_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I156_c011_n3_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I156_c011_n3_confirm1`
- Completed: 2026-10-21T16:12:55.844681-04:00; exit 0; runner elapsed 70.889 s.
  `[SearchEngine volume] iterations=0 elapsed=3.5596556549999998 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.994909848819916 normalized=0.99983032829399721 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.555678606033325 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792613557.4327688 program-deadline=1792613556.8771029`
- First incumbent: 27.555678606 s, cost 109.449; best cost 109.448563.
- Note: Confirmation 1 passes exact logits check across all 7 graphs.

### I157_c011_n3_confirm2
- In progress: 2026-10-21T16:13:07.729916-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I157_c011_n3_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I157_c011_n3_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I157_c011_n3_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I157_c011_n3_confirm2`
- Completed: 2026-10-21T16:14:18.504695-04:00; exit 0; runner elapsed 70.771 s.
  `[SearchEngine volume] iterations=0 elapsed=3.5849476760000001 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.995161795298159 normalized=0.99983872650993866 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.67666482925415 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792613640.2717764 program-deadline=1792613639.5951245`
- First incumbent: 27.676664829 s, cost 109.449; best cost 109.448563.
- Note: Confirmation 2 passes exact logits check across all 7 graphs.

### I158_c011_n2_regression
- In progress: 2026-10-21T16:14:25.042670-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I158_c011_n2_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I158_c011_n2_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I158_c011_n2_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I158_c011_n2_regression`
- Completed: 2026-10-21T16:15:09.785894-04:00; exit 1; runner elapsed 44.740 s.
  `[SearchEngine volume] iterations=0 elapsed=15.184245488 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792613696.1980789 program-deadline=1792613706.708451`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 17.5005 (no supervisor gap summary); best cost 17.500534.
- Note: Root bound refinement ran past 30s supervisor timeout because flexible bound loops lacked inner deadline checks.

### I159_c011_refine_deadline_build
- In progress: 2026-10-21T16:16:49.123727-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I159_c011_refine_deadline_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I159_c011_refine_deadline_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I159_c011_refine_deadline_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test --log-level INFO`
- Completed: 2026-10-21T16:19:06.805592-04:00; exit 0; runner elapsed 137.680 s.
- Note: Added inner deadline checks to RelaxationBoundComputer::flexibleIntervalBound.

### I160_c011_n2_regression
- In progress: 2026-10-21T16:19:13.412405-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I160_c011_n2_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I160_c011_n2_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I160_c011_n2_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 2 --ignore-checkpoints --run-dir logs/gemma_output_runs/I160_c011_n2_regression`
- Completed: 2026-10-21T16:20:13.150179-04:00; exit 0; runner elapsed 59.736 s.
  `[SearchEngine volume] iterations=0 elapsed=14.74220983 total-domain-bits=3110953.8904946768 remaining-domain-bits=3110953.8900098745 log2-remaining-fraction=-0.00048480229452252388 eliminated-bits=0.00048480229452252388 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.216268044482717 normalized=0.80720893481609057 final-gap=0.4810305216972236 final-incumbent=17.500533999999998 final-bound=9.0822430000000001 first-incumbent-elapsed=16.435552835464478 samples=5`
  `[SearchEngine] New best cost: 17.5005 at iteration 0 source=root-heuristic event-epoch=1792613984.3550589 program-deadline=1792613994.4195192`
- First incumbent: 16.435552835 s, cost 17.5005; best cost 17.500534.
- Note: N*=2 regression check passes; first incumbent is 16.435553 s (faster than C010's 17.190 s). All five graphs exact ok=true.

### I161_c011_n3
- In progress: 2026-10-21T16:20:24.342617-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I161_c011_n3.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I161_c011_n3.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I161_c011_n3.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I161_c011_n3`
- Completed: 2026-10-21T16:21:35.616438-04:00; exit 0; runner elapsed 71.272 s.
  `[SearchEngine volume] iterations=0 elapsed=3.7024840710000002 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.995745260073463 normalized=0.99985817533578214 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.95684814453125 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792614077.2825937 program-deadline=1792614075.8257585`
- First incumbent: 27.956848145 s, cost 109.449; best cost 109.448563.
- Note: Primary N*=3 accepted scaling run; all seven logits graphs exact ok=true.

### I162_c011_n3_confirm1
- In progress: 2026-10-21T16:21:58.014540-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I162_c011_n3_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I162_c011_n3_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I162_c011_n3_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I162_c011_n3_confirm1`
- Completed: 2026-10-21T16:23:08.890620-04:00; exit 0; runner elapsed 70.874 s.
  `[SearchEngine volume] iterations=0 elapsed=3.5921844420000002 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.995053976546153 normalized=0.99983513255153844 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.624889612197876 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792614170.3976765 program-deadline=1792614169.2728`
- First incumbent: 27.624889612 s, cost 109.449; best cost 109.448563.
- Note: Confirmation 1 passes; all seven logits graphs exact ok=true.

### I163_c011_n3_confirm2
- In progress: 2026-10-21T16:23:13.870101-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I163_c011_n3_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I163_c011_n3_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I163_c011_n3_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I163_c011_n3_confirm2`
- Completed: 2026-10-21T16:24:24.923435-04:00; exit 0; runner elapsed 71.051 s.
  `[SearchEngine volume] iterations=0 elapsed=3.5643803709999999 total-domain-bits=4359478.7265559016 remaining-domain-bits=4359478.7258782648 log2-remaining-fraction=-0.00067763682454824448 eliminated-bits=0.00067763682454824448 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.994992028712492 normalized=0.99983306762374968 final-gap=0.9979175605987628 final-incumbent=109.44856299999999 final-bound=0.22792000000000001 first-incumbent-elapsed=27.595141887664795 samples=4`
  `[SearchEngine] New best cost: 109.449 at iteration 0 source=root-heuristic event-epoch=1792614246.4417286 program-deadline=1792614245.3465998`
- First incumbent: 27.595141888 s, cost 109.449; best cost 109.448563.
- Note: Confirmation 2 passes; all seven logits graphs exact ok=true.

### I164_c011_n4
- In progress: 2026-10-21T16:26:41.818045-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I164_c011_n4.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I164_c011_n4.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I164_c011_n4.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I164_c011_n4`
- Completed: 2026-10-21T16:30:59.483834-04:00; exit 1; runner elapsed 257.664 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.
- Note: Frontier N=4 budget miss; saturation alone took 28.6616 s across the 9 buckets, killing compile at 30.092 s during reduction.

### I165_c012_in_sweep_propagation_build
- In progress: 2026-10-21T16:56:37.025152-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I165_c012_in_sweep_propagation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I165_c012_in_sweep_propagation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I165_c012_in_sweep_propagation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T16:56:37.607014-04:00; exit 0; runner elapsed 0.580 s.

### I166_c012_n3_regression
- In progress: 2026-10-21T16:56:42.371393-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I166_c012_n3_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I166_c012_n3_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I166_c012_n3_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I166_c012_n3_regression`
- Completed: 2026-10-21T16:57:37.784878-04:00; exit 0; runner elapsed 55.412 s.
  `[SearchEngine volume] iterations=1 elapsed=17.598069936000002 total-domain-bits=4320694.5502817221 remaining-domain-bits=4320694.2276704889 log2-remaining-fraction=-0.32261123321950436 eliminated-bits=0.32261123321950436 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.29511938505966 normalized=0.94317064616865531 final-gap=0.88461542460182996 final-incumbent=109.652542 final-bound=12.652212 first-incumbent-elapsed=13.062474966049194 samples=7`
  `[SearchEngine] New best cost: 109.653 at iteration 0 source=root-heuristic event-epoch=1792616225.5420637 program-deadline=1792616238.4796011`
- First incumbent: 13.062474966 s, cost 109.653; best cost 109.652542.

### I167_c012_n4
- In progress: 2026-10-21T16:57:46.222918-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I167_c012_n4.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I167_c012_n4.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I167_c012_n4.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I167_c012_n4`
- Completed: 2026-10-21T16:58:30.090883-04:00; exit 1; runner elapsed 43.865 s.
  `[SearchEngine volume] iterations=1 elapsed=14.305373026 total-domain-bits=5557624.6811577557 remaining-domain-bits=5557624.3583519179 log2-remaining-fraction=-0.32280583772808313 eliminated-bits=0.32280583772808313 remaining-nodes=2 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 17.3093 at iteration 0 source=root-heuristic event-epoch=1792616297.6621163 program-deadline=1792616305.979811`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 17.3093 (no supervisor gap summary); best cost 17.309278.

### I168_c012_margin_build
- In progress: 2026-10-21T16:59:59.651306-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I168_c012_margin_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I168_c012_margin_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I168_c012_margin_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T17:02:17.452540-04:00; exit 0; runner elapsed 137.799 s.
- Note: C012 build incorporating in-sweep saturation rule expansion (`egraph.getNumENodes()`), 0.5 s search deadline buffer before starting new iterations, and 4.5 s search extraction margin for multi-bucket plan serialization.

### I169_c012_n4
- In progress: 2026-10-21T17:02:22.291128-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I169_c012_n4.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I169_c012_n4.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I169_c012_n4.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I169_c012_n4`
- Completed: 2026-10-21T17:03:23.195783-04:00; exit 0; runner elapsed 60.903 s.
  `[SearchEngine volume] iterations=0 elapsed=12.430993483 total-domain-bits=5557624.6811577557 remaining-domain-bits=5557624.680280013 log2-remaining-fraction=-0.00087774265557527542 eliminated-bits=0.00087774265557527542 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.113962253003457 normalized=0.8037987417667819 final-gap=0.26789338064822799 final-incumbent=17.309277999999999 final-bound=12.672237000000001 first-incumbent-elapsed=17.712357521057129 samples=6`
  `[SearchEngine] New best cost: 17.3093 at iteration 0 source=root-heuristic event-epoch=1792616573.7626104 program-deadline=1792616581.5502658`
- First incumbent: 17.712357521 s, cost 17.3093; best cost 17.309278.
- Note: N*=4 scaling reference evaluation. Saturation 8.63 s; search finds first incumbent at 17.712 s; compile finishes cleanly at 28.542 s. All nine graphs exact-checked against reference logits with `ok=true`.

### I170_c012_n4_confirm1
- In progress: 2026-10-21T17:03:46.703406-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I170_c012_n4_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I170_c012_n4_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I170_c012_n4_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I170_c012_n4_confirm1`
- Completed: 2026-10-21T17:04:47.688219-04:00; exit 0; runner elapsed 60.983 s.
  `[SearchEngine volume] iterations=0 elapsed=12.426335642 total-domain-bits=5557624.6811577557 remaining-domain-bits=5557624.680280013 log2-remaining-fraction=-0.00087774265557527542 eliminated-bits=0.00087774265557527542 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.221535782311495 normalized=0.8073845260770498 final-gap=0.26789338064822799 final-incumbent=17.309277999999999 final-bound=12.672237000000001 first-incumbent-elapsed=17.791255235671997 samples=6`
  `[SearchEngine] New best cost: 17.3093 at iteration 0 source=root-heuristic event-epoch=1792616658.3162212 program-deadline=1792616666.0249782`
- First incumbent: 17.791255236 s, cost 17.3093; best cost 17.309278.
- Note: Confirmation 1 passes at N=4. First incumbent 17.791 s; compile 28.680 s; all nine graphs exact-checked (`ok=true`).

### I171_c012_n4_confirm2
- In progress: 2026-10-21T17:04:54.194973-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I171_c012_n4_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I171_c012_n4_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I171_c012_n4_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I171_c012_n4_confirm2`
- Completed: 2026-10-21T17:05:54.667537-04:00; exit 0; runner elapsed 60.469 s.
  `[SearchEngine volume] iterations=0 elapsed=12.390981547000001 total-domain-bits=5557624.6811577557 remaining-domain-bits=5557624.680280013 log2-remaining-fraction=-0.00087774265557527542 eliminated-bits=0.00087774265557527542 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=24.27792951912453 normalized=0.80926431730415105 final-gap=0.26789338064822799 final-incumbent=17.309277999999999 final-bound=12.672237000000001 first-incumbent-elapsed=17.8238525390625 samples=6`
  `[SearchEngine] New best cost: 17.3093 at iteration 0 source=root-heuristic event-epoch=1792616725.671138 program-deadline=1792616733.3472972`
- First incumbent: 17.823852539 s, cost 17.3093; best cost 17.309278.
- Note: Confirmation 2 passes at N=4. First incumbent 17.824 s; compile 28.272 s; all nine graphs exact-checked (`ok=true`).

### I172_c012_n3_regression
- In progress: 2026-10-21T17:06:01.595330-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I172_c012_n3_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I172_c012_n3_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I172_c012_n3_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 3 --ignore-checkpoints --run-dir logs/gemma_output_runs/I172_c012_n3_regression`
- Completed: 2026-10-21T17:06:55.208282-04:00; exit 0; runner elapsed 53.611 s.
  `[SearchEngine volume] iterations=0 elapsed=15.686885389 total-domain-bits=4320694.5502817221 remaining-domain-bits=4320694.549598584 log2-remaining-fraction=-0.00068313814699649811 eliminated-bits=0.00068313814699649811 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=28.295630457741872 normalized=0.94318768192472902 final-gap=0.88461542460182996 final-incumbent=109.652542 final-bound=12.652212 first-incumbent-elapsed=13.067456722259521 samples=6`
  `[SearchEngine] New best cost: 109.653 at iteration 0 source=root-heuristic event-epoch=1792616784.7288344 program-deadline=1792616797.1613908`
- First incumbent: 13.067456722 s, cost 109.653; best cost 109.652542.
- Note: Prior N*=3 regression check passes with flying colors. First incumbent drops from C011's 27.957 s to 13.067 s (over 14.8 s faster), compile 26.960 s, all seven graphs exact-checked (`ok=true`).


### I173_c012_n5
- In progress: 2026-10-21T17:11:28.539372-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I173_c012_n5.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I173_c012_n5.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I173_c012_n5.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I173_c012_n5`
- Completed: 2026-10-21T17:15:43.840318-04:00; exit 0; runner elapsed 255.299 s.
  `[SearchEngine volume] iterations=0 elapsed=8.5392075819999995 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.977580563539952 normalized=0.99925268545133172 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=22.635956048965454 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792617317.2031698 program-deadline=1792617320.067225`
- First incumbent: 22.635956049 s, cost 99.9016; best cost 99.901581.

### I174_c012_n5_confirm1
- In progress: 2026-10-21T17:15:53.858501-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I174_c012_n5_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I174_c012_n5_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I174_c012_n5_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I174_c012_n5_confirm1`
- Completed: 2026-10-21T17:17:00.725979-04:00; exit 0; runner elapsed 66.865 s.
  `[SearchEngine volume] iterations=0 elapsed=8.6055299909999992 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.977263230234954 normalized=0.9992421076744985 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=22.531722545623779 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792617393.8930526 program-deadline=1792617396.8613431`
- First incumbent: 22.531722546 s, cost 99.9016; best cost 99.901581.

### I175_c012_n5_confirm2
- In progress: 2026-10-21T17:17:07.538396-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I175_c012_n5_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I175_c012_n5_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I175_c012_n5_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I175_c012_n5_confirm2`
- Completed: 2026-10-21T17:17:55.161144-04:00; exit 1; runner elapsed 47.620 s.
  `[SearchEngine volume] iterations=0 elapsed=8.5976203340000001 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792617467.7876837 program-deadline=1792617470.5949118`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 99.9016 (no supervisor gap summary); best cost 99.901581.
- Note: Supervisor killed at 30.037 s (by 37 ms) while serializing 11 graphs; confirms 4.5 s search margin is too tight and buildCompiledGraph buffer lookup needs optimization.

### I176_c013_extraction_build
- In progress: 2026-10-21T17:22:12.766509-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I176_c013_extraction_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I176_c013_extraction_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I176_c013_extraction_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T17:24:31.157640-04:00; exit 0; runner elapsed 138.387 s.
- Note: Build introducing O(1) buffer lookup map in buildCompiledGraph and 5.5 s search margin.

### I177_c013_n5
- In progress: 2026-10-21T17:24:35.582745-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I177_c013_n5.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I177_c013_n5.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I177_c013_n5.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I177_c013_n5`
- Completed: 2026-10-21T18:25:23.768843-04:00; exit 1; runner elapsed 3648.184 s.
  `[SearchEngine volume] iterations=0 elapsed=8.4870958909999992 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.978777965496711 normalized=0.99929259884989041 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=23.029263019561768 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792617916.1395233 program-deadline=1792617917.6102736`
- First incumbent: 23.029263020 s, cost 99.9016; best cost 99.901581.
- Note: Experimental pubsetbuf stack buffer left plan.bin empty because buffer vector was destroyed before ofstream closed; reverted.

### I178_c013_restore_ofstream
- In progress: 2026-10-21T18:25:56.683382-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I178_c013_restore_ofstream.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I178_c013_restore_ofstream.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I178_c013_restore_ofstream.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-21T18:28:14.169373-04:00; exit 0; runner elapsed 137.484 s.
- Note: Clean C013 build restoring standard ofstream and keeping O(1) buffer index map.

### I179_c013_n5
- In progress: 2026-10-21T18:28:18.161774-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I179_c013_n5.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I179_c013_n5.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I179_c013_n5.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I179_c013_n5`
- Completed: 2026-10-21T18:29:24.782288-04:00; exit 0; runner elapsed 66.618 s.
  `[SearchEngine volume] iterations=0 elapsed=8.4359119830000004 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.977504722108929 normalized=0.99925015740363099 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=22.611044645309448 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792621738.3983002 program-deadline=1792621740.2872684`
- First incumbent: 22.611044645 s, cost 99.9016; best cost 99.901581.
- Note: N*=5 scaling reference evaluation. First incumbent 22.611 s, compile finishes cleanly in 29.085 s; all 11 logits graphs exact-checked (ok=true).

### I180_c013_n5_confirm1
- In progress: 2026-10-21T18:29:31.852186-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I180_c013_n5_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I180_c013_n5_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I180_c013_n5_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I180_c013_n5_confirm1`
- Completed: 2026-10-21T18:30:38.420335-04:00; exit 0; runner elapsed 66.566 s.
  `[SearchEngine volume] iterations=0 elapsed=8.4666061470000002 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.97730904967807 normalized=0.99924363498926894 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=22.546772718429565 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792621811.8750029 program-deadline=1792621813.8282428`
- First incumbent: 22.546772718 s, cost 99.9016; best cost 99.901581.
- Note: Confirmation 1 passes at N=5. First incumbent 22.547 s, compile 29.069 s, all 11 graphs exact-checked (ok=true).

### I181_c013_n5_confirm2
- In progress: 2026-10-21T18:30:56.102316-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I181_c013_n5_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I181_c013_n5_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I181_c013_n5_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I181_c013_n5_confirm2`
- Completed: 2026-10-21T18:32:03.205636-04:00; exit 0; runner elapsed 67.100 s.
  `[SearchEngine volume] iterations=0 elapsed=8.5643768439999999 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.977992285986563 normalized=0.99926640953288548 final-gap=0.99695555368638256 final-incumbent=99.901580999999993 final-bound=0.304145 first-incumbent-elapsed=22.771193265914917 samples=4`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792621896.4088633 program-deadline=1792621898.1376834`
- First incumbent: 22.771193266 s, cost 99.9016; best cost 99.901581.
- Note: Confirmation 2 passes at N=5. First incumbent 22.771 s, compile 29.816 s, all 11 graphs exact-checked (ok=true).

### I182_c013_n4_regression
- In progress: 2026-10-21T18:32:12.318160-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I182_c013_n4_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I182_c013_n4_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I182_c013_n4_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I182_c013_n4_regression`
- Completed: 2026-10-21T18:33:12.462844-04:00; exit 0; runner elapsed 60.143 s.
  `[SearchEngine volume] iterations=0 elapsed=11.260400368999999 total-domain-bits=5557624.6811577557 remaining-domain-bits=5557624.680280013 log2-remaining-fraction=-0.00087774265557527542 eliminated-bits=0.00087774265557527542 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=25.166535853546492 normalized=0.8388845284515497 final-gap=0.45286938022487133 final-incumbent=17.309277999999999 final-bound=9.4704359999999994 first-incumbent-elapsed=17.733593702316284 samples=5`
  `[SearchEngine] New best cost: 17.3093 at iteration 0 source=root-heuristic event-epoch=1792621964.0075364 program-deadline=1792621970.7739551`
- First incumbent: 17.733593702 s, cost 17.3093; best cost 17.309278.
- Note: Prior N*=4 regression check passes cleanly. Compile finishes in 27.604 s (0.94 s faster than C012), first incumbent 17.734 s, all nine graphs exact-checked (ok=true).

### I183_c013_n6
- In progress: 2026-10-21T18:38:35.197632-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I183_c013_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I183_c013_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I183_c013_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I183_c013_n6`
- Completed: 2026-10-21T18:43:10.308193-04:00; exit 1; runner elapsed 275.109 s.
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792622587.6988223 program-deadline=1792622584.6681309`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.008881.

### I184_c013_n6_timing
- In progress: 2026-10-22T02:57:09.460473-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I184_c013_n6_timing.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I184_c013_n6_timing.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I184_c013_n6_timing.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I184_c013_n6_timing`
- Completed: 2026-10-22T02:58:02.159689-04:00; exit 1; runner elapsed 52.696 s.
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792652279.9065752 program-deadline=1792652276.5188262`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.008881.

### I185_c014_disjoint_build
- In progress: 2026-10-22T02:59:45.855478-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I185_c014_disjoint_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I185_c014_disjoint_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I185_c014_disjoint_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

I183 reconciled on resume: terminal exit 1, supervisor timeout 30.109 s; saturation 13.855 s, search start 20.398 s. Incumbent logged after search deadline, no valid written plan or exact check. Honest frontier miss.

I184 is a diagnostic with TG_TIME_BUCKETS=1 in the parent environment; bucket timings show full rewrite saturation 1.131 s and shared partial rewrite saturation 12.785 s (1,364,725 classes, 881,114 enodes); path injection is only 0.548 ms and partial postprocessing about 103 ms. Shared partial closure dominates saturation. Terminal exit 1, compile killed at 30.120 s. No incumbent timing/gap summary and no exact pass. GPU detection independently confirmed with nvidia-smi (Quadro M4000); session commands run unrestricted.
- Completed: 2026-10-22T03:02:04.482801-04:00; exit 0; runner elapsed 138.626 s.

### I186_c014_n6
- In progress: 2026-10-22T03:02:09.643184-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I186_c014_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I186_c014_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I186_c014_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I186_c014_n6`
- Completed: 2026-10-22T03:03:01.318467-04:00; exit 1; runner elapsed 51.672 s.
  `[SearchEngine volume] iterations=0 elapsed=7.6700472140000002 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792652578.9127131 program-deadline=1792652575.6964073`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.009.

### I187_c014_partial_selection_build
- In progress: 2026-10-22T03:03:37.800224-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I187_c014_partial_selection_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I187_c014_partial_selection_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I187_c014_partial_selection_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:05:56.284915-04:00; exit 0; runner elapsed 138.482 s.

### I188_c014_n5
- In progress: 2026-10-22T03:06:18.690433-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I188_c014_n5.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I188_c014_n5.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I188_c014_n5.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 5 --ignore-checkpoints --run-dir logs/gemma_output_runs/I188_c014_n5`
- Completed: 2026-10-22T03:07:25.215084-04:00; exit 0; runner elapsed 66.523 s.
  `[SearchEngine volume] iterations=0 elapsed=7.7854263369999996 total-domain-bits=6795954.2585389037 remaining-domain-bits=6795954.2574669123 log2-remaining-fraction=-0.0010719913989305496 eliminated-bits=0.0010719913989305496 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.170769622546132 normalized=0.97235898741820437 final-gap=0.86342873792958286 final-incumbent=99.901580999999993 final-bound=13.643685 first-incumbent-elapsed=22.684114456176758 samples=5`
  `[SearchEngine] New best cost: 99.9016 at iteration 0 source=root-heuristic event-epoch=1792652818.9759724 program-deadline=1792652820.7918713`
- First incumbent: 22.684114456 s, cost 99.9016; best cost 99.901581.

### I189_c014_after_propagation_build
- In progress: 2026-10-22T03:07:50.373474-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I189_c014_after_propagation_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I189_c014_after_propagation_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I189_c014_after_propagation_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`

C014 interim verdict: I185 build passed; I186 N=6 misses at 30.103 s (saturation 13.898 s, search start 20.479 s, first validation 4.182 s, 9429 completions/72 fallbacks). I187 rebuild skips unconfirmed selections in the proposal; I188 N=5 exact-passes all eleven graphs, compile 29.160 s, first incumbent 22.684114 s versus accepted C013 22.611045 s: no speed improvement, not accepted. I189 moves the proposal after initial propagation so shared cache offsets are assigned before disjoint owners; candidate remains unaccepted. No bounds/pruning/saturation alternatives changed.
- Completed: 2026-10-22T03:10:09.773734-04:00; exit 0; runner elapsed 139.398 s.

### I190_c014_after_prop_n6
- In progress: 2026-10-22T03:15:57.838185-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I190_c014_after_prop_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I190_c014_after_prop_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I190_c014_after_prop_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I190_c014_after_prop_n6`
- Completed: 2026-10-22T03:16:49.601927-04:00; exit 1; runner elapsed 51.761 s.
  `[SearchEngine volume] iterations=0 elapsed=7.644179716 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792653404.2370367 program-deadline=1792653403.7110012`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.008881.

### I191_c015_dedup_validate_build
- In progress: 2026-10-22T03:18:06.107155-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I191_c015_dedup_validate_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I191_c015_dedup_validate_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I191_c015_dedup_validate_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python .venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:18:06.123898-04:00; exit 1; runner elapsed 0.014 s.

### I192_c015_n6
- In progress: 2026-10-22T03:18:11.606663-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I192_c015_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I192_c015_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I192_c015_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I192_c015_n6`
- Completed: 2026-10-22T03:19:03.141870-04:00; exit 1; runner elapsed 51.532 s.
  `[SearchEngine volume] iterations=0 elapsed=7.6536426759999996 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792653538.0049396 program-deadline=1792653537.4638841`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.008881.

### I193_c015_dedup_validate_build
- In progress: 2026-10-22T03:19:08.282924-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I193_c015_dedup_validate_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I193_c015_dedup_validate_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I193_c015_dedup_validate_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:21:26.444171-04:00; exit 0; runner elapsed 138.158 s.

### I194_c015_n6
- In progress: 2026-10-22T03:21:34.702912-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I194_c015_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I194_c015_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I194_c015_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I194_c015_n6`
- Completed: 2026-10-22T03:22:26.372170-04:00; exit 1; runner elapsed 51.666 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6932672249999996 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792653741.179966 program-deadline=1792653740.5321057`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.009.

### I195_c015_timeline_build
- In progress: 2026-10-22T03:23:17.173255-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I195_c015_timeline_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I195_c015_timeline_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I195_c015_timeline_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:25:35.103826-04:00; exit 0; runner elapsed 137.929 s.

### I196_c015_n6_timeline
- In progress: 2026-10-22T03:25:40.221427-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I196_c015_n6_timeline.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I196_c015_n6_timeline.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I196_c015_n6_timeline.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I196_c015_n6_timeline`
- Completed: 2026-10-22T03:26:31.934153-04:00; exit 1; runner elapsed 51.710 s.
  `[SearchEngine volume] iterations=0 elapsed=4.7217651810000003 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792653986.8578787 program-deadline=1792653986.0840323`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 100.009 (no supervisor gap summary); best cost 100.009.

### I197_c015_single_persist_build
- In progress: 2026-10-22T03:27:28.919343-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I197_c015_single_persist_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I197_c015_single_persist_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I197_c015_single_persist_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:29:47.325034-04:00; exit 0; runner elapsed 138.404 s.

### I198_c015_n6
- In progress: 2026-10-22T03:29:47.402091-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I198_c015_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I198_c015_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I198_c015_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I198_c015_n6`
- Completed: 2026-10-22T03:31:00.465976-04:00; exit 0; runner elapsed 73.062 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6505392399999996 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.981388445233293 normalized=0.99937961484110971 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=25.32092809677124 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792654234.1045344 program-deadline=1792654233.2836194`
- First incumbent: 25.320928097 s, cost 100.009; best cost 100.008881.

### I199_constant_size_diag_build
- In progress: 2026-10-22T03:32:09.159448-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I199_constant_size_diag_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I199_constant_size_diag_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I199_constant_size_diag_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:34:27.151313-04:00; exit 0; runner elapsed 137.989 s.

### I200_constant_size_diag_n6
- In progress: 2026-10-22T03:34:27.221489-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I200_constant_size_diag_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I200_constant_size_diag_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I200_constant_size_diag_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I200_constant_size_diag_n6`
- Completed: 2026-10-22T03:35:40.161114-04:00; exit 0; runner elapsed 72.938 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6817276300000001 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.980826200448035 normalized=0.99936087334826784 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=25.17957592010498 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792654513.6218004 program-deadline=1792654512.9422371`
- First incumbent: 25.179575920 s, cost 100.009; best cost 100.008881.

### I201_c015_compact_views_build
- In progress: 2026-10-22T03:37:04.549199-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I201_c015_compact_views_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I201_c015_compact_views_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I201_c015_compact_views_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:39:23.266518-04:00; exit 0; runner elapsed 138.716 s.

### I202_c015_compact_views_n6
- In progress: 2026-10-22T03:39:23.345061-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I202_c015_compact_views_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I202_c015_compact_views_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I202_c015_compact_views_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I202_c015_compact_views_n6`
- Completed: 2026-10-22T03:40:32.174338-04:00; exit 0; runner elapsed 68.828 s.
  `[SearchEngine volume] iterations=0 elapsed=4.644197417 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.980715463999267 normalized=0.99935718213330893 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=25.15173602104187 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792654809.7328286 program-deadline=1792654809.0811043`
- First incumbent: 25.151736021 s, cost 100.009; best cost 100.008881.

### I203_c015_background_free_build
- In progress: 2026-10-22T03:41:34.276546-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I203_c015_background_free_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I203_c015_background_free_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I203_c015_background_free_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:43:52.282679-04:00; exit 0; runner elapsed 138.004 s.

### I204_c015_n7
- In progress: 2026-10-22T03:43:58.293015-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I204_c015_n7.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I204_c015_n7.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I204_c015_n7.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 7 --ignore-checkpoints --run-dir logs/gemma_output_runs/I204_c015_n7`
- Completed: 2026-10-22T03:47:09.282526-04:00; exit 1; runner elapsed 190.987 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4429305340000003 total-domain-bits=9271780.4079822637 remaining-domain-bits=9271780.4065214191 log2-remaining-fraction=-0.0014608446508646011 eliminated-bits=0.0014608446508646011 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 24.6443 at iteration 0 source=root-heuristic event-epoch=1792655228.9126906 program-deadline=1792655223.6169996`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 24.6443 (no supervisor gap summary); best cost 24.6443.

### I205_c015_n6
- In progress: 2026-10-22T03:47:17.559379-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I205_c015_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I205_c015_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I205_c015_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I205_c015_n6`
- Completed: 2026-10-22T03:48:26.164315-04:00; exit 0; runner elapsed 68.603 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6772300109999998 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.981387358438248 normalized=0.99937957861460824 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=25.32065486907959 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792655284.390264 program-deadline=1792655283.5696213`
- First incumbent: 25.320654869 s, cost 100.009; best cost 100.008881.

### I206_c015_oracle
- In progress: 2026-10-22T03:48:33.962177-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I206_c015_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I206_c015_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I206_c015_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T03:48:38.530472-04:00; exit 0; runner elapsed 4.566 s.
- Oracle verdict: 48/48 cases passed.

### I207_c015_search_gap
- In progress: 2026-10-22T03:48:38.599142-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I207_c015_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I207_c015_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I207_c015_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T03:48:38.628633-04:00; exit 0; runner elapsed 0.028 s.

### I208_c015_native__private_layout
- In progress: 2026-10-22T03:48:38.694151-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I208_c015_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I208_c015_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I208_c015_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T03:48:38.711363-04:00; exit 1; runner elapsed 0.016 s.

### I209_c015_native__write_after_read_cache
- In progress: 2026-10-22T03:48:38.780450-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I209_c015_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I209_c015_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I209_c015_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T03:48:38.799034-04:00; exit 1; runner elapsed 0.017 s.

### I210_c015_native__incremental_propagators
- In progress: 2026-10-22T03:48:38.866459-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I210_c015_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I210_c015_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I210_c015_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T03:48:38.883619-04:00; exit 1; runner elapsed 0.015 s.

### I211_c015_native__start_rank_window
- In progress: 2026-10-22T03:48:38.953907-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I211_c015_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I211_c015_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I211_c015_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T03:48:38.971721-04:00; exit 1; runner elapsed 0.016 s.

### I212_c015_native__selection_reachability
- In progress: 2026-10-22T03:48:39.038508-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I212_c015_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I212_c015_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I212_c015_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T03:48:39.055962-04:00; exit 1; runner elapsed 0.016 s.

### I213_c015_native__relaxation_bound
- In progress: 2026-10-22T03:48:39.122723-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I213_c015_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I213_c015_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I213_c015_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T03:48:39.140111-04:00; exit 1; runner elapsed 0.016 s.

### I214_c015_native__private_layout
- In progress: 2026-10-22T03:48:56.406641-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I214_c015_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I214_c015_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I214_c015_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T03:48:56.627738-04:00; exit 0; runner elapsed 0.219 s.

### I215_c015_native__write_after_read_cache
- In progress: 2026-10-22T03:48:56.703198-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I215_c015_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I215_c015_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I215_c015_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T03:48:56.924983-04:00; exit 0; runner elapsed 0.220 s.

### I216_c015_native__incremental_propagators
- In progress: 2026-10-22T03:48:57.002163-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I216_c015_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I216_c015_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I216_c015_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T03:48:57.217133-04:00; exit 0; runner elapsed 0.213 s.

### I217_c015_native__start_rank_window
- In progress: 2026-10-22T03:48:57.293107-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I217_c015_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I217_c015_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I217_c015_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T03:48:57.516357-04:00; exit 0; runner elapsed 0.221 s.

### I218_c015_native__selection_reachability
- In progress: 2026-10-22T03:48:57.591609-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I218_c015_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I218_c015_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I218_c015_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T03:48:57.839949-04:00; exit 0; runner elapsed 0.245 s.

### I219_c015_native__relaxation_bound
- In progress: 2026-10-22T03:48:57.937237-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I219_c015_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I219_c015_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I219_c015_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T03:50:48.581762-04:00; exit 0; runner elapsed 110.643 s.

C014/C015 notes: I190 evaluated the C014 build at N=6: the disjoint-offset proposal validates (first tryPlan 1.8 s, no completions/fallbacks), but the same plan was then re-validated twice (1.4 s each, rejected by the cost bound) after the deadline, so search ended at 27.8 s planAll time. I191/I192 were a mistaken build invocation (interpreter passed twice) and a run on the stale binary; ignore. I193/I194 (C015 dedup + deadline stop): search done at 24.98 s but compile still killed. I195/I196 timeline diagnostics showed post-search: graph build 0.95 s, planAll teardown 0.77 s, persist 1.6 s, then persist again plus arenas. I197/I198 removed the duplicate persist: first N=6 pass (29.605 s). I199/I200 showed plan constants are only 17 KB; the 199 MB plan was per-graph `nodeViews` for all ~267k canonical classes. I201/I202 keep only referenced views: compile 26.939 s, plan 6 MB. I203 frees planner state on a detached thread. I204 N=7 miss (incumbent ~29.8 s). I205 is the accepted N=6 reference; I206-I219 gates (I208-I213 were mis-invoked without `--native-command` and are superseded by I214-I219, all passing).

### I220_reduce_timing_build
- In progress: 2026-10-22T03:52:14.545966-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I220_reduce_timing_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I220_reduce_timing_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I220_reduce_timing_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T03:54:32.483412-04:00; exit 0; runner elapsed 137.936 s.

### I221_reduce_timing_n6
- In progress: 2026-10-22T03:54:32.561742-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I221_reduce_timing_n6.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I221_reduce_timing_n6.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I221_reduce_timing_n6.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I221_reduce_timing_n6`
- Completed: 2026-10-22T03:55:41.161969-04:00; exit 0; runner elapsed 68.598 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6499122020000003 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.981602007097468 normalized=0.99938673356991559 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=25.374619007110596 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792655719.2243114 program-deadline=1792655718.3497064`
- First incumbent: 25.374619007 s, cost 100.009; best cost 100.008881.

### I222_c016_move_compaction_build
- In progress: 2026-10-22T04:02:25.343754-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I222_c016_move_compaction_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I222_c016_move_compaction_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I222_c016_move_compaction_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:08:27.953672-04:00; exit 0; runner elapsed 362.608 s.

### I223_rebuild_timing_build
- In progress: 2026-10-22T04:09:13.642129-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I223_rebuild_timing_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I223_rebuild_timing_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I223_rebuild_timing_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:15:15.940302-04:00; exit 0; runner elapsed 362.296 s.

### I224_c016_skip_hashcons_build
- In progress: 2026-10-22T04:16:27.367316-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I224_c016_skip_hashcons_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I224_c016_skip_hashcons_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I224_c016_skip_hashcons_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:22:31.314332-04:00; exit 0; runner elapsed 363.944 s.

### I225_c016_background_compaction_free_build
- In progress: 2026-10-22T04:23:20.754257-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I225_c016_background_compaction_free_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I225_c016_background_compaction_free_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I225_c016_background_compaction_free_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:29:24.328522-04:00; exit 0; runner elapsed 363.572 s.

### I226_c016_addop_probe_build
- In progress: 2026-10-22T04:34:00.222513-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I226_c016_addop_probe_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I226_c016_addop_probe_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I226_c016_addop_probe_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:40:04.166544-04:00; exit 0; runner elapsed 363.942 s.

### I227_c016_calloc_arena_build
- In progress: 2026-10-22T04:42:03.859230-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I227_c016_calloc_arena_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I227_c016_calloc_arena_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I227_c016_calloc_arena_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:48:07.240947-04:00; exit 0; runner elapsed 363.380 s.

### I228_c016_n7
- In progress: 2026-10-22T04:48:44.798828-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I228_c016_n7.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I228_c016_n7.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I228_c016_n7.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 7 --ignore-checkpoints --run-dir logs/gemma_output_runs/I228_c016_n7`
- Completed: 2026-10-22T04:50:04.529643-04:00; exit 0; runner elapsed 79.729 s.
  `[SearchEngine volume] iterations=0 elapsed=5.5498087719999996 total-domain-bits=9271780.4079822637 remaining-domain-bits=9271780.4065214191 log2-remaining-fraction=-0.0014608446508646011 eliminated-bits=0.0014608446508646011 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.983589564112474 normalized=0.99945298547041583 final-gap=0.983123469244368 final-incumbent=24.644283000000001 final-bound=0.41591 first-incumbent-elapsed=29.027617931365967 samples=4`
  `[SearchEngine] New best cost: 24.6443 at iteration 0 source=root-heuristic event-epoch=1792658978.191783 program-deadline=1792658973.6641767`
- First incumbent: 29.027617931 s, cost 24.6443; best cost 24.644283.

### I229_c016_kernel_memo_build
- In progress: 2026-10-22T04:51:07.409559-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I229_c016_kernel_memo_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I229_c016_kernel_memo_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I229_c016_kernel_memo_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:53:25.317759-04:00; exit 0; runner elapsed 137.906 s.

### I230_c016_parallel_propagation_state_build
- In progress: 2026-10-22T04:55:03.209247-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I230_c016_parallel_propagation_state_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I230_c016_parallel_propagation_state_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I230_c016_parallel_propagation_state_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T04:57:21.458953-04:00; exit 0; runner elapsed 138.247 s.

### I231_c016_build
- In progress: 2026-10-22T04:58:02.171060-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I231_c016_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I231_c016_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I231_c016_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:04:06.427801-04:00; exit 0; runner elapsed 364.255 s.

### I232_c016_n7
- In progress: 2026-10-22T05:04:06.515997-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I232_c016_n7.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I232_c016_n7.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I232_c016_n7.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 7 --ignore-checkpoints --run-dir logs/gemma_output_runs/I232_c016_n7`
- Completed: 2026-10-22T05:05:22.433948-04:00; exit 0; runner elapsed 75.916 s.
  `[SearchEngine volume] iterations=0 elapsed=4.6086670239999998 total-domain-bits=9271780.4079822637 remaining-domain-bits=9271780.4065214191 log2-remaining-fraction=-0.0014608446508646011 eliminated-bits=0.0014608446508646011 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.96586971460728 normalized=0.99886232382024265 final-gap=0.983123469244368 final-incumbent=24.644283000000001 final-bound=0.41591 first-incumbent-elapsed=27.977648019790649 samples=4`
  `[SearchEngine] New best cost: 24.6443 at iteration 0 source=root-heuristic event-epoch=1792659898.6300528 program-deadline=1792659895.1524179`
- First incumbent: 27.977648020 s, cost 24.6443; best cost 24.644283.

### I233_c016_n6_regression
- In progress: 2026-10-22T05:05:32.055556-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I233_c016_n6_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I233_c016_n6_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I233_c016_n6_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I233_c016_n6_regression`
- Completed: 2026-10-22T05:06:38.197451-04:00; exit 0; runner elapsed 66.139 s.
  `[SearchEngine volume] iterations=0 elapsed=4.3758338840000004 total-domain-bits=8033584.0807822179 remaining-domain-bits=8033584.0795157999 log2-remaining-fraction=-0.0012664180248975754 eliminated-bits=0.0012664180248975754 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.974822827139832 normalized=0.9991607609046611 final-gap=0.99602238325214343 final-incumbent=100.008881 final-bound=0.39779700000000001 first-incumbent-elapsed=23.670286893844604 samples=4`
  `[SearchEngine] New best cost: 100.009 at iteration 0 source=root-heuristic event-epoch=1792659976.3311639 program-deadline=1792659977.1608899`
- First incumbent: 23.670286894 s, cost 100.009; best cost 100.008881.

### I234_c016_oracle
- In progress: 2026-10-22T05:06:38.270185-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I234_c016_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I234_c016_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I234_c016_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T05:06:42.749693-04:00; exit 0; runner elapsed 4.478 s.
- Oracle verdict: 48/48 cases passed.

### I235_c016_search_gap
- In progress: 2026-10-22T05:06:42.810727-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I235_c016_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I235_c016_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I235_c016_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T05:06:42.834580-04:00; exit 0; runner elapsed 0.022 s.

### I236_c016_native__private_layout
- In progress: 2026-10-22T05:06:42.901496-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I236_c016_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I236_c016_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I236_c016_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T05:06:43.118467-04:00; exit 0; runner elapsed 0.215 s.

### I237_c016_native__write_after_read_cache
- In progress: 2026-10-22T05:06:43.195365-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I237_c016_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I237_c016_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I237_c016_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T05:06:43.388533-04:00; exit 0; runner elapsed 0.190 s.

### I238_c016_native__incremental_propagators
- In progress: 2026-10-22T05:06:43.462666-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I238_c016_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I238_c016_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I238_c016_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T05:06:43.681019-04:00; exit 0; runner elapsed 0.216 s.

### I239_c016_native__start_rank_window
- In progress: 2026-10-22T05:06:43.762460-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I239_c016_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I239_c016_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I239_c016_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T05:06:43.957482-04:00; exit 0; runner elapsed 0.193 s.

### I240_c016_native__selection_reachability
- In progress: 2026-10-22T05:06:44.035210-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I240_c016_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I240_c016_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I240_c016_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T05:06:44.290685-04:00; exit 0; runner elapsed 0.254 s.

### I241_c016_native__relaxation_bound
- In progress: 2026-10-22T05:06:44.391830-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I241_c016_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I241_c016_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I241_c016_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T05:08:36.539962-04:00; exit 0; runner elapsed 112.145 s.

C016 notes: I220/I221 added reduction sub-step timing (shared partial reduce: rebuild 2.38 s, infos+prune 0.9 s, cycle passes 0.67 s, final rebuild+infos 1.53 s). Profiling (`perf record --call-graph dwarf` on the compile phase) showed ~30% of samples in malloc/free. I222-I225 moved compaction, skipped the final hashcons, freed replaced containers in the background (reduction 20.27 s -> 19.39 s at N=6). I226 addOp probe (small, ~0.15 s). I227 calloc arena. I228 first N=7 pass (29.889 s). I229 kernel memo (isOpCacheOrScatter rebuilt fused reference graphs per call inside tryPlan), I230 parallel propagation state, I231 clean build (timing logs removed), I232 accepted N=7 reference. Diagnostics outside the evaluator used a 30 s kill.

### I242_c017_completion_bitmask_build
- In progress: 2026-10-22T05:09:28.918125-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I242_c017_completion_bitmask_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I242_c017_completion_bitmask_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I242_c017_completion_bitmask_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:11:47.165281-04:00; exit 0; runner elapsed 138.245 s.

### I243_c017_parallel_groups_build
- In progress: 2026-10-22T05:16:48.815348-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I243_c017_parallel_groups_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I243_c017_parallel_groups_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I243_c017_parallel_groups_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:19:07.057745-04:00; exit 0; runner elapsed 138.241 s.

### I244_c017_single_group_worker_build
- In progress: 2026-10-22T05:22:41.185054-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I244_c017_single_group_worker_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I244_c017_single_group_worker_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I244_c017_single_group_worker_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:25:00.121589-04:00; exit 0; runner elapsed 138.935 s.

### I245_c017_n8
- In progress: 2026-10-22T05:25:36.841339-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I245_c017_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I245_c017_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I245_c017_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I245_c017_n8`
- Completed: 2026-10-22T05:29:16.875834-04:00; exit 0; runner elapsed 220.033 s.
  `[SearchEngine volume] iterations=0 elapsed=5.2117764749999997 total-domain-bits=10508970.663028382 remaining-domain-bits=10508970.661373109 log2-remaining-fraction=-0.0016552731394767761 eliminated-bits=0.0016552731394767761 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.982992189186056 normalized=0.99943307297286854 final-gap=0.99562409150991171 final-incumbent=115.861198 final-bound=0.50699799999999995 first-incumbent-elapsed=26.113307476043701 samples=4`
  `[SearchEngine] New best cost: 115.861 at iteration 0 source=root-heuristic event-epoch=1792661332.3120501 program-deadline=1792661330.6987534`
- First incumbent: 26.113307476 s, cost 115.861; best cost 115.861198.

### I246_c017_n7_regression
- In progress: 2026-10-22T05:29:26.057817-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I246_c017_n7_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I246_c017_n7_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I246_c017_n7_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 7 --ignore-checkpoints --run-dir logs/gemma_output_runs/I246_c017_n7_regression`
- Completed: 2026-10-22T05:30:34.154695-04:00; exit 0; runner elapsed 68.095 s.
  `[SearchEngine volume] iterations=0 elapsed=6.2070057030000001 total-domain-bits=9271780.4079822637 remaining-domain-bits=9271780.4065214191 log2-remaining-fraction=-0.0014608446508646011 eliminated-bits=0.0014608446508646011 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=26.515810218991373 normalized=0.88386034063304575 final-gap=0.42486803125901457 final-incumbent=24.644283000000001 final-bound=14.173715 first-incumbent-elapsed=23.209906339645386 samples=5`
  `[SearchEngine] New best cost: 24.6443 at iteration 0 source=root-heuristic event-epoch=1792661409.1655538 program-deadline=1792661410.4556589`
- First incumbent: 23.209906340 s, cost 24.6443; best cost 24.644283.

### I247_c017_oracle
- In progress: 2026-10-22T05:30:34.213875-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I247_c017_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I247_c017_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I247_c017_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T05:30:38.564434-04:00; exit 0; runner elapsed 4.349 s.
- Oracle verdict: 48/48 cases passed.

### I248_c017_search_gap
- In progress: 2026-10-22T05:30:38.628280-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I248_c017_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I248_c017_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I248_c017_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T05:30:38.649836-04:00; exit 0; runner elapsed 0.020 s.

### I249_c017_native__private_layout
- In progress: 2026-10-22T05:30:38.716653-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I249_c017_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I249_c017_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I249_c017_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T05:30:38.916133-04:00; exit 0; runner elapsed 0.198 s.

### I250_c017_native__write_after_read_cache
- In progress: 2026-10-22T05:30:38.994262-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I250_c017_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I250_c017_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I250_c017_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T05:30:39.192099-04:00; exit 0; runner elapsed 0.196 s.

### I251_c017_native__incremental_propagators
- In progress: 2026-10-22T05:30:39.269927-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I251_c017_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I251_c017_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I251_c017_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T05:30:39.477123-04:00; exit 0; runner elapsed 0.206 s.

### I252_c017_native__start_rank_window
- In progress: 2026-10-22T05:30:39.557967-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I252_c017_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I252_c017_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I252_c017_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T05:30:39.766034-04:00; exit 0; runner elapsed 0.206 s.

### I253_c017_native__selection_reachability
- In progress: 2026-10-22T05:30:39.845033-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I253_c017_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I253_c017_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I253_c017_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T05:30:40.092707-04:00; exit 0; runner elapsed 0.246 s.

### I254_c017_native__relaxation_bound
- In progress: 2026-10-22T05:30:40.200975-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I254_c017_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I254_c017_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I254_c017_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T05:32:31.947795-04:00; exit 0; runner elapsed 111.745 s.

C017 notes: I242 completion bitmasks (N=7 diagnostic compile 25.6 s, match counts identical to I231). I243 parallel partial groups (default min(partial, threads)); diagnostics compared G=14/7/4/3/2/1. I244 default G=1 on a worker thread. I245 N=8 pass. I246-I254 regression and gates.

### I255_c018_addop_memo_build
- In progress: 2026-10-22T05:34:48.867665-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I255_c018_addop_memo_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I255_c018_addop_memo_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I255_c018_addop_memo_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:40:55.690139-04:00; exit 0; runner elapsed 366.821 s.

### I256_round_robin_diag_build
- In progress: 2026-10-22T05:41:52.967001-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I256_round_robin_diag_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I256_round_robin_diag_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I256_round_robin_diag_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:45:50.184135-04:00; exit 0; runner elapsed 237.215 s.

### I257_c018_parallel_infos_build
- In progress: 2026-10-22T05:47:49.996207-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I257_c018_parallel_infos_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I257_c018_parallel_infos_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I257_c018_parallel_infos_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:50:07.782829-04:00; exit 0; runner elapsed 137.784 s.

### I258_c018_parallel_infos_serial_first_build
- In progress: 2026-10-22T05:50:47.938960-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I258_c018_parallel_infos_serial_first_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I258_c018_parallel_infos_serial_first_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I258_c018_parallel_infos_serial_first_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:53:05.935358-04:00; exit 0; runner elapsed 137.993 s.

### I259_c018_parallel_reconcile_build
- In progress: 2026-10-22T05:57:03.225754-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I259_c018_parallel_reconcile_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I259_c018_parallel_reconcile_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I259_c018_parallel_reconcile_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T05:59:21.806884-04:00; exit 0; runner elapsed 138.579 s.

### I260_c018_n8
- In progress: 2026-10-22T06:02:46.598454-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I260_c018_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I260_c018_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I260_c018_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I260_c018_n8`
- Completed: 2026-10-22T06:03:59.216881-04:00; exit 0; runner elapsed 72.616 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1889168989999996 total-domain-bits=10508970.663028382 remaining-domain-bits=10508970.661373109 log2-remaining-fraction=-0.0016552731394767761 eliminated-bits=0.0016552731394767761 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.982916437413262 normalized=0.99943054791377539 final-gap=0.99562409150991171 final-incumbent=115.861198 final-bound=0.50699799999999995 first-incumbent-elapsed=26.095996379852295 samples=4`
  `[SearchEngine] New best cost: 115.861 at iteration 0 source=root-heuristic event-epoch=1792663414.7345235 program-deadline=1792663413.1385393`
- First incumbent: 26.095996380 s, cost 115.861; best cost 115.861198.

C018 notes: I255 per-graph addOpToEGraph memo (cleared on merge/rebuild, identical counts) gave no speedup and was reverted. I256 round-robin grouping diagnostic (crash on alias owner, then fixed). I257 parallel enode infos + reduce-all-in-pool (union reduce slower; pool change reverted in I258). I258/I259 parallel infos with serial first representative, parallel reconcile. I260 N=8 pass, not a meaningful speed change.

### I261_c018_parallel_indexes_build
- In progress: 2026-10-22T06:05:30.410124-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I261_c018_parallel_indexes_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I261_c018_parallel_indexes_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I261_c018_parallel_indexes_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T06:07:48.419160-04:00; exit 0; runner elapsed 138.007 s.

### I262_c018_pooled_group_reduce_build
- In progress: 2026-10-22T06:09:02.972993-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I262_c018_pooled_group_reduce_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I262_c018_pooled_group_reduce_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I262_c018_pooled_group_reduce_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T06:11:21.035836-04:00; exit 0; runner elapsed 138.061 s.

### I263_c018_g9_n9_diag
- In progress: 2026-10-22T06:59:18.489677-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I263_c018_g9_n9_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I263_c018_g9_n9_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I263_c018_g9_n9_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I263_c018_g9_n9_diag`
- Completed: 2026-10-22T07:02:39.602437-04:00; exit 1; runner elapsed 201.111 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I264_c019_parallel_root_setup_build
- In progress: 2026-10-22T07:07:33.889829-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I264_c019_parallel_root_setup_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I264_c019_parallel_root_setup_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I264_c019_parallel_root_setup_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python .venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T07:07:33.908087-04:00; exit 1; runner elapsed 0.016 s.

### I265_c019_parallel_root_setup_build
- In progress: 2026-10-22T07:07:38.097943-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I265_c019_parallel_root_setup_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I265_c019_parallel_root_setup_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I265_c019_parallel_root_setup_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T07:07:58.576189-04:00; exit 1; runner elapsed 20.476 s.

### I266_c019_parallel_root_setup_build
- In progress: 2026-10-22T07:08:04.709671-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I266_c019_parallel_root_setup_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I266_c019_parallel_root_setup_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I266_c019_parallel_root_setup_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T07:10:23.325886-04:00; exit 0; runner elapsed 138.615 s.

### I267_c019_oracle
- In progress: 2026-10-22T07:10:26.621697-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I267_c019_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I267_c019_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I267_c019_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T07:10:31.093824-04:00; exit 0; runner elapsed 4.470 s.
- Oracle verdict: 48/48 cases passed.

### I268_c019_g9_n9_diag
- In progress: 2026-10-22T07:10:38.346869-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I268_c019_g9_n9_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I268_c019_g9_n9_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I268_c019_g9_n9_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I268_c019_g9_n9_diag`
- Completed: 2026-10-22T07:11:23.454068-04:00; exit 1; runner elapsed 45.104 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I269_c019_event_init_build
- In progress: 2026-10-22T07:13:19.579418-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I269_c019_event_init_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I269_c019_event_init_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I269_c019_event_init_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T07:15:38.298002-04:00; exit 0; runner elapsed 138.715 s.

### I270_c019_g9_n9_diag
- In progress: 2026-10-22T07:15:42.845753-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I270_c019_g9_n9_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I270_c019_g9_n9_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I270_c019_g9_n9_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I270_c019_g9_n9_diag`
- Completed: 2026-10-22T07:16:29.349712-04:00; exit 1; runner elapsed 46.502 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I271_c019_g4_n4_diag
- In progress: 2026-10-22T07:16:58.271589-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I271_c019_g4_n4_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I271_c019_g4_n4_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I271_c019_g4_n4_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 4 --ignore-checkpoints --run-dir logs/gemma_output_runs/I271_c019_g4_n4_diag`
- Completed: 2026-10-22T07:18:05.492641-04:00; exit 0; runner elapsed 67.219 s.
  `[SearchEngine volume] iterations=0 elapsed=16.200588810999999 total-domain-bits=11768446.335911898 remaining-domain-bits=11763972.110497605 log2-remaining-fraction=-4474.2254142928869 eliminated-bits=4474.2254142928869 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=27.921970969944034 normalized=0.93073236566480111 final-gap=0.79598318911777721 final-incumbent=17.21266 final-bound=3.5116719999999999 first-incumbent-elapsed=14.500185489654541 samples=5`
  `[SearchEngine] New best cost: 17.2127 at iteration 0 source=root-heuristic event-epoch=1792667854.7345772 program-deadline=1792667864.7344029`
- First incumbent: 14.500185490 s, cost 17.2127; best cost 17.21266.

### I272_c019_g6_n6_diag
- In progress: 2026-10-22T07:18:26.633505-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I272_c019_g6_n6_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I272_c019_g6_n6_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I272_c019_g6_n6_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 6 --ignore-checkpoints --run-dir logs/gemma_output_runs/I272_c019_g6_n6_diag`
- Completed: 2026-10-22T07:19:44.515965-04:00; exit 0; runner elapsed 77.880 s.
  `[SearchEngine volume] iterations=0 elapsed=15.908254715 total-domain-bits=18415831.315205097 remaining-domain-bits=18402737.772223633 log2-remaining-fraction=-13093.542981464416 eliminated-bits=13093.542981464416 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.955138701506662 normalized=0.99850462338355539 final-gap=0.99429381453040455 final-incumbent=217.636459 final-bound=1.2418739999999999 first-incumbent-elapsed=19.443185567855835 samples=5`
  `[SearchEngine] New best cost: 217.636 at iteration 0 source=root-heuristic event-epoch=1792667953.3263953 program-deadline=1792667958.3832219`
- First incumbent: 19.443185568 s, cost 217.636; best cost 217.636459.

### I273_c019_g8_n8_diag
- In progress: 2026-10-22T07:19:52.725102-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I273_c019_g8_n8_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I273_c019_g8_n8_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I273_c019_g8_n8_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I273_c019_g8_n8_diag`
- Completed: 2026-10-22T07:20:50.888638-04:00; exit 1; runner elapsed 58.161 s.
  `[SearchEngine volume] iterations=0 elapsed=8.2932671389999992 total-domain-bits=25040045.428493973 remaining-domain-bits=25002191.500899293 log2-remaining-fraction=-37853.927594680339 eliminated-bits=37853.927594680339 remaining-nodes=1 exhausted=false incomplete-branches=0`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I274_c019_g3_n9_diag
- In progress: 2026-10-22T07:21:28.115447-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I274_c019_g3_n9_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I274_c019_g3_n9_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I274_c019_g3_n9_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I274_c019_g3_n9_diag`
- Completed: 2026-10-22T07:22:41.778278-04:00; exit 1; runner elapsed 73.661 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I275_relaxdiag_build
- In progress: 2026-10-22T07:23:03.493953-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I275_relaxdiag_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I275_relaxdiag_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I275_relaxdiag_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T07:25:22.390442-04:00; exit 0; runner elapsed 138.895 s.

### I276_relaxdiag_g8_n8
- In progress: 2026-10-22T07:25:22.473158-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I276_relaxdiag_g8_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I276_relaxdiag_g8_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I276_relaxdiag_g8_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I276_relaxdiag_g8_n8`
- Completed: 2026-10-22T07:26:04.010224-04:00; exit 1; runner elapsed 41.535 s.
  `[SearchEngine volume] iterations=0 elapsed=8.1008781180000007 total-domain-bits=25040045.428493973 remaining-domain-bits=25002191.500899293 log2-remaining-fraction=-37853.927594680339 eliminated-bits=37853.927594680339 remaining-nodes=1 exhausted=false incomplete-branches=0`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I277_relaxdiag_g8_n8
- In progress: 2026-10-22T07:28:36.446016-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I277_relaxdiag_g8_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I277_relaxdiag_g8_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I277_relaxdiag_g8_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I277_relaxdiag_g8_n8`
- Completed: 2026-10-22T07:29:17.857414-04:00; exit 1; runner elapsed 41.410 s.
  `[SearchEngine volume] iterations=0 elapsed=8.3343262809999992 total-domain-bits=25040045.428493973 remaining-domain-bits=25002191.500899293 log2-remaining-fraction=-37853.927594680339 eliminated-bits=37853.927594680339 remaining-nodes=1 exhausted=false incomplete-branches=0`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I278_relaxdiag_g8_n8
- In progress: 2026-10-22T07:32:01.236612-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I278_relaxdiag_g8_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I278_relaxdiag_g8_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I278_relaxdiag_g8_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I278_relaxdiag_g8_n8`
- Completed: 2026-10-22T07:32:42.565883-04:00; exit 1; runner elapsed 41.328 s.
  `[SearchEngine volume] iterations=0 elapsed=8.308078751 total-domain-bits=25040045.428493973 remaining-domain-bits=25002191.500899293 log2-remaining-fraction=-37853.927594680339 eliminated-bits=37853.927594680339 remaining-nodes=1 exhausted=false incomplete-branches=0`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I279_emptydiag_g8_n8
- In progress: 2026-10-22T07:36:08.415743-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I279_emptydiag_g8_n8.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I279_emptydiag_g8_n8.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I279_emptydiag_g8_n8.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I279_emptydiag_g8_n8`
- Completed: 2026-10-22T07:36:50.105692-04:00; exit 1; runner elapsed 41.688 s.
  `[SearchEngine volume] iterations=0 elapsed=8.3850542109999999 total-domain-bits=25040045.428493973 remaining-domain-bits=25002191.500899293 log2-remaining-fraction=-37853.927594680339 eliminated-bits=37853.927594680339 remaining-nodes=1 exhausted=false incomplete-branches=0`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I280_standin_g8_n8_diag
- In progress: 2026-10-22T07:44:19.749380-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I280_standin_g8_n8_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I280_standin_g8_n8_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I280_standin_g8_n8_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I280_standin_g8_n8_diag`
- Completed: 2026-10-22T07:45:27.944416-04:00; exit 0; runner elapsed 68.193 s.
  `[SearchEngine volume] iterations=0 elapsed=10.391021778000001 total-domain-bits=25042004.224216491 remaining-domain-bits=25042004.223441347 log2-remaining-fraction=-0.00077514350414276123 eliminated-bits=0.00077514350414276123 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99859241765504 normalized=0.9999530805885013 final-gap=0.99975206035440611 final-incumbent=235.77108799999999 final-bound=0.058457000000000002 first-incumbent-elapsed=24.322883129119873 samples=5`
  `[SearchEngine] New best cost: 235.771 at iteration 0 source=root-heuristic event-epoch=1792669500.4597759 program-deadline=1792669500.6369042`
- First incumbent: 24.322883129 s, cost 235.771; best cost 235.771088.

### I281_standin_g9_n9_diag
- In progress: 2026-10-22T07:45:34.350077-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I281_standin_g9_n9_diag.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I281_standin_g9_n9_diag.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I281_standin_g9_n9_diag.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I281_standin_g9_n9_diag`
- Completed: 2026-10-22T07:46:46.248886-04:00; exit 0; runner elapsed 71.897 s.
  `[SearchEngine volume] iterations=0 elapsed=11.227849255000001 total-domain-bits=28389836.302245375 remaining-domain-bits=28389836.30139529 log2-remaining-fraction=-0.00085008516907691956 eliminated-bits=0.00085008516907691956 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991822159079813 normalized=0.99972740530266047 final-gap=0.99783196189684176 final-incumbent=27.161422999999999 final-bound=0.058887000000000002 first-incumbent-elapsed=26.227999448776245 samples=5`
  `[SearchEngine] New best cost: 27.1614 at iteration 0 source=root-heuristic event-epoch=1792669578.3031399 program-deadline=1792669576.5751526`
- First incumbent: 26.227999449 s, cost 27.1614; best cost 27.161423.

### I282_c019_build
- In progress: 2026-10-22T07:58:15.091049-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I282_c019_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I282_c019_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I282_c019_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:00:31.697327-04:00; exit 0; runner elapsed 136.603 s.

### I283_c019_n9
- In progress: 2026-10-22T08:00:31.790776-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I283_c019_n9.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I283_c019_n9.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I283_c019_n9.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I283_c019_n9`
- Completed: 2026-10-22T08:01:43.760476-04:00; exit 0; runner elapsed 71.968 s.
  `[SearchEngine volume] iterations=0 elapsed=11.197419597 total-domain-bits=28389836.302245375 remaining-domain-bits=28389836.30139529 log2-remaining-fraction=-0.00085008516907691956 eliminated-bits=0.00085008516907691956 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991988514222125 normalized=0.99973295047407085 final-gap=0.99783196189684176 final-incumbent=27.161422999999999 final-bound=0.058887000000000002 first-incumbent-elapsed=26.304730176925659 samples=5`
  `[SearchEngine] New best cost: 27.1614 at iteration 0 source=root-heuristic event-epoch=1792670475.6286883 program-deadline=1792670473.8239701`
- First incumbent: 26.304730177 s, cost 27.1614; best cost 27.161423.

### I284_c019_n9_confirm1
- In progress: 2026-10-22T08:01:50.289059-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I284_c019_n9_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I284_c019_n9_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I284_c019_n9_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I284_c019_n9_confirm1`
- Completed: 2026-10-22T08:03:01.663007-04:00; exit 0; runner elapsed 71.372 s.
  `[SearchEngine volume] iterations=0 elapsed=11.266842348999999 total-domain-bits=28389836.302245375 remaining-domain-bits=28389836.30139529 log2-remaining-fraction=-0.00085008516907691956 eliminated-bits=0.00085008516907691956 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991480506415435 normalized=0.99971601688051448 final-gap=0.99783196189684176 final-incumbent=27.161422999999999 final-bound=0.058887000000000002 first-incumbent-elapsed=26.07041335105896 samples=5`
  `[SearchEngine] New best cost: 27.1614 at iteration 0 source=root-heuristic event-epoch=1792670553.7899604 program-deadline=1792670552.2195599`
- First incumbent: 26.070413351 s, cost 27.1614; best cost 27.161423.

### I285_c019_n9_confirm2
- In progress: 2026-10-22T08:03:01.737397-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I285_c019_n9_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I285_c019_n9_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I285_c019_n9_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I285_c019_n9_confirm2`
- Completed: 2026-10-22T08:04:12.830439-04:00; exit 0; runner elapsed 71.092 s.
  `[SearchEngine volume] iterations=0 elapsed=11.135149246999999 total-domain-bits=28389836.302245375 remaining-domain-bits=28389836.30139529 log2-remaining-fraction=-0.00085008516907691956 eliminated-bits=0.00085008516907691956 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991677273919329 normalized=0.99972257579731094 final-gap=0.99783196189684176 final-incumbent=27.161422999999999 final-bound=0.058887000000000002 first-incumbent-elapsed=26.161171674728394 samples=5`
  `[SearchEngine] New best cost: 27.1614 at iteration 0 source=root-heuristic event-epoch=1792670625.0412841 program-deadline=1792670623.3801265`
- First incumbent: 26.161171675 s, cost 27.1614; best cost 27.161423.

### I286_c019_n8_regression
- In progress: 2026-10-22T08:04:12.897893-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I286_c019_n8_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I286_c019_n8_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I286_c019_n8_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 8 --ignore-checkpoints --run-dir logs/gemma_output_runs/I286_c019_n8_regression`
- Completed: 2026-10-22T08:05:21.137571-04:00; exit 0; runner elapsed 68.236 s.
  `[SearchEngine volume] iterations=0 elapsed=11.352885283000001 total-domain-bits=25042004.224216491 remaining-domain-bits=25042004.223441347 log2-remaining-fraction=-0.00077514350414276123 eliminated-bits=0.00077514350414276123 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.998510036722799 normalized=0.99995033455742666 final-gap=0.99975206035440611 final-incumbent=235.77108799999999 final-bound=0.058457000000000002 first-incumbent-elapsed=23.990621089935303 samples=5`
  `[SearchEngine] New best cost: 235.771 at iteration 0 source=root-heuristic event-epoch=1792670692.8813422 program-deadline=1792670693.3907349`
- First incumbent: 23.990621090 s, cost 235.771; best cost 235.771088.

### I287_c019_oracle
- In progress: 2026-10-22T08:05:28.714422-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I287_c019_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I287_c019_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I287_c019_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T08:05:33.189494-04:00; exit 0; runner elapsed 4.473 s.
- Oracle verdict: 48/48 cases passed.

### I288_c019_search_gap
- In progress: 2026-10-22T08:05:33.259001-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I288_c019_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I288_c019_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I288_c019_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T08:05:33.283091-04:00; exit 0; runner elapsed 0.022 s.

### I289_c019_native__private_layout
- In progress: 2026-10-22T08:05:33.358407-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I289_c019_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I289_c019_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I289_c019_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T08:05:33.375791-04:00; exit 1; runner elapsed 0.016 s.

### I290_c019_native__write_after_read_cache
- In progress: 2026-10-22T08:05:33.448973-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I290_c019_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I290_c019_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I290_c019_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T08:05:33.464907-04:00; exit 1; runner elapsed 0.014 s.

### I291_c019_native__incremental_propagators
- In progress: 2026-10-22T08:05:33.537958-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I291_c019_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I291_c019_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I291_c019_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T08:05:33.552490-04:00; exit 1; runner elapsed 0.013 s.

### I292_c019_native__start_rank_window
- In progress: 2026-10-22T08:05:33.625768-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I292_c019_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I292_c019_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I292_c019_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T08:05:33.642944-04:00; exit 1; runner elapsed 0.015 s.

### I293_c019_native__selection_reachability
- In progress: 2026-10-22T08:05:33.721025-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I293_c019_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I293_c019_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I293_c019_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T08:05:33.736720-04:00; exit 1; runner elapsed 0.013 s.

### I294_c019_native__relaxation_bound
- In progress: 2026-10-22T08:05:33.811228-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I294_c019_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I294_c019_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I294_c019_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T08:05:33.830586-04:00; exit 1; runner elapsed 0.016 s.

### I295_c019_native__prune_egraph
- In progress: 2026-10-22T08:05:33.910093-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I295_c019_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I295_c019_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I295_c019_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T08:05:33.925042-04:00; exit 1; runner elapsed 0.013 s.

### I296_c019_native__private_layout
- In progress: 2026-10-22T08:05:43.770468-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I296_c019_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I296_c019_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I296_c019_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T08:05:43.786370-04:00; exit 1; runner elapsed 0.013 s.

### I297_c019_native__write_after_read_cache
- In progress: 2026-10-22T08:05:43.860127-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I297_c019_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I297_c019_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I297_c019_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T08:05:43.877578-04:00; exit 1; runner elapsed 0.015 s.

### I298_c019_native__incremental_propagators
- In progress: 2026-10-22T08:05:43.953264-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I298_c019_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I298_c019_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I298_c019_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T08:05:43.969745-04:00; exit 1; runner elapsed 0.015 s.

### I299_c019_native__start_rank_window
- In progress: 2026-10-22T08:05:44.044773-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I299_c019_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I299_c019_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I299_c019_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T08:05:44.060071-04:00; exit 1; runner elapsed 0.013 s.

### I300_c019_native__selection_reachability
- In progress: 2026-10-22T08:05:44.130938-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I300_c019_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I300_c019_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I300_c019_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T08:05:44.147898-04:00; exit 1; runner elapsed 0.015 s.

### I301_c019_native__relaxation_bound
- In progress: 2026-10-22T08:05:44.220520-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I301_c019_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I301_c019_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I301_c019_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T08:05:44.235851-04:00; exit 1; runner elapsed 0.014 s.

### I302_c019_native__prune_egraph
- In progress: 2026-10-22T08:05:44.312003-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I302_c019_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I302_c019_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I302_c019_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python /home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T08:05:44.330571-04:00; exit 1; runner elapsed 0.017 s.

### I303_c019_native__private_layout
- In progress: 2026-10-22T08:06:00.309629-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I303_c019_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I303_c019_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I303_c019_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T08:06:00.533000-04:00; exit 0; runner elapsed 0.221 s.

### I304_c019_native__write_after_read_cache
- In progress: 2026-10-22T08:06:00.618326-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I304_c019_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I304_c019_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I304_c019_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T08:06:00.825866-04:00; exit 0; runner elapsed 0.206 s.

### I305_c019_native__incremental_propagators
- In progress: 2026-10-22T08:06:00.913049-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I305_c019_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I305_c019_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I305_c019_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T08:06:01.132735-04:00; exit 0; runner elapsed 0.218 s.

### I306_c019_native__start_rank_window
- In progress: 2026-10-22T08:06:01.223970-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I306_c019_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I306_c019_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I306_c019_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T08:06:01.422882-04:00; exit 0; runner elapsed 0.196 s.

### I307_c019_native__selection_reachability
- In progress: 2026-10-22T08:06:01.506727-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I307_c019_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I307_c019_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I307_c019_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T08:06:01.754400-04:00; exit 0; runner elapsed 0.246 s.

### I308_c019_native__relaxation_bound
- In progress: 2026-10-22T08:06:01.871018-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I308_c019_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I308_c019_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I308_c019_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T08:07:52.804002-04:00; exit 0; runner elapsed 110.931 s.

### I309_c019_native__prune_egraph
- In progress: 2026-10-22T08:07:52.888148-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I309_c019_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I309_c019_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I309_c019_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T08:07:53.108782-04:00; exit 0; runner elapsed 0.218 s.

### I310_c019_n10
- In progress: 2026-10-22T08:08:00.545822-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I310_c019_n10.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I310_c019_n10.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I310_c019_n10.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 10 --ignore-checkpoints --run-dir logs/gemma_output_runs/I310_c019_n10`
- Completed: 2026-10-22T08:11:41.772260-04:00; exit 1; runner elapsed 221.223 s.
  `[SearchEngine] New best cost: 28.2337 at iteration 0 source=root-heuristic event-epoch=1792671100.7861922 program-deadline=1792671095.9920523`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 28.2337 (no supervisor gap summary); best cost 28.2337.

C019 notes: I261/I262 (C018 tree, unevaluated before this session) parallelize per-bucket parent/index setup and pool the reduction of several representatives. I263 (G=9, N=9 diagnostic) still spent 7.6 s in root propagation; perf (`perf record -F 97 --call-graph dwarf` on the compile phase, main-thread samples under `runPropagators`) attributed it to serial per-bucket setup. I264/I265 were build invocation/compile errors; I266 built the first parallel setups, but SelectionReachability and InputPrune setups were reached from per-variable events rather than the root call (I268: 6.27 s), fixed in I269 (I270: 5.06 s). I271 (G=4, N=4) and I272 (G=6, N=6, 19.44 s) passed; I273 (G=8, N=8) and I274 (G=3, N=9) found buckets whose relaxation root is infeasible. I275-I278 traced it (temporary `[RelaxDiag]`) to class 7069 whose only acyclic alternative needed class 16335 with no e-nodes; I279 (temporary `[EmptyChildDiag]` after each reduction step) showed the empty classes appear only at the final `rebuild(true, false)`; cause and fix as in the checkpoint. I280/I281 (env G=8/9) passed; diagnostics were removed and the default grouping added for I282. I289-I302 failed only because the runner prefixed the native binary with Python (use `--native-command`); I303-I309 are the real native gate runs. I310 is the N=10 miss.

### I311_c020_trial_bound_skip_build
- In progress: 2026-10-22T08:14:58.239531-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I311_c020_trial_bound_skip_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I311_c020_trial_bound_skip_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I311_c020_trial_bound_skip_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:17:18.936479-04:00; exit 0; runner elapsed 140.695 s.

### I312_c020_n10
- In progress: 2026-10-22T08:17:19.027239-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I312_c020_n10.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I312_c020_n10.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I312_c020_n10.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 10 --ignore-checkpoints --run-dir logs/gemma_output_runs/I312_c020_n10`
- Completed: 2026-10-22T08:18:08.684934-04:00; exit 1; runner elapsed 49.656 s.
  `[SearchEngine volume] iterations=0 elapsed=12.104176312 total-domain-bits=31713062.947669145 remaining-domain-bits=31713062.946743578 log2-remaining-fraction=-0.00092556700110435486 eliminated-bits=0.00092556700110435486 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 28.2337 at iteration 0 source=root-heuristic event-epoch=1792671486.9829328 program-deadline=1792671482.9181993`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 28.2337 (no supervisor gap summary); best cost 28.2337.

### I313_c020_primal_free_build
- In progress: 2026-10-22T08:19:37.492030-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I313_c020_primal_free_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I313_c020_primal_free_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I313_c020_primal_free_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:21:57.659892-04:00; exit 0; runner elapsed 140.166 s.

### I314_c020_n10
- In progress: 2026-10-22T08:21:57.745174-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I314_c020_n10.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I314_c020_n10.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I314_c020_n10.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 10 --ignore-checkpoints --run-dir logs/gemma_output_runs/I314_c020_n10`
- Completed: 2026-10-22T08:23:14.272974-04:00; exit 0; runner elapsed 76.526 s.
  `[SearchEngine volume] iterations=0 elapsed=10.867815405 total-domain-bits=31713062.947669145 remaining-domain-bits=31713062.946743578 log2-remaining-fraction=-0.00092556700110435486 eliminated-bits=0.00092556700110435486 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.996561845635451 normalized=0.999885394854515 final-gap=0.99793331913515904 final-incumbent=28.233677 final-bound=0.058349999999999999 first-incumbent-elapsed=28.336388349533081 samples=4`
  `[SearchEngine] New best cost: 28.2337 at iteration 0 source=root-heuristic event-epoch=1792671765.4787307 program-deadline=1792671761.6423542`
- First incumbent: 28.336388350 s, cost 28.2337; best cost 28.233677.

### I315_c020_knuth_dp_build
- In progress: 2026-10-22T08:26:03.265662-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I315_c020_knuth_dp_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I315_c020_knuth_dp_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I315_c020_knuth_dp_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:28:22.178626-04:00; exit 0; runner elapsed 138.911 s.

### I316_c020_build
- In progress: 2026-10-22T08:33:07.499473-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I316_c020_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I316_c020_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I316_c020_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:33:08.788021-04:00; exit 0; runner elapsed 1.287 s.

### I317_c020_n10
- In progress: 2026-10-22T08:33:08.878812-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I317_c020_n10.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I317_c020_n10.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I317_c020_n10.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 10 --ignore-checkpoints --run-dir logs/gemma_output_runs/I317_c020_n10`
- Completed: 2026-10-22T08:34:18.880129-04:00; exit 0; runner elapsed 70.000 s.
  `[SearchEngine volume] iterations=0 elapsed=10.823541763 total-domain-bits=31713062.947669145 remaining-domain-bits=31713062.946743578 log2-remaining-fraction=-0.00092556700110435486 eliminated-bits=0.00092556700110435486 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.989733003558214 normalized=0.99965776678527385 final-gap=0.99793331913515904 final-incumbent=28.233677 final-bound=0.058349999999999999 first-incumbent-elapsed=25.032132625579834 samples=4`
  `[SearchEngine] New best cost: 28.2337 at iteration 0 source=root-heuristic event-epoch=1792672430.116117 program-deadline=1792672429.5839968`
- First incumbent: 25.032132626 s, cost 28.2337; best cost 28.233677.

### I318_c020_n12
- In progress: 2026-10-22T08:34:27.063292-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I318_c020_n12.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I318_c020_n12.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I318_c020_n12.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I318_c020_n12`
- Completed: 2026-10-22T08:38:58.282585-04:00; exit 1; runner elapsed 271.216 s.
  `[SearchEngine volume] iterations=0 elapsed=12.743253654 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792672737.301146 program-deadline=1792672732.4654572`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 28.2464 (no supervisor gap summary); best cost 28.2464.

### I319_c020_n11
- In progress: 2026-10-22T08:39:11.317823-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I319_c020_n11.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I319_c020_n11.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I319_c020_n11.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 11 --ignore-checkpoints --run-dir logs/gemma_output_runs/I319_c020_n11`
- Completed: 2026-10-22T08:43:42.275498-04:00; exit 0; runner elapsed 270.956 s.
  `[SearchEngine volume] iterations=0 elapsed=11.992455604 total-domain-bits=35037325.620663755 remaining-domain-bits=35037325.619662695 log2-remaining-fraction=-0.0010010600090026855 eliminated-bits=0.0010010600090026855 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.994643646909257 normalized=0.99982145489697527 final-gap=0.99790970536489521 final-incumbent=27.965914000000001 final-bound=0.058457000000000002 first-incumbent-elapsed=27.437512874603271 samples=4`
  `[SearchEngine] New best cost: 27.9659 at iteration 0 source=root-heuristic event-epoch=1792672991.676038 program-deadline=1792672988.738538`
- First incumbent: 27.437512875 s, cost 27.9659; best cost 27.965914.

### I320_c020_n11_confirm1
- In progress: 2026-10-22T08:43:48.422310-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I320_c020_n11_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I320_c020_n11_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I320_c020_n11_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 11 --ignore-checkpoints --run-dir logs/gemma_output_runs/I320_c020_n11_confirm1`
- Completed: 2026-10-22T08:45:04.439158-04:00; exit 0; runner elapsed 76.014 s.
  `[SearchEngine volume] iterations=0 elapsed=12.002730536 total-domain-bits=35037325.620663755 remaining-domain-bits=35037325.619662695 log2-remaining-fraction=-0.0010010600090026855 eliminated-bits=0.0010010600090026855 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.994521689499486 normalized=0.9998173896499829 final-gap=0.99790970536489521 final-incumbent=27.965914000000001 final-bound=0.058457000000000002 first-incumbent-elapsed=27.379168272018433 samples=4`
  `[SearchEngine] New best cost: 27.9659 at iteration 0 source=root-heuristic event-epoch=1792673073.4095016 program-deadline=1792673070.530344`
- First incumbent: 27.379168272 s, cost 27.9659; best cost 27.965914.

### I321_c020_n11_confirm2
- In progress: 2026-10-22T08:45:04.517356-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I321_c020_n11_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I321_c020_n11_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I321_c020_n11_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 11 --ignore-checkpoints --run-dir logs/gemma_output_runs/I321_c020_n11_confirm2`
- Completed: 2026-10-22T08:46:20.856116-04:00; exit 0; runner elapsed 76.337 s.
  `[SearchEngine volume] iterations=0 elapsed=11.754345774000001 total-domain-bits=35037325.620663755 remaining-domain-bits=35037325.619662695 log2-remaining-fraction=-0.0010010600090026855 eliminated-bits=0.0010010600090026855 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99413688836783 normalized=0.99980456294559439 final-gap=0.99790970536489521 final-incumbent=27.965914000000001 final-bound=0.058457000000000002 first-incumbent-elapsed=27.19507884979248 samples=4`
  `[SearchEngine] New best cost: 27.9659 at iteration 0 source=root-heuristic event-epoch=1792673149.9344935 program-deadline=1792673147.2394271`
- First incumbent: 27.195078850 s, cost 27.9659; best cost 27.965914.

### I322_c020_n9_regression
- In progress: 2026-10-22T08:46:20.931794-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I322_c020_n9_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I322_c020_n9_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I322_c020_n9_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 9 --ignore-checkpoints --run-dir logs/gemma_output_runs/I322_c020_n9_regression`
- Completed: 2026-10-22T08:47:27.631541-04:00; exit 0; runner elapsed 66.698 s.
  `[SearchEngine volume] iterations=0 elapsed=10.653905884 total-domain-bits=28389836.302245375 remaining-domain-bits=28389836.30139529 log2-remaining-fraction=-0.00085008516907691956 eliminated-bits=0.00085008516907691956 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.98465489339457 normalized=0.99948849644648563 final-gap=0.99783196189684176 final-incumbent=27.161422999999999 final-bound=0.058887000000000002 first-incumbent-elapsed=22.922123193740845 samples=4`
  `[SearchEngine] New best cost: 27.1614 at iteration 0 source=root-heuristic event-epoch=1792673219.0736594 program-deadline=1792673220.651547`
- First incumbent: 22.922123194 s, cost 27.1614; best cost 27.161423.

### I323_c020_oracle
- In progress: 2026-10-22T08:47:32.865603-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I323_c020_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I323_c020_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I323_c020_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T08:47:37.340088-04:00; exit 0; runner elapsed 4.471 s.
- Oracle verdict: 48/48 cases passed.

### I324_c020_search_gap
- In progress: 2026-10-22T08:47:37.413552-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I324_c020_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I324_c020_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I324_c020_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T08:47:37.439942-04:00; exit 0; runner elapsed 0.024 s.

### I325_c020_native__private_layout
- In progress: 2026-10-22T08:47:37.512671-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I325_c020_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I325_c020_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I325_c020_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T08:47:37.716753-04:00; exit 0; runner elapsed 0.203 s.

### I326_c020_native__write_after_read_cache
- In progress: 2026-10-22T08:47:37.799440-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I326_c020_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I326_c020_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I326_c020_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T08:47:38.010657-04:00; exit 0; runner elapsed 0.210 s.

### I327_c020_native__incremental_propagators
- In progress: 2026-10-22T08:47:38.097972-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I327_c020_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I327_c020_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I327_c020_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T08:47:38.315324-04:00; exit 0; runner elapsed 0.215 s.

### I328_c020_native__start_rank_window
- In progress: 2026-10-22T08:47:38.402729-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I328_c020_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I328_c020_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I328_c020_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T08:47:38.588634-04:00; exit 0; runner elapsed 0.182 s.

### I329_c020_native__selection_reachability
- In progress: 2026-10-22T08:47:38.670908-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I329_c020_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I329_c020_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I329_c020_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T08:47:38.919549-04:00; exit 0; runner elapsed 0.246 s.

### I330_c020_native__relaxation_bound
- In progress: 2026-10-22T08:47:39.028840-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I330_c020_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I330_c020_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I330_c020_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T08:49:29.169000-04:00; exit 0; runner elapsed 110.138 s.

### I331_c020_native__prune_egraph
- In progress: 2026-10-22T08:49:29.255000-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I331_c020_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I331_c020_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I331_c020_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T08:49:29.472500-04:00; exit 0; runner elapsed 0.216 s.

C020 notes: I311/I312 skip bound-only propagators in trial plans (tryPlan 4.43 -> 3.71 s at N=10). N=9 perf (I282 build) showed ~0.75 s of main-thread `~RelaxationBucket` after the heuristic and ~0.35 s serial `relaxationSignature`; I313/I314 moved the free to a detached thread, parallelized signatures and made the per-bucket threads dynamic: first N=10 pass (29.286 s). Reduction profile: `computeENodeHeuristicCosts` ~38 CPU-s across the 11 representatives (chaotic iteration on per-t graphs with large cyclic parts); I315 replaced it with Knuth's algorithm; a temporary comparison build (not recorded; `TG_CHECK_DP`, N=2 and N=4 compiles through a 30 s pinned-environment driver) found identical values; I316 is the clean build. I317 N=10 26.025 s, I318 N=12 miss, I319-I322 N=11 reference, confirmations and N=9 regression, I323-I331 gates.

### I332_c021_pk_walk_split_state_build
- In progress: 2026-10-22T08:51:57.879268-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I332_c021_pk_walk_split_state_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I332_c021_pk_walk_split_state_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I332_c021_pk_walk_split_state_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T08:54:17.564169-04:00; exit 0; runner elapsed 139.682 s.

### I333_c021_n12
- In progress: 2026-10-22T08:54:22.047084-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I333_c021_n12.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I333_c021_n12.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I333_c021_n12.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I333_c021_n12`
- Completed: 2026-10-22T08:55:11.258993-04:00; exit 1; runner elapsed 49.210 s.
  `[SearchEngine volume] iterations=0 elapsed=12.58624691 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792673710.384546 program-deadline=1792673705.4658618`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 28.2464 (no supervisor gap summary); best cost 28.2464.

### I334_c021_malloc_build
- In progress: 2026-10-22T09:16:56.067367-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I334_c021_malloc_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I334_c021_malloc_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I334_c021_malloc_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T09:19:16.280367-04:00; exit 0; runner elapsed 140.209 s.

### I335_c021_n12
- In progress: 2026-10-22T09:19:16.368287-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I335_c021_n12.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I335_c021_n12.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I335_c021_n12.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I335_c021_n12`
- Completed: 2026-10-22T09:20:32.883475-04:00; exit 0; runner elapsed 76.513 s.
  `[SearchEngine volume] iterations=0 elapsed=10.866618918 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991220076956687 normalized=0.9997073358985562 final-gap=0.99793687238761286 final-incumbent=28.246435000000002 final-bound=0.058276000000000001 first-incumbent-elapsed=25.744362592697144 samples=4`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792675198.9929347 program-deadline=1792675197.7485852`
- First incumbent: 25.744362593 s, cost 28.2464; best cost 28.246435.

### I336_c021_n14
- In progress: 2026-10-22T09:20:39.213695-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I336_c021_n14.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I336_c021_n14.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I336_c021_n14.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 14 --ignore-checkpoints --run-dir logs/gemma_output_runs/I336_c021_n14`
- Completed: 2026-10-22T09:24:07.759729-04:00; exit 1; runner elapsed 208.544 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I337_c021_n13
- In progress: 2026-10-22T09:24:19.087711-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I337_c021_n13.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I337_c021_n13.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I337_c021_n13.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 13 --ignore-checkpoints --run-dir logs/gemma_output_runs/I337_c021_n13`
- Completed: 2026-10-22T09:29:11.873177-04:00; exit 0; runner elapsed 292.784 s.
  `[SearchEngine volume] iterations=0 elapsed=12.320634782000001 total-domain-bits=41685345.855125785 remaining-domain-bits=41685345.853973746 log2-remaining-fraction=-0.00115203857421875 eliminated-bits=0.00115203857421875 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99939214788926 normalized=0.99997973826297537 final-gap=0.99959541803551333 final-incumbent=158.328857 final-bound=0.064057000000000003 first-incumbent-elapsed=28.49757981300354 samples=4`
  `[SearchEngine] New best cost: 158.329 at iteration 0 source=root-heuristic event-epoch=1792675714.1119623 program-deadline=1792675710.1143937`
- First incumbent: 28.497579813 s, cost 158.329; best cost 158.328857.

### I338_c021_n12_confirm1
- In progress: 2026-10-22T09:29:29.234086-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I338_c021_n12_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I338_c021_n12_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I338_c021_n12_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I338_c021_n12_confirm1`
- Completed: 2026-10-22T09:30:44.230141-04:00; exit 0; runner elapsed 74.994 s.
  `[SearchEngine volume] iterations=0 elapsed=10.839901985999999 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.991581929411055 normalized=0.99971939764703521 final-gap=0.99793687238761286 final-incumbent=28.246435000000002 final-bound=0.058276000000000001 first-incumbent-elapsed=25.919752836227417 samples=4`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792675812.3766506 program-deadline=1792675810.9569113`
- First incumbent: 25.919752836 s, cost 28.2464; best cost 28.246435.

### I339_c021_n12_confirm2
- In progress: 2026-10-22T09:30:44.301338-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I339_c021_n12_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I339_c021_n12_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I339_c021_n12_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I339_c021_n12_confirm2`
- Completed: 2026-10-22T09:31:59.285699-04:00; exit 0; runner elapsed 74.983 s.
  `[SearchEngine volume] iterations=0 elapsed=11.001028104 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.992051901297124 normalized=0.99973506337657081 final-gap=0.99793687238761286 final-incumbent=28.246435000000002 final-bound=0.058276000000000001 first-incumbent-elapsed=26.147548675537109 samples=4`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792675887.513941 program-deadline=1792675885.8664038`
- First incumbent: 26.147548676 s, cost 28.2464; best cost 28.246435.

### I340_c021_n11_regression
- In progress: 2026-10-22T09:31:59.359077-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I340_c021_n11_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I340_c021_n11_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I340_c021_n11_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 11 --ignore-checkpoints --run-dir logs/gemma_output_runs/I340_c021_n11_regression`
- Completed: 2026-10-22T09:33:10.372659-04:00; exit 0; runner elapsed 71.012 s.
  `[SearchEngine volume] iterations=0 elapsed=10.529988696 total-domain-bits=35037325.620663755 remaining-domain-bits=35037325.619662695 log2-remaining-fraction=-0.0010010600090026855 eliminated-bits=0.0010010600090026855 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.987797947727625 normalized=0.99959326492425415 final-gap=0.99790970536489521 final-incumbent=27.965914000000001 final-bound=0.058457000000000002 first-incumbent-elapsed=24.162520408630371 samples=4`
  `[SearchEngine] New best cost: 27.9659 at iteration 0 source=root-heuristic event-epoch=1792675959.4789777 program-deadline=1792675959.8164687`
- First incumbent: 24.162520409 s, cost 27.9659; best cost 27.965914.

### I341_c021_oracle
- In progress: 2026-10-22T09:33:10.446979-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I341_c021_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I341_c021_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I341_c021_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T09:33:15.071254-04:00; exit 0; runner elapsed 4.623 s.
- Oracle verdict: 48/48 cases passed.

### I342_c021_search_gap
- In progress: 2026-10-22T09:33:15.143064-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I342_c021_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I342_c021_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I342_c021_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T09:33:15.173423-04:00; exit 0; runner elapsed 0.027 s.

### I343_c021_native__private_layout
- In progress: 2026-10-22T09:33:15.245510-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I343_c021_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I343_c021_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I343_c021_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T09:33:15.453237-04:00; exit 0; runner elapsed 0.204 s.

### I344_c021_native__write_after_read_cache
- In progress: 2026-10-22T09:33:15.528648-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I344_c021_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I344_c021_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I344_c021_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T09:33:15.735484-04:00; exit 0; runner elapsed 0.203 s.

### I345_c021_native__incremental_propagators
- In progress: 2026-10-22T09:33:15.817535-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I345_c021_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I345_c021_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I345_c021_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T09:33:16.020283-04:00; exit 0; runner elapsed 0.201 s.

### I346_c021_native__start_rank_window
- In progress: 2026-10-22T09:33:16.103232-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I346_c021_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I346_c021_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I346_c021_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T09:33:16.312895-04:00; exit 0; runner elapsed 0.208 s.

### I347_c021_native__selection_reachability
- In progress: 2026-10-22T09:33:16.395149-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I347_c021_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I347_c021_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I347_c021_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T09:33:16.640202-04:00; exit 0; runner elapsed 0.243 s.

### I348_c021_native__relaxation_bound
- In progress: 2026-10-22T09:33:16.738049-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I348_c021_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I348_c021_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I348_c021_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T09:35:07.379585-04:00; exit 0; runner elapsed 110.640 s.

### I349_c021_native__prune_egraph
- In progress: 2026-10-22T09:35:07.459478-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I349_c021_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I349_c021_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I349_c021_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T09:35:07.680683-04:00; exit 0; runner elapsed 0.219 s.

C021 notes: I332/I333 PearceKelly shared walk and split propagation-state tasks (root propagation 6.11 -> 5.81 s at N=12, still a miss). Per-group saturation logging showed middle groups at ~8 s under contention; inline closures and 6 groups were measured (not recorded, 30 s diagnostics) and rejected. An LD_PRELOAD jemalloc diagnostic cut N=12 search start from 16.6 s to 10.7 s; GLIBC_TUNABLES with mmap/trim/top-pad recovered 2 s of it; I334 sets those values from code. I335 N=12 pass, I336 N=14 miss, I337 N=13 marginal pass, I338-I340 confirmations and regression, I341-I349 gates.

### I350_c022_blocking_wait_build
- In progress: 2026-10-22T09:37:52.364046-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I350_c022_blocking_wait_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I350_c022_blocking_wait_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I350_c022_blocking_wait_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T09:43:58.967143-04:00; exit 0; runner elapsed 366.601 s.

### I351_c022_n13
- In progress: 2026-10-22T09:43:59.064112-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I351_c022_n13.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I351_c022_n13.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I351_c022_n13.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 13 --ignore-checkpoints --run-dir logs/gemma_output_runs/I351_c022_n13`
- Completed: 2026-10-22T09:45:22.796995-04:00; exit 0; runner elapsed 83.730 s.
  `[SearchEngine volume] iterations=0 elapsed=12.3056786 total-domain-bits=41685345.855125785 remaining-domain-bits=41685345.853973746 log2-remaining-fraction=-0.00115203857421875 eliminated-bits=0.00115203857421875 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999300586746003 normalized=0.9999766862248668 final-gap=0.99959541803551333 final-incumbent=158.328857 final-bound=0.064057000000000003 first-incumbent-elapsed=28.27126932144165 samples=4`
  `[SearchEngine] New best cost: 158.329 at iteration 0 source=root-heuristic event-epoch=1792676685.1796668 program-deadline=1792676681.4084101`
- First incumbent: 28.271269321 s, cost 158.329; best cost 158.328857.

### I352_c022_trial_skips_build
- In progress: 2026-10-22T09:46:53.036532-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I352_c022_trial_skips_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I352_c022_trial_skips_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I352_c022_trial_skips_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T09:49:12.301668-04:00; exit 0; runner elapsed 139.263 s.

### I353_c022_n13
- In progress: 2026-10-22T09:49:17.030141-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I353_c022_n13.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I353_c022_n13.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I353_c022_n13.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 13 --ignore-checkpoints --run-dir logs/gemma_output_runs/I353_c022_n13`
- Completed: 2026-10-22T09:50:40.406823-04:00; exit 0; runner elapsed 83.375 s.
  `[SearchEngine volume] iterations=0 elapsed=12.047297217000001 total-domain-bits=41685345.855125785 remaining-domain-bits=41685345.853973746 log2-remaining-fraction=-0.00115203857421875 eliminated-bits=0.00115203857421875 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999182764985402 normalized=0.99997275883284675 final-gap=0.99959541803551333 final-incumbent=158.328857 final-bound=0.064057000000000003 first-incumbent-elapsed=27.980050802230835 samples=4`
  `[SearchEngine] New best cost: 158.329 at iteration 0 source=root-heuristic event-epoch=1792677003.1754727 program-deadline=1792676999.6954341`
- First incumbent: 27.980050802 s, cost 158.329; best cost 158.328857.

### I354_c022_reconcile_skip_build
- In progress: 2026-10-22T09:51:18.046928-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I354_c022_reconcile_skip_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I354_c022_reconcile_skip_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I354_c022_reconcile_skip_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T09:53:37.297537-04:00; exit 0; runner elapsed 139.249 s.

### I355_c022_n13
- In progress: 2026-10-22T09:53:37.383260-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I355_c022_n13.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I355_c022_n13.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I355_c022_n13.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 13 --ignore-checkpoints --run-dir logs/gemma_output_runs/I355_c022_n13`
- Completed: 2026-10-22T09:55:00.535295-04:00; exit 0; runner elapsed 83.150 s.
  `[SearchEngine volume] iterations=0 elapsed=12.003242782999999 total-domain-bits=41685345.855125785 remaining-domain-bits=41685345.853973746 log2-remaining-fraction=-0.00115203857421875 eliminated-bits=0.00115203857421875 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99901819501612 normalized=0.99996727316720402 final-gap=0.99959541803551333 final-incumbent=158.328857 final-bound=0.064057000000000003 first-incumbent-elapsed=27.573285341262817 samples=4`
  `[SearchEngine] New best cost: 158.329 at iteration 0 source=root-heuristic event-epoch=1792677262.6825867 program-deadline=1792677259.6093137`
- First incumbent: 27.573285341 s, cost 158.329; best cost 158.328857.

### I356_c022_parallel_vars_build
- In progress: 2026-10-22T09:58:49.806559-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I356_c022_parallel_vars_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I356_c022_parallel_vars_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I356_c022_parallel_vars_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T09:58:50.332706-04:00; exit 0; runner elapsed 0.524 s.

### I357_c022_n13
- In progress: 2026-10-22T09:58:50.417565-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I357_c022_n13.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I357_c022_n13.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I357_c022_n13.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 13 --ignore-checkpoints --run-dir logs/gemma_output_runs/I357_c022_n13`
- Completed: 2026-10-22T10:00:12.412538-04:00; exit 0; runner elapsed 81.993 s.
  `[SearchEngine volume] iterations=0 elapsed=11.328733318999999 total-domain-bits=41685345.855125785 remaining-domain-bits=41685345.853973746 log2-remaining-fraction=-0.00115203857421875 eliminated-bits=0.00115203857421875 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.998553342232039 normalized=0.99995177807440128 final-gap=0.99959541803551333 final-incumbent=158.328857 final-bound=0.064057000000000003 first-incumbent-elapsed=26.424314737319946 samples=4`
  `[SearchEngine] New best cost: 158.329 at iteration 0 source=root-heuristic event-epoch=1792677574.3673956 program-deadline=1792677572.4430933`
- First incumbent: 26.424314737 s, cost 158.329; best cost 158.328857.

### I358_c022_n14
- In progress: 2026-10-22T10:00:18.160709-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I358_c022_n14.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I358_c022_n14.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I358_c022_n14.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 14 --ignore-checkpoints --run-dir logs/gemma_output_runs/I358_c022_n14`
- Completed: 2026-10-22T10:02:02.077521-04:00; exit 0; runner elapsed 103.915 s.
  `[SearchEngine volume] iterations=0 elapsed=11.809514476 total-domain-bits=44877722.271791115 remaining-domain-bits=44877722.270560592 log2-remaining-fraction=-0.0012305229902267456 eliminated-bits=0.0012305229902267456 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999810531684151 normalized=0.99999368438947167 final-gap=0.99990979661016954 final-incumbent=649 final-bound=0.058541999999999997 first-incumbent-elapsed=27.899543285369873 samples=4`
  `[SearchEngine] New best cost: 649 at iteration 0 source=root-heuristic event-epoch=1792677664.8149598 program-deadline=1792677661.4154298`
- First incumbent: 27.899543285 s, cost 649.0; best cost 649.0.

### I359_c022_n14_confirm1
- In progress: 2026-10-22T10:02:11.891837-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I359_c022_n14_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I359_c022_n14_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I359_c022_n14_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 14 --ignore-checkpoints --run-dir logs/gemma_output_runs/I359_c022_n14_confirm1`
- Completed: 2026-10-22T10:03:55.038419-04:00; exit 0; runner elapsed 103.143 s.
  `[SearchEngine volume] iterations=0 elapsed=11.850925313999999 total-domain-bits=44877722.271791115 remaining-domain-bits=44877722.270560592 log2-remaining-fraction=-0.0012305229902267456 eliminated-bits=0.0012305229902267456 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999805074817615 normalized=0.99999350249392049 final-gap=0.99990979661016954 final-incumbent=649 final-bound=0.058541999999999997 first-incumbent-elapsed=27.839048147201538 samples=4`
  `[SearchEngine] New best cost: 649 at iteration 0 source=root-heuristic event-epoch=1792677778.6487589 program-deadline=1792677775.3097255`
- First incumbent: 27.839048147 s, cost 649.0; best cost 649.0.

### I360_c022_n14_confirm2
- In progress: 2026-10-22T10:03:55.109732-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I360_c022_n14_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I360_c022_n14_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I360_c022_n14_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 14 --ignore-checkpoints --run-dir logs/gemma_output_runs/I360_c022_n14_confirm2`
- Completed: 2026-10-22T10:05:39.728154-04:00; exit 0; runner elapsed 104.617 s.
  `[SearchEngine volume] iterations=0 elapsed=11.877110168 total-domain-bits=44877722.271791115 remaining-domain-bits=44877722.270560592 log2-remaining-fraction=-0.0012305229902267456 eliminated-bits=0.0012305229902267456 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999826027240942 normalized=0.99999420090803137 final-gap=0.99990979661016954 final-incumbent=649 final-bound=0.058541999999999997 first-incumbent-elapsed=28.071327924728394 samples=4`
  `[SearchEngine] New best cost: 649 at iteration 0 source=root-heuristic event-epoch=1792677881.9422839 program-deadline=1792677878.3709683`
- First incumbent: 28.071327925 s, cost 649.0; best cost 649.0.

### I361_c022_n12_regression
- In progress: 2026-10-22T10:05:39.802085-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I361_c022_n12_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I361_c022_n12_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I361_c022_n12_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 12 --ignore-checkpoints --run-dir logs/gemma_output_runs/I361_c022_n12_regression`
- Completed: 2026-10-22T10:06:52.862376-04:00; exit 0; runner elapsed 73.059 s.
  `[SearchEngine volume] iterations=0 elapsed=10.39460834 total-domain-bits=38360798.395627551 remaining-domain-bits=38360798.394551001 log2-remaining-fraction=-0.0010765492916107178 eliminated-bits=0.0010765492916107178 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.987853206225275 normalized=0.99959510687417585 final-gap=0.99793687238761286 final-incumbent=28.246435000000002 final-bound=0.058276000000000001 first-incumbent-elapsed=24.112437009811401 samples=4`
  `[SearchEngine] New best cost: 28.2464 at iteration 0 source=root-heuristic event-epoch=1792677980.5609074 program-deadline=1792677980.9484856`
- First incumbent: 24.112437010 s, cost 28.2464; best cost 28.246435.

### I362_c022_oracle
- In progress: 2026-10-22T10:06:58.766595-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I362_c022_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I362_c022_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I362_c022_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T10:07:03.391637-04:00; exit 0; runner elapsed 4.623 s.
- Oracle verdict: 48/48 cases passed.

### I363_c022_search_gap
- In progress: 2026-10-22T10:07:03.465076-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I363_c022_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I363_c022_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I363_c022_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T10:07:03.487137-04:00; exit 0; runner elapsed 0.020 s.

### I364_c022_native__private_layout
- In progress: 2026-10-22T10:07:03.558506-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I364_c022_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I364_c022_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I364_c022_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T10:07:03.757453-04:00; exit 0; runner elapsed 0.196 s.

### I365_c022_native__write_after_read_cache
- In progress: 2026-10-22T10:07:03.838071-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I365_c022_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I365_c022_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I365_c022_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T10:07:04.049794-04:00; exit 0; runner elapsed 0.210 s.

### I366_c022_native__incremental_propagators
- In progress: 2026-10-22T10:07:04.135750-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I366_c022_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I366_c022_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I366_c022_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T10:07:04.343016-04:00; exit 0; runner elapsed 0.205 s.

### I367_c022_native__start_rank_window
- In progress: 2026-10-22T10:07:04.431525-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I367_c022_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I367_c022_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I367_c022_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T10:07:04.637178-04:00; exit 0; runner elapsed 0.203 s.

### I368_c022_native__selection_reachability
- In progress: 2026-10-22T10:07:04.715768-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I368_c022_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I368_c022_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I368_c022_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T10:07:04.976060-04:00; exit 0; runner elapsed 0.259 s.

### I369_c022_native__relaxation_bound
- In progress: 2026-10-22T10:07:05.093674-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I369_c022_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I369_c022_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I369_c022_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T10:08:53.524586-04:00; exit 0; runner elapsed 108.429 s.

### I370_c022_native__prune_egraph
- In progress: 2026-10-22T10:08:53.597108-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I370_c022_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I370_c022_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I370_c022_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T10:08:53.831845-04:00; exit 0; runner elapsed 0.232 s.

C022 notes: profile of N=12 saturation: `EGraph::rebuild` 1761/6069 samples, of which ~400 `sched_yield` in parallel_for waits and ~700 allocation. I350/I351 blocking wait (small gain). I352/I353 trial-plan skips (oracle 48/48). I354/I355 reconcile skip. Search setup profiling showed `addBucketVariables` serial (~1 s at N=13); I356/I357 two-phase parallel creation (oracle, `--selection-reachability`, `--incremental-propagators` pass before the build). I358 first N=14 pass, I359/I360 confirmations, I361 N=12 regression, I362-I370 gates.

### I371_c023_incremental_rebuild_build
- In progress: 2026-10-22T11:50:01.466518-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I371_c023_incremental_rebuild_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I371_c023_incremental_rebuild_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I371_c023_incremental_rebuild_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T11:52:18.521964-04:00; exit 0; runner elapsed 137.053 s.

### I372_c023_n16
- In progress: 2026-10-22T11:53:05.317053-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I372_c023_n16.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I372_c023_n16.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I372_c023_n16.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 16 --ignore-checkpoints --run-dir logs/gemma_output_runs/I372_c023_n16`
- Completed: 2026-10-22T11:57:40.457845-04:00; exit 0; runner elapsed 275.139 s.
  `[SearchEngine volume] iterations=0 elapsed=13.438695742 total-domain-bits=51684281.275955945 remaining-domain-bits=51684281.274578325 log2-remaining-fraction=-0.0013776198029518127 eliminated-bits=0.0013776198029518127 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.992931949014388 normalized=0.99976439830047958 final-gap=0.99812000000246259 final-incumbent=32.487766000000001 final-bound=0.061076999999999999 first-incumbent-elapsed=26.240398406982422 samples=4`
  `[SearchEngine] New best cost: 32.4878 at iteration 0 source=root-heuristic event-epoch=1792684599.0068917 program-deadline=1792684597.2665055`
- First incumbent: 26.240398407 s, cost 32.4878; best cost 32.487766.

### I373_c023_n17
- In progress: 2026-10-22T11:57:50.488779-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I373_c023_n17.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I373_c023_n17.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I373_c023_n17.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 17 --ignore-checkpoints --run-dir logs/gemma_output_runs/I373_c023_n17`
- Completed: 2026-10-22T12:02:08.634298-04:00; exit 0; runner elapsed 258.143 s.
  `[SearchEngine volume] iterations=0 elapsed=14.089580098000001 total-domain-bits=54225133.898260109 remaining-domain-bits=54225133.896784946 log2-remaining-fraction=-0.0014751628041267395 eliminated-bits=0.0014751628041267395 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999384895717384 normalized=0.99997949652391283 final-gap=0.99963900168932041 final-incumbent=210.00929300000001 final-bound=0.075813000000000005 first-incumbent-elapsed=28.296102046966553 samples=4`
  `[SearchEngine] New best cost: 210.009 at iteration 0 source=root-heuristic event-epoch=1792684878.5806704 program-deadline=1792684874.7845829`
- First incumbent: 28.296102047 s, cost 210.009; best cost 210.009293.

### I374_c023_n17_confirm1
- In progress: 2026-10-22T12:04:13.830067-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I374_c023_n17_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I374_c023_n17_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I374_c023_n17_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 17 --ignore-checkpoints --run-dir logs/gemma_output_runs/I374_c023_n17_confirm1`
- Completed: 2026-10-22T12:05:49.898100-04:00; exit 0; runner elapsed 96.066 s.
  `[SearchEngine volume] iterations=0 elapsed=14.026891072 total-domain-bits=54225133.898260109 remaining-domain-bits=54225133.896784946 log2-remaining-fraction=-0.0014751628041267395 eliminated-bits=0.0014751628041267395 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999326215107626 normalized=0.99997754050358756 final-gap=0.99963900168932041 final-incumbent=210.00929300000001 final-bound=0.075813000000000005 first-incumbent-elapsed=28.133551120758057 samples=4`
  `[SearchEngine] New best cost: 210.009 at iteration 0 source=root-heuristic event-epoch=1792685099.596122 program-deadline=1792685095.9625843`
- First incumbent: 28.133551121 s, cost 210.009; best cost 210.009293.

### I375_c023_n17_confirm2
- In progress: 2026-10-22T12:05:49.955017-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I375_c023_n17_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I375_c023_n17_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I375_c023_n17_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 17 --ignore-checkpoints --run-dir logs/gemma_output_runs/I375_c023_n17_confirm2`
- Completed: 2026-10-22T12:07:26.221128-04:00; exit 0; runner elapsed 96.265 s.
  `[SearchEngine volume] iterations=0 elapsed=14.113786481 total-domain-bits=54225133.898260109 remaining-domain-bits=54225133.896784946 log2-remaining-fraction=-0.0014751628041267395 eliminated-bits=0.0014751628041267395 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999453804387691 normalized=0.99998179347958971 final-gap=0.99963900168932041 final-incumbent=210.00929300000001 final-bound=0.075813000000000005 first-incumbent-elapsed=28.486985683441162 samples=4`
  `[SearchEngine] New best cost: 210.009 at iteration 0 source=root-heuristic event-epoch=1792685196.0338826 program-deadline=1792685192.0469089`
- First incumbent: 28.486985683 s, cost 210.009; best cost 210.009293.

### I376_c023_n14_regression
- In progress: 2026-10-22T12:07:38.108243-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I376_c023_n14_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I376_c023_n14_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I376_c023_n14_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 14 --ignore-checkpoints --run-dir logs/gemma_output_runs/I376_c023_n14_regression`
- Completed: 2026-10-22T12:09:14.243258-04:00; exit 0; runner elapsed 96.133 s.
  `[SearchEngine volume] iterations=0 elapsed=12.586549735 total-domain-bits=44877722.271791115 remaining-domain-bits=44877722.2705606 log2-remaining-fraction=-0.0012305155396461487 eliminated-bits=0.0012305155396461487 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999405509984275 normalized=0.9999801836661425 final-gap=0.99990966723472008 final-incumbent=648.07049600000005 final-bound=0.058541999999999997 first-incumbent-elapsed=23.418888330459595 samples=4`
  `[SearchEngine] New best cost: 648.07 at iteration 0 source=root-heuristic event-epoch=1792685296.1957517 program-deadline=1792685297.2768776`
- First incumbent: 23.418888330 s, cost 648.07; best cost 648.070496.

### I377_c023_oracle
- In progress: 2026-10-22T12:09:14.292016-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I377_c023_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I377_c023_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I377_c023_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T12:09:18.857273-04:00; exit 0; runner elapsed 4.564 s.
- Oracle verdict: 48/48 cases passed.

### I378_c023_search_gap
- In progress: 2026-10-22T12:09:18.904981-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I378_c023_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I378_c023_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I378_c023_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T12:09:18.927335-04:00; exit 0; runner elapsed 0.021 s.

### I379_c023_native__private_layout
- In progress: 2026-10-22T12:09:18.979255-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I379_c023_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I379_c023_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I379_c023_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T12:09:19.170316-04:00; exit 0; runner elapsed 0.189 s.

### I380_c023_native__write_after_read_cache
- In progress: 2026-10-22T12:09:19.229480-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I380_c023_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I380_c023_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I380_c023_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T12:09:19.442031-04:00; exit 0; runner elapsed 0.211 s.

### I381_c023_native__incremental_propagators
- In progress: 2026-10-22T12:09:19.501026-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I381_c023_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I381_c023_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I381_c023_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T12:09:19.707124-04:00; exit 0; runner elapsed 0.204 s.

### I382_c023_native__start_rank_window
- In progress: 2026-10-22T12:09:19.769624-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I382_c023_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I382_c023_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I382_c023_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T12:09:19.986366-04:00; exit 0; runner elapsed 0.215 s.

### I383_c023_native__selection_reachability
- In progress: 2026-10-22T12:09:20.042250-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I383_c023_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I383_c023_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I383_c023_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T12:09:20.285949-04:00; exit 0; runner elapsed 0.242 s.

### I384_c023_native__relaxation_bound
- In progress: 2026-10-22T12:09:20.361808-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I384_c023_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I384_c023_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I384_c023_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T12:11:08.702393-04:00; exit 0; runner elapsed 108.339 s.

### I385_c023_native__prune_egraph
- In progress: 2026-10-22T12:11:08.761075-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I385_c023_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I385_c023_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I385_c023_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T12:11:08.981238-04:00; exit 0; runner elapsed 0.217 s.

### I386_c024_unselected_events_build
- In progress: 2026-10-22T12:37:45.341387-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I386_c024_unselected_events_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I386_c024_unselected_events_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I386_c024_unselected_events_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T12:40:04.560816-04:00; exit 0; runner elapsed 139.217 s.

### I387_c024_n18
- In progress: 2026-10-22T12:40:04.617860-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I387_c024_n18.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I387_c024_n18.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I387_c024_n18.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 18 --ignore-checkpoints --run-dir logs/gemma_output_runs/I387_c024_n18`
- Completed: 2026-10-22T12:43:48.099325-04:00; exit 1; runner elapsed 223.480 s.
  `[SearchEngine] New best cost: 223.352 at iteration 0 source=root-heuristic event-epoch=1792687428.023803 program-deadline=1792687422.5269284`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 223.352 (no supervisor gap summary); best cost 223.352.

### I388_c024_groups_build
- In progress: 2026-10-22T12:44:53.494653-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I388_c024_groups_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I388_c024_groups_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I388_c024_groups_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T12:47:12.780518-04:00; exit 0; runner elapsed 139.282 s.

### I389_c024_n18
- In progress: 2026-10-22T12:47:12.851733-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I389_c024_n18.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I389_c024_n18.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I389_c024_n18.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 18 --ignore-checkpoints --run-dir logs/gemma_output_runs/I389_c024_n18`
- Completed: 2026-10-22T12:49:11.628343-04:00; exit 0; runner elapsed 118.775 s.
  `[SearchEngine volume] iterations=0 elapsed=13.692577633999999 total-domain-bits=58307093.479124464 remaining-domain-bits=58307093.477595061 log2-remaining-fraction=-0.001529403030872345 eliminated-bits=0.001529403030872345 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999473703858929 normalized=0.99998245679529763 final-gap=0.9997387592102579 final-incumbent=223.34567300000001 final-bound=0.058347000000000003 first-incumbent-elapsed=27.985398292541504 samples=4`
  `[SearchEngine] New best cost: 223.346 at iteration 0 source=root-heuristic event-epoch=1792687697.3920193 program-deadline=1792687693.9066353`
- First incumbent: 27.985398293 s, cost 223.346; best cost 223.345673.

### I390_c024_n18_confirm1
- In progress: 2026-10-22T12:49:19.298438-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I390_c024_n18_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I390_c024_n18_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I390_c024_n18_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 18 --ignore-checkpoints --run-dir logs/gemma_output_runs/I390_c024_n18_confirm1`
- Completed: 2026-10-22T12:50:58.851159-04:00; exit 0; runner elapsed 99.551 s.
  `[SearchEngine volume] iterations=0 elapsed=13.707321319 total-domain-bits=58307093.479124464 remaining-domain-bits=58307093.477595061 log2-remaining-fraction=-0.001529403030872345 eliminated-bits=0.001529403030872345 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999499398086563 normalized=0.99998331326955214 final-gap=0.9997387592102579 final-incumbent=223.34567300000001 final-bound=0.058347000000000003 first-incumbent-elapsed=28.083752870559692 samples=4`
  `[SearchEngine] New best cost: 223.346 at iteration 0 source=root-heuristic event-epoch=1792687805.3174407 program-deadline=1792687801.7337039`
- First incumbent: 28.083752871 s, cost 223.346; best cost 223.345673.

### I391_c024_n18_confirm2
- In progress: 2026-10-22T12:50:58.913299-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I391_c024_n18_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I391_c024_n18_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I391_c024_n18_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 18 --ignore-checkpoints --run-dir logs/gemma_output_runs/I391_c024_n18_confirm2`
- Completed: 2026-10-22T12:52:38.348512-04:00; exit 0; runner elapsed 99.434 s.
  `[SearchEngine volume] iterations=0 elapsed=13.628415220999999 total-domain-bits=58307093.479124464 remaining-domain-bits=58307093.477595061 log2-remaining-fraction=-0.001529403030872345 eliminated-bits=0.001529403030872345 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999448094151571 normalized=0.99998160313838569 final-gap=0.9997387592102579 final-incumbent=223.34567300000001 final-bound=0.058347000000000003 first-incumbent-elapsed=27.887367248535156 samples=4`
  `[SearchEngine] New best cost: 223.346 at iteration 0 source=root-heuristic event-epoch=1792687904.6446633 program-deadline=1792687901.2573094`
- First incumbent: 27.887367249 s, cost 223.346; best cost 223.345673.

### I392_c024_n17_regression
- In progress: 2026-10-22T12:52:44.008495-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I392_c024_n17_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I392_c024_n17_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I392_c024_n17_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 17 --ignore-checkpoints --run-dir logs/gemma_output_runs/I392_c024_n17_regression`
- Completed: 2026-10-22T12:54:34.810361-04:00; exit 0; runner elapsed 110.800 s.
  `[SearchEngine volume] iterations=0 elapsed=12.395381262000001 total-domain-bits=54983325.089566827 remaining-domain-bits=54983325.088112921 log2-remaining-fraction=-0.0014539062976837158 eliminated-bits=0.0014539062976837158 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.998770219586262 normalized=0.99995900731954213 final-gap=0.99968175841915619 final-incumbent=210.01026899999999 final-bound=0.066834000000000005 first-incumbent-elapsed=26.135701656341553 samples=4`
  `[SearchEngine] New best cost: 210.01 at iteration 0 source=root-heuristic event-epoch=1792688024.4675627 program-deadline=1792688022.8318744`
- First incumbent: 26.135701656 s, cost 210.01; best cost 210.010269.

### I393_c024_oracle
- In progress: 2026-10-22T12:54:34.878933-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I393_c024_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I393_c024_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I393_c024_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T12:54:39.479523-04:00; exit 0; runner elapsed 4.599 s.
- Oracle verdict: 48/48 cases passed.

### I394_c024_search_gap
- In progress: 2026-10-22T12:54:39.528041-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I394_c024_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I394_c024_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I394_c024_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T12:54:39.553694-04:00; exit 0; runner elapsed 0.024 s.

### I395_c024_native__private_layout
- In progress: 2026-10-22T12:54:39.605303-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I395_c024_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I395_c024_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I395_c024_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T12:54:39.820469-04:00; exit 0; runner elapsed 0.214 s.

### I396_c024_native__write_after_read_cache
- In progress: 2026-10-22T12:54:39.887036-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I396_c024_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I396_c024_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I396_c024_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T12:54:40.065158-04:00; exit 0; runner elapsed 0.175 s.

### I397_c024_native__incremental_propagators
- In progress: 2026-10-22T12:54:40.121852-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I397_c024_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I397_c024_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I397_c024_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T12:54:40.325477-04:00; exit 0; runner elapsed 0.202 s.

### I398_c024_native__start_rank_window
- In progress: 2026-10-22T12:54:40.385347-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I398_c024_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I398_c024_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I398_c024_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T12:54:40.582550-04:00; exit 0; runner elapsed 0.195 s.

### I399_c024_native__selection_reachability
- In progress: 2026-10-22T12:54:40.645290-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I399_c024_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I399_c024_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I399_c024_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T12:54:40.893702-04:00; exit 0; runner elapsed 0.247 s.

### I400_c024_native__relaxation_bound
- In progress: 2026-10-22T12:54:40.974046-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I400_c024_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I400_c024_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I400_c024_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T12:56:29.006005-04:00; exit 0; runner elapsed 108.030 s.

### I401_c024_native__prune_egraph
- In progress: 2026-10-22T12:56:29.068527-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I401_c024_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I401_c024_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I401_c024_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T12:56:29.271390-04:00; exit 0; runner elapsed 0.200 s.

### I402_c025_root_build
- In progress: 2026-10-22T13:09:17.086135-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I402_c025_root_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I402_c025_root_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I402_c025_root_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T13:11:17.851125-04:00; exit 0; runner elapsed 120.763 s.

### I403_c025_n19
- In progress: 2026-10-22T13:11:17.925399-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I403_c025_n19.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I403_c025_n19.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I403_c025_n19.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I403_c025_n19`
- Completed: 2026-10-22T13:16:01.670448-04:00; exit 0; runner elapsed 283.743 s.
  `[SearchEngine volume] iterations=0 elapsed=13.180596546 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999621241830184 normalized=0.99998737472767274 final-gap=0.99975006428322699 final-incumbent=233.980164 final-bound=0.058479999999999997 first-incumbent-elapsed=28.484577655792236 samples=4`
  `[SearchEngine] New best cost: 233.98 at iteration 0 source=root-heuristic event-epoch=1792689304.9091144 program-deadline=1792689300.9245503`
- First incumbent: 28.484577656 s, cost 233.98; best cost 233.980164.

### I404_c025_alias_build
- In progress: 2026-10-22T13:24:41.655309-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I404_c025_alias_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I404_c025_alias_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I404_c025_alias_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T13:27:01.388254-04:00; exit 0; runner elapsed 139.731 s.

### I405_c025_n19
- In progress: 2026-10-22T13:27:01.449642-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I405_c025_n19.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I405_c025_n19.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I405_c025_n19.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I405_c025_n19`
- Completed: 2026-10-22T13:28:45.354292-04:00; exit 0; runner elapsed 103.901 s.
  `[SearchEngine volume] iterations=0 elapsed=12.947738684000001 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999692829747143 normalized=0.99998976099157144 final-gap=0.99983224515621738 final-incumbent=348.60394300000002 final-bound=0.058479999999999997 first-incumbent-elapsed=28.168933629989624 samples=4`
  `[SearchEngine] New best cost: 348.604 at iteration 0 source=root-heuristic event-epoch=1792690068.5390017 program-deadline=1792690064.8700812`
- First incumbent: 28.168933630 s, cost 348.604; best cost 348.603943.

### I406_c025_primal_build
- In progress: 2026-10-22T13:32:52.306563-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I406_c025_primal_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I406_c025_primal_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I406_c025_primal_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T13:35:12.395377-04:00; exit 0; runner elapsed 140.085 s.

### I407_c025_n19
- In progress: 2026-10-22T13:35:12.473300-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I407_c025_n19.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I407_c025_n19.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I407_c025_n19.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I407_c025_n19`
- Completed: 2026-10-22T13:36:55.893830-04:00; exit 0; runner elapsed 103.417 s.
  `[SearchEngine volume] iterations=0 elapsed=12.444816102000001 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999630326718915 normalized=0.99998767755729712 final-gap=0.99983224515621738 final-incumbent=348.60394300000002 final-bound=0.058479999999999997 first-incumbent-elapsed=27.796348094940186 samples=4`
  `[SearchEngine] New best cost: 348.604 at iteration 0 source=root-heuristic event-epoch=1792690559.4731345 program-deadline=1792690556.1767974`
- First incumbent: 27.796348095 s, cost 348.604; best cost 348.603943.

### I408_c025_n19_confirm1
- In progress: 2026-10-22T13:37:00.643300-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I408_c025_n19_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I408_c025_n19_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I408_c025_n19_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I408_c025_n19_confirm1`
- Completed: 2026-10-22T13:38:43.905757-04:00; exit 0; runner elapsed 103.260 s.
  `[SearchEngine volume] iterations=0 elapsed=12.325420834999999 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999636488442881 normalized=0.99998788294809604 final-gap=0.99983224515621738 final-incumbent=348.60394300000002 final-bound=0.058479999999999997 first-incumbent-elapsed=27.833078622817993 samples=4`
  `[SearchEngine] New best cost: 348.604 at iteration 0 source=root-heuristic event-epoch=1792690667.4254665 program-deadline=1792690664.0924022`
- First incumbent: 27.833078623 s, cost 348.604; best cost 348.603943.

### I409_c025_n19_confirm2
- In progress: 2026-10-22T13:38:43.966296-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I409_c025_n19_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I409_c025_n19_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I409_c025_n19_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I409_c025_n19_confirm2`
- Completed: 2026-10-22T13:40:27.635867-04:00; exit 0; runner elapsed 103.668 s.
  `[SearchEngine volume] iterations=0 elapsed=12.217489655 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999587835784972 normalized=0.9999862611928324 final-gap=0.99983224515621738 final-incumbent=348.60394300000002 final-bound=0.058479999999999997 first-incumbent-elapsed=27.54305624961853 samples=4`
  `[SearchEngine] New best cost: 348.604 at iteration 0 source=root-heuristic event-epoch=1792690770.5946374 program-deadline=1792690767.5515935`
- First incumbent: 27.543056250 s, cost 348.604; best cost 348.603943.

### I410_c025_n18_regression
- In progress: 2026-10-22T13:40:33.393893-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I410_c025_n18_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I410_c025_n18_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I410_c025_n18_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 18 --ignore-checkpoints --run-dir logs/gemma_output_runs/I410_c025_n18_regression`
- Completed: 2026-10-22T13:42:18.660207-04:00; exit 0; runner elapsed 105.264 s.
  `[SearchEngine volume] iterations=0 elapsed=11.769040627000001 total-domain-bits=58307093.479124464 remaining-domain-bits=58307093.477595061 log2-remaining-fraction=-0.001529403030872345 eliminated-bits=0.001529403030872345 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999443173211322 normalized=0.99998143910704407 final-gap=0.99985898101747739 final-incumbent=413.75280800000002 final-bound=0.058347000000000003 first-incumbent-elapsed=26.051405429840088 samples=4`
  `[SearchEngine] New best cost: 413.753 at iteration 0 source=root-heuristic event-epoch=1792690877.8203321 program-deadline=1792690876.2689407`
- First incumbent: 26.051405430 s, cost 413.753; best cost 413.752808.

### I411_c025_oracle
- In progress: 2026-10-22T13:42:18.721054-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I411_c025_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I411_c025_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I411_c025_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T13:42:23.481844-04:00; exit 0; runner elapsed 4.759 s.
- Oracle verdict: 48/48 cases passed.

### I412_c025_search_gap
- In progress: 2026-10-22T13:42:23.538989-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I412_c025_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I412_c025_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I412_c025_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T13:42:23.565708-04:00; exit 0; runner elapsed 0.025 s.

### I413_c025_native__private_layout
- In progress: 2026-10-22T13:42:23.624511-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I413_c025_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I413_c025_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I413_c025_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T13:42:23.844443-04:00; exit 0; runner elapsed 0.218 s.

### I414_c025_native__write_after_read_cache
- In progress: 2026-10-22T13:42:23.913688-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I414_c025_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I414_c025_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I414_c025_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T13:42:24.151286-04:00; exit 0; runner elapsed 0.236 s.

### I415_c025_native__incremental_propagators
- In progress: 2026-10-22T13:42:24.221153-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I415_c025_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I415_c025_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I415_c025_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T13:42:24.417400-04:00; exit 0; runner elapsed 0.194 s.

### I416_c025_native__start_rank_window
- In progress: 2026-10-22T13:42:24.476974-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I416_c025_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I416_c025_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I416_c025_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T13:42:24.665630-04:00; exit 0; runner elapsed 0.187 s.

### I417_c025_native__selection_reachability
- In progress: 2026-10-22T13:42:24.729033-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I417_c025_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I417_c025_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I417_c025_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T13:42:24.984615-04:00; exit 0; runner elapsed 0.254 s.

### I418_c025_native__relaxation_bound
- In progress: 2026-10-22T13:42:25.070747-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I418_c025_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I418_c025_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I418_c025_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T13:44:14.293814-04:00; exit 0; runner elapsed 109.221 s.

### I419_c025_native__prune_egraph
- In progress: 2026-10-22T13:44:14.361322-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I419_c025_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I419_c025_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I419_c025_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T13:44:14.582724-04:00; exit 0; runner elapsed 0.219 s.

### I420_c026_alloc_build
- In progress: 2026-10-22T14:07:05.567146-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I420_c026_alloc_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I420_c026_alloc_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I420_c026_alloc_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T14:07:05.830364-04:00; exit 1; runner elapsed 0.261 s.

### I421_c026_n20
- In progress: 2026-10-22T14:07:05.902408-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I421_c026_n20.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I421_c026_n20.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I421_c026_n20.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 20 --ignore-checkpoints --run-dir logs/gemma_output_runs/I421_c026_n20`
- Completed: 2026-10-22T14:12:17.004699-04:00; exit 0; runner elapsed 311.099 s.
  `[SearchEngine volume] iterations=0 elapsed=12.757365679999999 total-domain-bits=64861245.8245904 remaining-domain-bits=64861245.822908171 log2-remaining-fraction=-0.0016822293400764465 eliminated-bits=0.0016822293400764465 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999572375918071 normalized=0.99998574586393574 final-gap=0.99986807610742612 final-incumbent=443.285889 final-bound=0.058479999999999997 first-incumbent-elapsed=26.758554697036743 samples=4`
  `[SearchEngine] New best cost: 443.286 at iteration 0 source=root-heuristic event-epoch=1792692670.36554 program-deadline=1792692668.1069977`
- First incumbent: 26.758554697 s, cost 443.286; best cost 443.285889.

### I422_c026_alloc_build
- In progress: 2026-10-22T14:12:52.793081-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I422_c026_alloc_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I422_c026_alloc_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I422_c026_alloc_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T14:19:06.745716-04:00; exit 0; runner elapsed 373.951 s.

### I4223_c026_n20
- In progress: 2026-10-22T14:19:13.154211-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I4223_c026_n20.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I4223_c026_n20.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I4223_c026_n20.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 20 --ignore-checkpoints --run-dir logs/gemma_output_runs/I4223_c026_n20`
- Completed: 2026-10-22T14:21:02.945719-04:00; exit 0; runner elapsed 109.790 s.
  `[SearchEngine volume] iterations=0 elapsed=12.90919008 total-domain-bits=64861245.8245904 remaining-domain-bits=64861245.822908171 log2-remaining-fraction=-0.0016822293400764465 eliminated-bits=0.0016822293400764465 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999556456462653 normalized=0.99998521521542172 final-gap=0.99986807610742612 final-incumbent=443.285889 final-bound=0.058479999999999997 first-incumbent-elapsed=26.637883186340332 samples=4`
  `[SearchEngine] New best cost: 443.286 at iteration 0 source=root-heuristic event-epoch=1792693196.0204785 program-deadline=1792693193.8826089`
- First incumbent: 26.637883186 s, cost 443.286; best cost 443.285889.

### I4224_c026_n21
- In progress: 2026-10-22T14:21:03.018173-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I4224_c026_n21.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I4224_c026_n21.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I4224_c026_n21.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 21 --ignore-checkpoints --run-dir logs/gemma_output_runs/I4224_c026_n21`
- Completed: 2026-10-22T14:26:12.505095-04:00; exit 0; runner elapsed 309.485 s.
  `[SearchEngine volume] iterations=0 elapsed=13.243372219999999 total-domain-bits=68080310.160167813 remaining-domain-bits=68080310.158407524 log2-remaining-fraction=-0.0017602890729904175 eliminated-bits=0.0017602890729904175 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999681539477514 normalized=0.99998938464925047 final-gap=0.99986145311733476 final-incumbent=420.52191199999999 final-bound=0.058262000000000001 first-incumbent-elapsed=27.701424121856689 samples=4`
  `[SearchEngine] New best cost: 420.522 at iteration 0 source=root-heuristic event-epoch=1792693502.6166954 program-deadline=1792693499.4152842`
- First incumbent: 27.701424122 s, cost 420.522; best cost 420.521912.

### I425_c026_n21_confirm1
- In progress: 2026-10-22T14:26:19.369880-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I425_c026_n21_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I425_c026_n21_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I425_c026_n21_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 21 --ignore-checkpoints --run-dir logs/gemma_output_runs/I425_c026_n21_confirm1`
- Completed: 2026-10-22T14:28:14.194142-04:00; exit 0; runner elapsed 114.822 s.
  `[SearchEngine volume] iterations=0 elapsed=13.198591566999999 total-domain-bits=68080310.160167813 remaining-domain-bits=68080310.158407524 log2-remaining-fraction=-0.0017602890729904175 eliminated-bits=0.0017602890729904175 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999664305514326 normalized=0.99998881018381092 final-gap=0.99986145311733476 final-incumbent=420.52191199999999 final-bound=0.058262000000000001 first-incumbent-elapsed=27.577033281326294 samples=4`
  `[SearchEngine] New best cost: 420.522 at iteration 0 source=root-heuristic event-epoch=1792693623.9919729 program-deadline=1792693620.9149523`
- First incumbent: 27.577033281 s, cost 420.522; best cost 420.521912.

### I426_c026_n21_confirm2
- In progress: 2026-10-22T14:28:14.260340-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I426_c026_n21_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I426_c026_n21_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I426_c026_n21_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 21 --ignore-checkpoints --run-dir logs/gemma_output_runs/I426_c026_n21_confirm2`
- Completed: 2026-10-22T14:30:08.193669-04:00; exit 0; runner elapsed 113.931 s.
  `[SearchEngine volume] iterations=0 elapsed=13.32598978 total-domain-bits=68080310.160167813 remaining-domain-bits=68080310.158407524 log2-remaining-fraction=-0.0017602890729904175 eliminated-bits=0.0017602890729904175 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.99966565425008 normalized=0.99998885514166935 final-gap=0.99986145311733476 final-incumbent=420.52191199999999 final-bound=0.058262000000000001 first-incumbent-elapsed=27.58676815032959 samples=4`
  `[SearchEngine] New best cost: 420.522 at iteration 0 source=root-heuristic event-epoch=1792693738.993449 program-deadline=1792693735.9067042`
- First incumbent: 27.586768150 s, cost 420.522; best cost 420.521912.

### I427_c026_n19_regression
- In progress: 2026-10-22T14:30:13.814211-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I427_c026_n19_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I427_c026_n19_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I427_c026_n19_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 19 --ignore-checkpoints --run-dir logs/gemma_output_runs/I427_c026_n19_regression`
- Completed: 2026-10-22T14:31:52.139218-04:00; exit 0; runner elapsed 98.323 s.
  `[SearchEngine volume] iterations=0 elapsed=12.188216567 total-domain-bits=61571166.049465388 remaining-domain-bits=61571166.04785917 log2-remaining-fraction=-0.0016062185168266296 eliminated-bits=0.0016062185168266296 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999238517122816 normalized=0.9999746172374272 final-gap=0.99983224515621738 final-incumbent=348.60394300000002 final-bound=0.058479999999999997 first-incumbent-elapsed=25.460739850997925 samples=4`
  `[SearchEngine] New best cost: 348.604 at iteration 0 source=root-heuristic event-epoch=1792693855.2307968 program-deadline=1792693854.2700689`
- First incumbent: 25.460739851 s, cost 348.604; best cost 348.603943.

### I428_c026_oracle
- In progress: 2026-10-22T14:31:52.211944-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I428_c026_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I428_c026_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I428_c026_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T14:31:56.760138-04:00; exit 0; runner elapsed 4.545 s.
- Oracle verdict: 48/48 cases passed.

### I429_c026_search_gap
- In progress: 2026-10-22T14:31:56.821437-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I429_c026_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I429_c026_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I429_c026_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T14:31:56.847047-04:00; exit 0; runner elapsed 0.024 s.

### I430_c026_native__private_layout
- In progress: 2026-10-22T14:31:56.910527-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I430_c026_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I430_c026_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I430_c026_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T14:31:57.124508-04:00; exit 0; runner elapsed 0.212 s.

### I431_c026_native__write_after_read_cache
- In progress: 2026-10-22T14:31:57.196416-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I431_c026_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I431_c026_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I431_c026_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T14:31:57.406523-04:00; exit 0; runner elapsed 0.208 s.

### I432_c026_native__incremental_propagators
- In progress: 2026-10-22T14:31:57.479008-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I432_c026_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I432_c026_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I432_c026_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T14:31:57.716383-04:00; exit 0; runner elapsed 0.236 s.

### I433_c026_native__start_rank_window
- In progress: 2026-10-22T14:31:57.795211-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I433_c026_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I433_c026_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I433_c026_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T14:31:58.017203-04:00; exit 0; runner elapsed 0.220 s.

### I434_c026_native__selection_reachability
- In progress: 2026-10-22T14:31:58.089401-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I434_c026_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I434_c026_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I434_c026_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T14:31:58.348139-04:00; exit 0; runner elapsed 0.257 s.

### I435_c026_native__relaxation_bound
- In progress: 2026-10-22T14:31:58.438857-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I435_c026_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I435_c026_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I435_c026_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T14:33:46.819216-04:00; exit 0; runner elapsed 108.378 s.

### I436_c026_native__prune_egraph
- In progress: 2026-10-22T14:33:46.885343-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I436_c026_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I436_c026_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I436_c026_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T14:33:47.106694-04:00; exit 0; runner elapsed 0.219 s.

### I437_c027_eclass_intern_build
- In progress: 2026-10-22T14:44:12.774313-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I437_c027_eclass_intern_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I437_c027_eclass_intern_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I437_c027_eclass_intern_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T14:46:14.199703-04:00; exit 0; runner elapsed 121.424 s.

### I438_c027_n22
- In progress: 2026-10-22T14:46:18.526019-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I438_c027_n22.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I438_c027_n22.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I438_c027_n22.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 22 --ignore-checkpoints --run-dir logs/gemma_output_runs/I438_c027_n22`
- Completed: 2026-10-22T14:51:38.672647-04:00; exit 0; runner elapsed 320.145 s.
  `[SearchEngine volume] iterations=0 elapsed=13.559483804999999 total-domain-bits=71394640.980641842 remaining-domain-bits=71394640.978805855 log2-remaining-fraction=-0.0018359869718551636 eliminated-bits=0.0018359869718551636 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999650767104267 normalized=0.99998835890347559 final-gap=0.99986917760413119 final-incumbent=446.98004200000003 final-bound=0.058474999999999999 first-incumbent-elapsed=27.330480813980103 samples=4`
  `[SearchEngine] New best cost: 446.98 at iteration 0 source=root-heuristic event-epoch=1792695025.4142013 program-deadline=1792695022.5837352`
- First incumbent: 27.330480814 s, cost 446.98; best cost 446.980042.

### I439_c027_n23
- In progress: 2026-10-22T14:51:38.741815-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I439_c027_n23.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I439_c027_n23.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I439_c027_n23.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 23 --ignore-checkpoints --run-dir logs/gemma_output_runs/I439_c027_n23`
- Completed: 2026-10-22T14:56:01.581182-04:00; exit 1; runner elapsed 262.838 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I440_c027_n22_confirm1
- In progress: 2026-10-22T14:56:11.856907-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I440_c027_n22_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I440_c027_n22_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I440_c027_n22_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 22 --ignore-checkpoints --run-dir logs/gemma_output_runs/I440_c027_n22_confirm1`
- Completed: 2026-10-22T14:58:10.181038-04:00; exit 0; runner elapsed 118.322 s.
  `[SearchEngine volume] iterations=0 elapsed=13.659087061999999 total-domain-bits=71394640.980641842 remaining-domain-bits=71394640.978805855 log2-remaining-fraction=-0.0018359869718551636 eliminated-bits=0.0018359869718551636 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999706198062867 normalized=0.99999020660209559 final-gap=0.99986917760413119 final-incumbent=446.98004200000003 final-bound=0.058474999999999999 first-incumbent-elapsed=27.754192352294922 samples=4`
  `[SearchEngine] New best cost: 446.98 at iteration 0 source=root-heuristic event-epoch=1792695416.4896073 program-deadline=1792695413.2354319`
- First incumbent: 27.754192352 s, cost 446.98; best cost 446.980042.

### I441_c027_n22_confirm2
- In progress: 2026-10-22T14:58:10.255217-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I441_c027_n22_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I441_c027_n22_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I441_c027_n22_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 22 --ignore-checkpoints --run-dir logs/gemma_output_runs/I441_c027_n22_confirm2`
- Completed: 2026-10-22T15:00:09.581539-04:00; exit 0; runner elapsed 119.325 s.
  `[SearchEngine volume] iterations=0 elapsed=13.862539112 total-domain-bits=71394640.980641842 remaining-domain-bits=71394640.978805855 log2-remaining-fraction=-0.0018359869718551636 eliminated-bits=0.0018359869718551636 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999710892637051 normalized=0.99999036308790168 final-gap=0.99986917760413119 final-incumbent=446.98004200000003 final-bound=0.058474999999999999 first-incumbent-elapsed=27.790077447891235 samples=4`
  `[SearchEngine] New best cost: 446.98 at iteration 0 source=root-heuristic event-epoch=1792695534.6463418 program-deadline=1792695531.3562775`
- First incumbent: 27.790077448 s, cost 446.98; best cost 446.980042.

### I442_c027_n21_regression
- In progress: 2026-10-22T15:00:09.649620-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I442_c027_n21_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I442_c027_n21_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I442_c027_n21_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 21 --ignore-checkpoints --run-dir logs/gemma_output_runs/I442_c027_n21_regression`
- Completed: 2026-10-22T15:02:02.242063-04:00; exit 0; runner elapsed 112.591 s.
  `[SearchEngine volume] iterations=0 elapsed=12.992875055000001 total-domain-bits=68080310.160167813 remaining-domain-bits=68080310.158407524 log2-remaining-fraction=-0.0017602890729904175 eliminated-bits=0.0017602890729904175 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=29.999507423508692 normalized=0.99998358078362304 final-gap=0.99986145311733476 final-incumbent=420.52191199999999 final-bound=0.058262000000000001 first-incumbent-elapsed=26.444694519042969 samples=4`
  `[SearchEngine] New best cost: 420.522 at iteration 0 source=root-heuristic event-epoch=1792695652.2758849 program-deadline=1792695650.3312025`
- First incumbent: 26.444694519 s, cost 420.522; best cost 420.521912.

### I443_c027_oracle
- In progress: 2026-10-22T15:02:02.307536-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I443_c027_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I443_c027_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I443_c027_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T15:02:06.927063-04:00; exit 0; runner elapsed 4.618 s.
- Oracle verdict: 48/48 cases passed.

### I444_c027_search_gap
- In progress: 2026-10-22T15:02:06.982588-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I444_c027_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I444_c027_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I444_c027_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T15:02:07.008633-04:00; exit 0; runner elapsed 0.025 s.

### I445_c027_native__private_layout
- In progress: 2026-10-22T15:02:07.064666-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I445_c027_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I445_c027_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I445_c027_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T15:02:07.252354-04:00; exit 0; runner elapsed 0.186 s.

### I446_c027_native__write_after_read_cache
- In progress: 2026-10-22T15:02:07.319223-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I446_c027_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I446_c027_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I446_c027_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T15:02:07.530732-04:00; exit 0; runner elapsed 0.210 s.

### I447_c027_native__incremental_propagators
- In progress: 2026-10-22T15:02:07.601834-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I447_c027_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I447_c027_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I447_c027_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T15:02:07.814651-04:00; exit 0; runner elapsed 0.211 s.

### I448_c027_native__start_rank_window
- In progress: 2026-10-22T15:02:07.887210-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I448_c027_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I448_c027_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I448_c027_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T15:02:08.085675-04:00; exit 0; runner elapsed 0.195 s.

### I449_c027_native__selection_reachability
- In progress: 2026-10-22T15:02:08.153213-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I449_c027_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I449_c027_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I449_c027_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T15:02:08.425283-04:00; exit 0; runner elapsed 0.270 s.

### I450_c027_native__relaxation_bound
- In progress: 2026-10-22T15:02:08.519207-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I450_c027_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I450_c027_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I450_c027_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T15:03:56.708196-04:00; exit 0; runner elapsed 108.186 s.

### I451_c027_native__prune_egraph
- In progress: 2026-10-22T15:03:56.778462-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I451_c027_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I451_c027_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I451_c027_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T15:03:57.006291-04:00; exit 0; runner elapsed 0.224 s.

### I452_c028_restricted_build
- In progress: 2026-10-22T15:34:49.755390-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I452_c028_restricted_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I452_c028_restricted_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I452_c028_restricted_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T15:36:51.051435-04:00; exit 0; runner elapsed 121.294 s.

### I453_c028_n26
- In progress: 2026-10-22T15:36:51.139270-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I453_c028_n26.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I453_c028_n26.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I453_c028_n26.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 26 --ignore-checkpoints --run-dir logs/gemma_output_runs/I453_c028_n26`
- Completed: 2026-10-22T15:44:05.469437-04:00; exit 0; runner elapsed 434.327 s.
  `[SearchEngine volume] iterations=0 elapsed=6.6443493589999996 total-domain-bits=84654449.764165878 remaining-domain-bits=84654449.764165878 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=652.28533900000002 final-bound=0 first-incumbent-elapsed=23.364295482635498 samples=3`
  `[SearchEngine] New best cost: 652.285 at iteration 0 source=root-heuristic-restricted event-epoch=1792698133.0241106 program-deadline=1792698134.1598272`
- First incumbent: 23.364295483 s, cost 652.285; best cost 652.285339.

### I454_c028_oracle
- In progress: 2026-10-22T15:44:11.326413-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I454_c028_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I454_c028_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I454_c028_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T15:44:16.174007-04:00; exit 0; runner elapsed 4.844 s.
- Oracle verdict: 48/48 cases passed.

### I455_c028_search_gap
- In progress: 2026-10-22T15:44:16.234350-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I455_c028_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I455_c028_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I455_c028_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T15:44:16.257024-04:00; exit 0; runner elapsed 0.021 s.

### I456_c028_n30
- In progress: 2026-10-22T15:44:21.786810-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I456_c028_n30.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I456_c028_n30.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I456_c028_n30.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 30 --ignore-checkpoints --run-dir logs/gemma_output_runs/I456_c028_n30`
- Completed: 2026-10-22T15:51:23.190498-04:00; exit 0; runner elapsed 421.402 s.
  `[SearchEngine volume] iterations=0 elapsed=8.9221951510000004 total-domain-bits=97912699.603301644 remaining-domain-bits=97912699.603301644 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=45.403739999999999 final-bound=0 first-incumbent-elapsed=28.108504772186279 samples=3`
  `[SearchEngine] New best cost: 45.4037 at iteration 0 source=root-heuristic-restricted event-epoch=1792698545.1737165 program-deadline=1792698541.5652258`
- First incumbent: 28.108504772 s, cost 45.4037; best cost 45.40374.

### I457_c028_batched_build
- In progress: 2026-10-22T15:56:22.038138-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I457_c028_batched_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I457_c028_batched_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I457_c028_batched_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T15:58:23.971291-04:00; exit 0; runner elapsed 121.931 s.

### I458_c028_n32
- In progress: 2026-10-22T15:58:24.065423-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I458_c028_n32.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I458_c028_n32.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I458_c028_n32.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 32 --ignore-checkpoints --run-dir logs/gemma_output_runs/I458_c028_n32`
- Completed: 2026-10-22T16:05:58.114260-04:00; exit 0; runner elapsed 454.047 s.
  `[SearchEngine volume] iterations=0 elapsed=5.7862407779999998 total-domain-bits=104592057.17486978 remaining-domain-bits=104592057.17486978 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=43.419162999999998 final-bound=0 first-incumbent-elapsed=26.034607410430908 samples=3`
  `[SearchEngine] New best cost: 43.4192 at iteration 0 source=root-heuristic-restricted event-epoch=1792699390.6462843 program-deadline=1792699389.1116893`
- First incumbent: 26.034607410 s, cost 43.4192; best cost 43.419163.

### I459_c028_n36
- In progress: 2026-10-22T16:06:05.410333-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I459_c028_n36.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I459_c028_n36.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I459_c028_n36.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 36 --ignore-checkpoints --run-dir logs/gemma_output_runs/I459_c028_n36`
- Completed: 2026-10-22T16:11:57.507069-04:00; exit 1; runner elapsed 352.095 s.
  `[SearchEngine volume] iterations=0 elapsed=6.4346364109999996 total-domain-bits=117856628.49071015 remaining-domain-bits=117856628.49071015 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 687.363 at iteration 0 source=root-heuristic-restricted event-epoch=1792699916.6473563 program-deadline=1792699911.8964279`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 687.363 (no supervisor gap summary); best cost 687.363.

### I460_c028_n34
- In progress: 2026-10-22T16:12:30.574763-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I460_c028_n34.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I460_c028_n34.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I460_c028_n34.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 34 --ignore-checkpoints --run-dir logs/gemma_output_runs/I460_c028_n34`
- Completed: 2026-10-22T16:19:11.877937-04:00; exit 0; runner elapsed 401.300 s.
  `[SearchEngine volume] iterations=0 elapsed=6.15583773 total-domain-bits=111169665.27186811 remaining-domain-bits=111169665.27186811 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=605.77691700000003 final-bound=0 first-incumbent-elapsed=27.794993877410889 samples=3`
  `[SearchEngine] New best cost: 605.777 at iteration 0 source=root-heuristic-restricted event-epoch=1792700220.5328717 program-deadline=1792700217.2378886`
- First incumbent: 27.794993877 s, cost 605.777; best cost 605.776917.

### I461_c028_n34_confirm1
- In progress: 2026-10-22T16:21:39.377682-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I461_c028_n34_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I461_c028_n34_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I461_c028_n34_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 34 --ignore-checkpoints --run-dir logs/gemma_output_runs/I461_c028_n34_confirm1`
- Completed: 2026-10-22T16:24:43.909042-04:00; exit 0; runner elapsed 184.530 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0210892400000002 total-domain-bits=111169665.27186811 remaining-domain-bits=111169665.27186811 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=605.77691700000003 final-bound=0 first-incumbent-elapsed=27.899348258972168 samples=3`
  `[SearchEngine] New best cost: 605.777 at iteration 0 source=root-heuristic-restricted event-epoch=1792700552.6520379 program-deadline=1792700549.2527015`
- First incumbent: 27.899348259 s, cost 605.777; best cost 605.776917.

### I462_c028_n34_confirm2
- In progress: 2026-10-22T16:24:43.983166-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I462_c028_n34_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I462_c028_n34_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I462_c028_n34_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 34 --ignore-checkpoints --run-dir logs/gemma_output_runs/I462_c028_n34_confirm2`
- Completed: 2026-10-22T16:27:50.507997-04:00; exit 0; runner elapsed 186.523 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1216220750000003 total-domain-bits=111169665.27186811 remaining-domain-bits=111169665.27186811 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=605.77691700000003 final-bound=0 first-incumbent-elapsed=28.038949966430664 samples=3`
  `[SearchEngine] New best cost: 605.777 at iteration 0 source=root-heuristic-restricted event-epoch=1792700737.2758515 program-deadline=1792700733.736913`
- First incumbent: 28.038949966 s, cost 605.777; best cost 605.776917.

### I463_c028_n22_regression
- In progress: 2026-10-22T16:27:57.321621-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I463_c028_n22_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I463_c028_n22_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I463_c028_n22_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 22 --ignore-checkpoints --run-dir logs/gemma_output_runs/I463_c028_n22_regression`
- Completed: 2026-10-22T16:29:48.656550-04:00; exit 0; runner elapsed 111.333 s.
  `[SearchEngine volume] iterations=0 elapsed=3.797860386 total-domain-bits=71394640.980641842 remaining-domain-bits=71394640.980641842 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=446.98004200000003 final-bound=0 first-incumbent-elapsed=19.329610347747803 samples=3`
  `[SearchEngine] New best cost: 446.98 at iteration 0 source=root-heuristic-restricted event-epoch=1792700915.3215468 program-deadline=1792700920.4919472`
- First incumbent: 19.329610348 s, cost 446.98; best cost 446.980042.

### I464_c028_oracle
- In progress: 2026-10-22T16:29:48.722034-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I464_c028_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I464_c028_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I464_c028_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T16:29:53.423600-04:00; exit 0; runner elapsed 4.700 s.
- Oracle verdict: 48/48 cases passed.

### I465_c028_search_gap
- In progress: 2026-10-22T16:29:53.490138-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I465_c028_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I465_c028_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I465_c028_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T16:29:53.514302-04:00; exit 0; runner elapsed 0.023 s.

### I466_c028_native__private_layout
- In progress: 2026-10-22T16:29:53.581392-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I466_c028_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I466_c028_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I466_c028_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T16:29:53.777650-04:00; exit 0; runner elapsed 0.195 s.

### I467_c028_native__write_after_read_cache
- In progress: 2026-10-22T16:29:53.848674-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I467_c028_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I467_c028_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I467_c028_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T16:29:54.047683-04:00; exit 0; runner elapsed 0.197 s.

### I468_c028_native__incremental_propagators
- In progress: 2026-10-22T16:29:54.121543-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I468_c028_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I468_c028_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I468_c028_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T16:29:54.319335-04:00; exit 0; runner elapsed 0.195 s.

### I469_c028_native__start_rank_window
- In progress: 2026-10-22T16:29:54.395201-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I469_c028_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I469_c028_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I469_c028_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T16:29:54.597146-04:00; exit 0; runner elapsed 0.199 s.

### I470_c028_native__selection_reachability
- In progress: 2026-10-22T16:29:54.673571-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I470_c028_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I470_c028_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I470_c028_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T16:29:54.894906-04:00; exit 0; runner elapsed 0.220 s.

### I471_c028_native__relaxation_bound
- In progress: 2026-10-22T16:29:54.993710-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I471_c028_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I471_c028_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I471_c028_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T16:31:42.654098-04:00; exit 0; runner elapsed 107.659 s.

### I472_c028_native__prune_egraph
- In progress: 2026-10-22T16:31:42.726942-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I472_c028_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I472_c028_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I472_c028_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T16:31:42.942274-04:00; exit 0; runner elapsed 0.212 s.

### I473_c029_records_build
- In progress: 2026-10-22T16:36:57.661669-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I473_c029_records_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I473_c029_records_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I473_c029_records_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T16:38:59.345521-04:00; exit 0; runner elapsed 121.681 s.

### I474_c029_n36
- In progress: 2026-10-22T16:38:59.436398-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I474_c029_n36.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I474_c029_n36.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I474_c029_n36.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 36 --ignore-checkpoints --run-dir logs/gemma_output_runs/I474_c029_n36`
- Completed: 2026-10-22T16:42:11.561065-04:00; exit 0; runner elapsed 192.123 s.
  `[SearchEngine volume] iterations=0 elapsed=6.3884737400000002 total-domain-bits=117856628.49071015 remaining-domain-bits=117856628.49071015 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=687.36328100000003 final-bound=0 first-incumbent-elapsed=28.063624620437622 samples=3`
  `[SearchEngine] New best cost: 687.363 at iteration 0 source=root-heuristic-restricted event-epoch=1792701592.6891046 program-deadline=1792701589.125493`
- First incumbent: 28.063624620 s, cost 687.363; best cost 687.363281.

### I475_c029_share_build
- In progress: 2026-10-22T16:50:45.471096-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I475_c029_share_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I475_c029_share_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I475_c029_share_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T16:52:47.541590-04:00; exit 0; runner elapsed 122.069 s.

### I476_c029_n37
- In progress: 2026-10-22T16:52:47.631262-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I476_c029_n37.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I476_c029_n37.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I476_c029_n37.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 37 --ignore-checkpoints --run-dir logs/gemma_output_runs/I476_c029_n37`
- Completed: 2026-10-22T17:01:11.510206-04:00; exit 0; runner elapsed 503.877 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1553251160000002 total-domain-bits=121113513.43611048 remaining-domain-bits=121113513.43611048 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=693.13391100000001 final-bound=0 first-incumbent-elapsed=27.549184083938599 samples=3`
  `[SearchEngine] New best cost: 693.134 at iteration 0 source=root-heuristic-restricted event-epoch=1792702700.3265953 program-deadline=1792702697.2774248`
- First incumbent: 27.549184084 s, cost 693.134; best cost 693.133911.

### I477_c029_n38
- In progress: 2026-10-22T17:02:13.962692-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I477_c029_n38.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I477_c029_n38.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I477_c029_n38.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 38 --ignore-checkpoints --run-dir logs/gemma_output_runs/I477_c029_n38`
- Completed: 2026-10-22T17:07:38.746259-04:00; exit 1; runner elapsed 324.782 s.
- Missing volume or gap report: FAILED evaluation.
- First incumbent: none within budget (+infinity); best cost None.

### I478_c029_n37_confirm1
- In progress: 2026-10-22T17:07:47.147308-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I478_c029_n37_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I478_c029_n37_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I478_c029_n37_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 37 --ignore-checkpoints --run-dir logs/gemma_output_runs/I478_c029_n37_confirm1`
- Completed: 2026-10-22T17:11:22.673578-04:00; exit 0; runner elapsed 215.524 s.
  `[SearchEngine volume] iterations=0 elapsed=5.26036883 total-domain-bits=121113513.43611048 remaining-domain-bits=121113513.43611048 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=693.13391100000001 final-bound=0 first-incumbent-elapsed=27.768410205841064 samples=3`
  `[SearchEngine] New best cost: 693.134 at iteration 0 source=root-heuristic-restricted event-epoch=1792703321.6087017 program-deadline=1792703318.340302`
- First incumbent: 27.768410206 s, cost 693.134; best cost 693.133911.

### I479_c029_n37_confirm2
- In progress: 2026-10-22T17:11:22.741464-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I479_c029_n37_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I479_c029_n37_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I479_c029_n37_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 37 --ignore-checkpoints --run-dir logs/gemma_output_runs/I479_c029_n37_confirm2`
- Completed: 2026-10-22T17:15:02.239747-04:00; exit 0; runner elapsed 219.497 s.
  `[SearchEngine volume] iterations=0 elapsed=5.2198928369999997 total-domain-bits=121113513.43611048 remaining-domain-bits=121113513.43611048 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=693.13391100000001 final-bound=0 first-incumbent-elapsed=27.896376132965088 samples=3`
  `[SearchEngine] New best cost: 693.134 at iteration 0 source=root-heuristic-restricted event-epoch=1792703537.0910366 program-deadline=1792703533.6946723`
- First incumbent: 27.896376133 s, cost 693.134; best cost 693.133911.

### I480_c029_n34_regression
- In progress: 2026-10-22T17:15:11.633145-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I480_c029_n34_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I480_c029_n34_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I480_c029_n34_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 34 --ignore-checkpoints --run-dir logs/gemma_output_runs/I480_c029_n34_regression`
- Completed: 2026-10-22T17:18:17.320685-04:00; exit 0; runner elapsed 185.685 s.
  `[SearchEngine volume] iterations=0 elapsed=4.966105175 total-domain-bits=111169665.27186811 remaining-domain-bits=111169665.27186811 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=605.77691700000003 final-bound=0 first-incumbent-elapsed=25.970638275146484 samples=3`
  `[SearchEngine] New best cost: 605.777 at iteration 0 source=root-heuristic-restricted event-epoch=1792703762.5889542 program-deadline=1792703761.1183257`
- First incumbent: 25.970638275 s, cost 605.777; best cost 605.776917.

### I481_c029_oracle
- In progress: 2026-10-22T17:18:17.389677-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I481_c029_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I481_c029_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I481_c029_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T17:18:22.036659-04:00; exit 0; runner elapsed 4.645 s.
- Oracle verdict: 48/48 cases passed.

### I482_c029_search_gap
- In progress: 2026-10-22T17:18:22.097863-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I482_c029_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I482_c029_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I482_c029_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T17:18:22.124816-04:00; exit 0; runner elapsed 0.025 s.

### I483_c029_native__private_layout
- In progress: 2026-10-22T17:18:22.192383-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I483_c029_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I483_c029_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I483_c029_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T17:18:22.382069-04:00; exit 0; runner elapsed 0.188 s.

### I484_c029_native__write_after_read_cache
- In progress: 2026-10-22T17:18:22.450771-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I484_c029_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I484_c029_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I484_c029_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T17:18:22.642190-04:00; exit 0; runner elapsed 0.189 s.

### I485_c029_native__incremental_propagators
- In progress: 2026-10-22T17:18:22.716356-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I485_c029_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I485_c029_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I485_c029_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T17:18:22.918283-04:00; exit 0; runner elapsed 0.200 s.

### I486_c029_native__start_rank_window
- In progress: 2026-10-22T17:18:22.995313-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I486_c029_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I486_c029_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I486_c029_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T17:18:23.206177-04:00; exit 0; runner elapsed 0.209 s.

### I487_c029_native__selection_reachability
- In progress: 2026-10-22T17:18:23.282400-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I487_c029_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I487_c029_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I487_c029_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T17:18:23.555209-04:00; exit 0; runner elapsed 0.271 s.

### I488_c029_native__relaxation_bound
- In progress: 2026-10-22T17:18:23.641579-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I488_c029_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I488_c029_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I488_c029_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T17:20:11.865581-04:00; exit 0; runner elapsed 108.222 s.

### I489_c029_native__prune_egraph
- In progress: 2026-10-22T17:20:11.938022-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I489_c029_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I489_c029_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I489_c029_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T17:20:12.158164-04:00; exit 0; runner elapsed 0.219 s.

### I490_c030_parents_build
- In progress: 2026-10-22T17:30:19.063721-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I490_c030_parents_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I490_c030_parents_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I490_c030_parents_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T17:32:21.769910-04:00; exit 0; runner elapsed 122.703 s.

### I491_c030_n38
- In progress: 2026-10-22T17:32:21.863262-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I491_c030_n38.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I491_c030_n38.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I491_c030_n38.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 38 --ignore-checkpoints --run-dir logs/gemma_output_runs/I491_c030_n38`
- Completed: 2026-10-22T17:33:18.153344-04:00; exit 1; runner elapsed 56.288 s.
  `[SearchEngine volume] iterations=0 elapsed=6.6906662389999996 total-domain-bits=124429959.84881338 remaining-domain-bits=124429959.84881338 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 73.3998 at iteration 0 source=root-heuristic-restricted event-epoch=1792704797.0579731 program-deadline=1792704792.5819983`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 73.3998 (no supervisor gap summary); best cost 73.3998.

### I492_c030_setup_build
- In progress: 2026-10-22T17:43:05.822419-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I492_c030_setup_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I492_c030_setup_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I492_c030_setup_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T17:45:28.009475-04:00; exit 0; runner elapsed 142.185 s.

### I493_c030_n38
- In progress: 2026-10-22T17:45:28.099701-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I493_c030_n38.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I493_c030_n38.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I493_c030_n38.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 38 --ignore-checkpoints --run-dir logs/gemma_output_runs/I493_c030_n38`
- Completed: 2026-10-22T17:48:30.164687-04:00; exit 0; runner elapsed 182.062 s.
  `[SearchEngine volume] iterations=0 elapsed=6.7211209519999997 total-domain-bits=124429959.84881338 remaining-domain-bits=124429959.84881338 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=73.399780000000007 final-bound=0 first-incumbent-elapsed=27.860941171646118 samples=3`
  `[SearchEngine] New best cost: 73.3998 at iteration 0 source=root-heuristic-restricted event-epoch=1792705581.3437729 program-deadline=1792705577.9828444`
- First incumbent: 27.860941172 s, cost 73.3998; best cost 73.39978.

### I494_c030_persist_build
- In progress: 2026-10-22T17:49:09.051560-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I494_c030_persist_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I494_c030_persist_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I494_c030_persist_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T17:51:30.038777-04:00; exit 0; runner elapsed 140.984 s.

### I495_c030_n38
- In progress: 2026-10-22T17:51:30.133529-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I495_c030_n38.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I495_c030_n38.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I495_c030_n38.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 38 --ignore-checkpoints --run-dir logs/gemma_output_runs/I495_c030_n38`
- Completed: 2026-10-22T17:54:47.391919-04:00; exit 0; runner elapsed 197.257 s.
  `[SearchEngine volume] iterations=0 elapsed=6.7374884860000002 total-domain-bits=124429959.84881338 remaining-domain-bits=124429959.84881338 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=73.399780000000007 final-bound=0 first-incumbent-elapsed=27.693156957626343 samples=3`
  `[SearchEngine] New best cost: 73.3998 at iteration 0 source=root-heuristic-restricted event-epoch=1792705943.244545 program-deadline=1792705940.0514004`
- First incumbent: 27.693156958 s, cost 73.3998; best cost 73.39978.

### I496_c030_twins_build
- In progress: 2026-10-22T18:03:39.322212-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I496_c030_twins_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I496_c030_twins_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I496_c030_twins_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T18:05:41.177108-04:00; exit 0; runner elapsed 121.853 s.

### I497_c030_n38
- In progress: 2026-10-22T18:05:41.270301-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I497_c030_n38.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I497_c030_n38.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I497_c030_n38.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 38 --ignore-checkpoints --run-dir logs/gemma_output_runs/I497_c030_n38`
- Completed: 2026-10-22T18:08:39.550233-04:00; exit 0; runner elapsed 178.278 s.
  `[SearchEngine volume] iterations=0 elapsed=5.9619729640000001 total-domain-bits=124429959.84881338 remaining-domain-bits=124429959.84881338 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=73.399780000000007 final-bound=0 first-incumbent-elapsed=26.879645586013794 samples=3`
  `[SearchEngine] New best cost: 73.3998 at iteration 0 source=root-heuristic-restricted event-epoch=1792706793.7105823 program-deadline=1792706791.3309481`
- First incumbent: 26.879645586 s, cost 73.3998; best cost 73.39978.

### I498_c030_n39
- In progress: 2026-10-22T18:08:39.616502-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I498_c030_n39.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I498_c030_n39.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I498_c030_n39.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 39 --ignore-checkpoints --run-dir logs/gemma_output_runs/I498_c030_n39`
- Completed: 2026-10-22T18:16:30.215133-04:00; exit 0; runner elapsed 470.597 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0670546400000003 total-domain-bits=127740759.46516791 remaining-domain-bits=127740759.46516791 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=77.747307000000006 final-bound=0 first-incumbent-elapsed=27.543094396591187 samples=3`
  `[SearchEngine] New best cost: 77.7473 at iteration 0 source=root-heuristic-restricted event-epoch=1792707253.4432259 program-deadline=1792707250.4001439`
- First incumbent: 27.543094397 s, cost 77.7473; best cost 77.747307.

### I499_c030_n39_confirm1
- In progress: 2026-10-22T18:17:31.506727-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I499_c030_n39_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I499_c030_n39_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I499_c030_n39_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 39 --ignore-checkpoints --run-dir logs/gemma_output_runs/I499_c030_n39_confirm1`
- Completed: 2026-10-22T18:20:55.145510-04:00; exit 0; runner elapsed 203.637 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0835623490000001 total-domain-bits=127740759.46516791 remaining-domain-bits=127740759.46516791 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=77.747307000000006 final-bound=0 first-incumbent-elapsed=27.564896106719971 samples=3`
  `[SearchEngine] New best cost: 77.7473 at iteration 0 source=root-heuristic-restricted event-epoch=1792707505.4663942 program-deadline=1792707502.4015126`
- First incumbent: 27.564896107 s, cost 77.7473; best cost 77.747307.

### I500_c030_n39_confirm2
- In progress: 2026-10-22T18:20:55.237878-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I500_c030_n39_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I500_c030_n39_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I500_c030_n39_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 39 --ignore-checkpoints --run-dir logs/gemma_output_runs/I500_c030_n39_confirm2`
- Completed: 2026-10-22T18:23:55.694141-04:00; exit 0; runner elapsed 180.455 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1232778699999999 total-domain-bits=127740759.46516791 remaining-domain-bits=127740759.46516791 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=77.747307000000006 final-bound=0 first-incumbent-elapsed=27.590890407562256 samples=3`
  `[SearchEngine] New best cost: 77.7473 at iteration 0 source=root-heuristic-restricted event-epoch=1792707709.1244206 program-deadline=1792707706.0335438`
- First incumbent: 27.590890408 s, cost 77.7473; best cost 77.747307.

### I501_c030_n37_regression
- In progress: 2026-10-22T18:23:55.764009-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I501_c030_n37_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I501_c030_n37_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I501_c030_n37_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 37 --ignore-checkpoints --run-dir logs/gemma_output_runs/I501_c030_n37_regression`
- Completed: 2026-10-22T18:27:34.105484-04:00; exit 0; runner elapsed 218.340 s.
  `[SearchEngine volume] iterations=0 elapsed=4.7049511659999999 total-domain-bits=121113513.43611048 remaining-domain-bits=121113513.43611048 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=693.13391100000001 final-bound=0 first-incumbent-elapsed=25.123485803604126 samples=3`
  `[SearchEngine] New best cost: 693.134 at iteration 0 source=root-heuristic-restricted event-epoch=1792707886.1845913 program-deadline=1792707885.5611196`
- First incumbent: 25.123485804 s, cost 693.134; best cost 693.133911.

### I502_c030_oracle
- In progress: 2026-10-22T18:27:34.172137-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I502_c030_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I502_c030_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I502_c030_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T18:27:38.862182-04:00; exit 0; runner elapsed 4.689 s.
- Oracle verdict: 48/48 cases passed.

### I503_c030_search_gap
- In progress: 2026-10-22T18:27:38.921861-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I503_c030_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I503_c030_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I503_c030_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T18:27:38.945777-04:00; exit 0; runner elapsed 0.022 s.

### I504_c030_native__private_layout
- In progress: 2026-10-22T18:27:39.004167-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I504_c030_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I504_c030_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I504_c030_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T18:27:39.183657-04:00; exit 0; runner elapsed 0.178 s.

### I505_c030_native__write_after_read_cache
- In progress: 2026-10-22T18:27:39.255739-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I505_c030_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I505_c030_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I505_c030_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T18:27:39.472376-04:00; exit 0; runner elapsed 0.215 s.

### I506_c030_native__incremental_propagators
- In progress: 2026-10-22T18:27:39.550532-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I506_c030_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I506_c030_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I506_c030_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T18:27:39.769599-04:00; exit 0; runner elapsed 0.217 s.

### I507_c030_native__start_rank_window
- In progress: 2026-10-22T18:27:39.849761-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I507_c030_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I507_c030_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I507_c030_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T18:27:40.059892-04:00; exit 0; runner elapsed 0.207 s.

### I508_c030_native__selection_reachability
- In progress: 2026-10-22T18:27:40.129049-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I508_c030_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I508_c030_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I508_c030_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T18:27:40.378176-04:00; exit 0; runner elapsed 0.247 s.

### I509_c030_native__relaxation_bound
- In progress: 2026-10-22T18:27:40.479021-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I509_c030_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I509_c030_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I509_c030_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T18:29:29.441265-04:00; exit 0; runner elapsed 108.961 s.

### I510_c030_native__prune_egraph
- In progress: 2026-10-22T18:29:29.518289-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I510_c030_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I510_c030_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I510_c030_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T18:29:29.763111-04:00; exit 0; runner elapsed 0.243 s.

### I511_c031_origins_build
- In progress: 2026-10-22T18:39:42.410030-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I511_c031_origins_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I511_c031_origins_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I511_c031_origins_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T18:41:45.226847-04:00; exit 0; runner elapsed 122.815 s.

### I512_c031_n40
- In progress: 2026-10-22T18:41:45.317082-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I512_c031_n40.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I512_c031_n40.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I512_c031_n40.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 40 --ignore-checkpoints --run-dir logs/gemma_output_runs/I512_c031_n40`
- Completed: 2026-10-22T18:50:28.951155-04:00; exit 0; runner elapsed 523.632 s.
  `[SearchEngine volume] iterations=0 elapsed=5.6761750419999997 total-domain-bits=131121970.79991183 remaining-domain-bits=131121970.79991183 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.352119000000002 final-bound=0 first-incumbent-elapsed=27.162854194641113 samples=3`
  `[SearchEngine] New best cost: 76.3521 at iteration 0 source=root-heuristic-restricted event-epoch=1792709256.401252 program-deadline=1792709253.7384105`
- First incumbent: 27.162854195 s, cost 76.3521; best cost 76.352119.

### I513_c031_n41
- In progress: 2026-10-22T18:50:57.494838-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I513_c031_n41.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I513_c031_n41.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I513_c031_n41.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I513_c031_n41`
- Completed: 2026-10-22T18:57:07.363502-04:00; exit 1; runner elapsed 369.867 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4296952809999999 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792709825.8430181 program-deadline=1792709822.8459306`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 72.3879 (no supervisor gap summary); best cost 72.3879.

### I514_c031_serial_build
- In progress: 2026-10-22T19:04:56.354206-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I514_c031_serial_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I514_c031_serial_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I514_c031_serial_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T19:06:58.213736-04:00; exit 0; runner elapsed 121.858 s.

### I515_c031_n40
- In progress: 2026-10-22T19:06:58.301496-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I515_c031_n40.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I515_c031_n40.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I515_c031_n40.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 40 --ignore-checkpoints --run-dir logs/gemma_output_runs/I515_c031_n40`
- Completed: 2026-10-22T19:10:45.973748-04:00; exit 0; runner elapsed 227.671 s.
  `[SearchEngine volume] iterations=0 elapsed=5.5979900000000002 total-domain-bits=131121970.79991183 remaining-domain-bits=131121970.79991183 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.352119000000002 final-bound=0 first-incumbent-elapsed=27.103334426879883 samples=3`
  `[SearchEngine] New best cost: 76.3521 at iteration 0 source=root-heuristic-restricted event-epoch=1792710472.297595 program-deadline=1792710469.6942732`
- First incumbent: 27.103334427 s, cost 76.3521; best cost 76.352119.

### I516_c031_n41
- In progress: 2026-10-22T19:10:46.039062-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I516_c031_n41.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I516_c031_n41.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I516_c031_n41.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I516_c031_n41`
- Completed: 2026-10-22T19:14:55.425322-04:00; exit 0; runner elapsed 249.385 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4240842430000002 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.549988985061646 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792710700.9343145 program-deadline=1792710697.8843389`
- First incumbent: 27.549988985 s, cost 72.3879; best cost 72.387917.

### I517_c031_n41_confirm1
- In progress: 2026-10-22T19:16:37.767496-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I517_c031_n41_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I517_c031_n41_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I517_c031_n41_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I517_c031_n41_confirm1`
- Completed: 2026-10-22T19:20:41.502100-04:00; exit 0; runner elapsed 243.731 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4130133000000002 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.477222442626953 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792711052.8399823 program-deadline=1792711049.8627734`
- First incumbent: 27.477222443 s, cost 72.3879; best cost 72.387917.

### I518_c031_n41_confirm2
- In progress: 2026-10-22T19:20:41.571184-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I518_c031_n41_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I518_c031_n41_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I518_c031_n41_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I518_c031_n41_confirm2`
- Completed: 2026-10-22T19:24:38.824939-04:00; exit 0; runner elapsed 237.250 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4163219009999999 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.53730583190918 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792711296.5119953 program-deadline=1792711293.4747028`
- First incumbent: 27.537305832 s, cost 72.3879; best cost 72.387917.

### I519_c031_n39_regression
- In progress: 2026-10-22T19:24:38.899275-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I519_c031_n39_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I519_c031_n39_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I519_c031_n39_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 39 --ignore-checkpoints --run-dir logs/gemma_output_runs/I519_c031_n39_regression`
- Completed: 2026-10-22T19:27:41.788373-04:00; exit 0; runner elapsed 182.888 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0600560850000003 total-domain-bits=127740759.46516791 remaining-domain-bits=127740759.46516791 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=77.747307000000006 final-bound=0 first-incumbent-elapsed=27.051803112030029 samples=3`
  `[SearchEngine] New best cost: 77.7473 at iteration 0 source=root-heuristic-restricted event-epoch=1792711532.3594782 program-deadline=1792711529.8076859`
- First incumbent: 27.051803112 s, cost 77.7473; best cost 77.747307.

### I520_c031_oracle
- In progress: 2026-10-22T19:27:41.857236-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I520_c031_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I520_c031_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I520_c031_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T19:27:46.593254-04:00; exit 0; runner elapsed 4.734 s.
- Oracle verdict: 48/48 cases passed.

### I521_c031_search_gap
- In progress: 2026-10-22T19:27:46.655049-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I521_c031_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I521_c031_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I521_c031_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T19:27:46.681015-04:00; exit 0; runner elapsed 0.023 s.

### I522_c031_native__private_layout
- In progress: 2026-10-22T19:27:46.741975-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I522_c031_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I522_c031_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I522_c031_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T19:27:46.958326-04:00; exit 0; runner elapsed 0.215 s.

### I523_c031_native__write_after_read_cache
- In progress: 2026-10-22T19:27:47.037547-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I523_c031_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I523_c031_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I523_c031_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T19:27:47.242076-04:00; exit 0; runner elapsed 0.202 s.

### I524_c031_native__incremental_propagators
- In progress: 2026-10-22T19:27:47.314343-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I524_c031_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I524_c031_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I524_c031_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T19:27:47.525948-04:00; exit 0; runner elapsed 0.210 s.

### I525_c031_native__start_rank_window
- In progress: 2026-10-22T19:27:47.597642-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I525_c031_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I525_c031_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I525_c031_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T19:27:47.771385-04:00; exit 0; runner elapsed 0.172 s.

### I526_c031_native__selection_reachability
- In progress: 2026-10-22T19:27:47.843859-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I526_c031_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I526_c031_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I526_c031_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T19:27:48.099459-04:00; exit 0; runner elapsed 0.254 s.

### I527_c031_native__relaxation_bound
- In progress: 2026-10-22T19:27:48.204935-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I527_c031_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I527_c031_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I527_c031_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T19:29:36.040943-04:00; exit 0; runner elapsed 107.833 s.

### I528_c031_native__prune_egraph
- In progress: 2026-10-22T19:29:36.116310-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I528_c031_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I528_c031_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I528_c031_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T19:29:36.321715-04:00; exit 0; runner elapsed 0.204 s.

### I529_c032_memo_build
- In progress: 2026-10-22T19:42:59.671565-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I529_c032_memo_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I529_c032_memo_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I529_c032_memo_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T19:45:02.319898-04:00; exit 0; runner elapsed 122.646 s.

### I530_c032_n42
- In progress: 2026-10-22T19:45:02.406804-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I530_c032_n42.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I530_c032_n42.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I530_c032_n42.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 42 --ignore-checkpoints --run-dir logs/gemma_output_runs/I530_c032_n42`
- Completed: 2026-10-22T19:51:14.060860-04:00; exit 1; runner elapsed 371.652 s.
  `[SearchEngine volume] iterations=0 elapsed=5.7562764709999996 total-domain-bits=137687443.63319042 remaining-domain-bits=137687443.63319042 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 76.459 at iteration 0 source=root-heuristic-restricted event-epoch=1792713071.8998737 program-deadline=1792713068.3996544`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 76.459 (no supervisor gap summary); best cost 76.459.

### I531_c032_n41
- In progress: 2026-10-22T19:51:27.424188-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I531_c032_n41.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I531_c032_n41.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I531_c032_n41.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I531_c032_n41`
- Completed: 2026-10-22T19:55:38.691455-04:00; exit 0; runner elapsed 251.265 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1884923010000001 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.005216836929321 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792713141.6839716 program-deadline=1792713139.1787672`
- First incumbent: 27.005216837 s, cost 72.3879; best cost 72.387917.

### I532_c032_n41_confirm1
- In progress: 2026-10-22T19:55:38.758781-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I532_c032_n41_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I532_c032_n41_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I532_c032_n41_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I532_c032_n41_confirm1`
- Completed: 2026-10-22T19:59:50.168291-04:00; exit 0; runner elapsed 251.406 s.
  `[SearchEngine volume] iterations=0 elapsed=5.2278101220000002 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.078228712081909 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792713392.9700685 program-deadline=1792713390.3918545`
- First incumbent: 27.078228712 s, cost 72.3879; best cost 72.387917.

### I533_c032_n41_confirm2
- In progress: 2026-10-22T19:59:50.239892-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I533_c032_n41_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I533_c032_n41_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I533_c032_n41_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I533_c032_n41_confirm2`
- Completed: 2026-10-22T20:03:47.514588-04:00; exit 0; runner elapsed 237.273 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1506089959999999 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=27.03784704208374 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792713644.5679059 program-deadline=1792713642.0300717`
- First incumbent: 27.037847042 s, cost 72.3879; best cost 72.387917.

### I534_c032_oracle
- In progress: 2026-10-22T20:04:45.614943-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I534_c032_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I534_c032_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I534_c032_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T20:04:50.460920-04:00; exit 0; runner elapsed 4.842 s.
- Oracle verdict: 48/48 cases passed.

### I535_c032_search_gap
- In progress: 2026-10-22T20:04:50.524474-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I535_c032_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I535_c032_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I535_c032_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T20:04:50.545505-04:00; exit 0; runner elapsed 0.019 s.

### I536_c032_native__private_layout
- In progress: 2026-10-22T20:04:50.605717-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I536_c032_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I536_c032_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I536_c032_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T20:04:50.805794-04:00; exit 0; runner elapsed 0.198 s.

### I537_c032_native__write_after_read_cache
- In progress: 2026-10-22T20:04:50.881611-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I537_c032_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I537_c032_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I537_c032_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T20:04:51.072341-04:00; exit 0; runner elapsed 0.189 s.

### I538_c032_native__incremental_propagators
- In progress: 2026-10-22T20:04:51.147137-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I538_c032_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I538_c032_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I538_c032_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T20:04:51.351933-04:00; exit 0; runner elapsed 0.203 s.

### I539_c032_native__start_rank_window
- In progress: 2026-10-22T20:04:51.429373-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I539_c032_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I539_c032_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I539_c032_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T20:04:51.617679-04:00; exit 0; runner elapsed 0.187 s.

### I540_c032_native__selection_reachability
- In progress: 2026-10-22T20:04:51.695741-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I540_c032_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I540_c032_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I540_c032_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T20:04:51.943859-04:00; exit 0; runner elapsed 0.245 s.

### I541_c032_native__relaxation_bound
- In progress: 2026-10-22T20:04:52.050737-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I541_c032_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I541_c032_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I541_c032_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T20:06:39.832636-04:00; exit 0; runner elapsed 107.780 s.

### I542_c032_native__prune_egraph
- In progress: 2026-10-22T20:06:39.902246-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I542_c032_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I542_c032_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I542_c032_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T20:06:40.096172-04:00; exit 0; runner elapsed 0.192 s.

### I543_c033_filter_build
- In progress: 2026-10-22T20:16:43.005342-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I543_c033_filter_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I543_c033_filter_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I543_c033_filter_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T20:18:45.358226-04:00; exit 0; runner elapsed 122.350 s.

### I544_c033_n42
- In progress: 2026-10-22T20:18:45.453918-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I544_c033_n42.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I544_c033_n42.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I544_c033_n42.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 42 --ignore-checkpoints --run-dir logs/gemma_output_runs/I544_c033_n42`
- Completed: 2026-10-22T20:22:38.204065-04:00; exit 0; runner elapsed 232.748 s.
  `[SearchEngine volume] iterations=0 elapsed=5.6882953020000002 total-domain-bits=137687443.63319042 remaining-domain-bits=137687443.63319042 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.459014999999994 final-bound=0 first-incumbent-elapsed=27.534729480743408 samples=3`
  `[SearchEngine] New best cost: 76.459 at iteration 0 source=root-heuristic-restricted event-epoch=1792714780.3654308 program-deadline=1792714777.3307128`
- First incumbent: 27.534729481 s, cost 76.459; best cost 76.459015.

### I545_c033_n42_confirm1
- In progress: 2026-10-22T20:22:52.791130-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I545_c033_n42_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I545_c033_n42_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I545_c033_n42_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 42 --ignore-checkpoints --run-dir logs/gemma_output_runs/I545_c033_n42_confirm1`
- Completed: 2026-10-22T20:27:03.384970-04:00; exit 0; runner elapsed 250.592 s.
  `[SearchEngine volume] iterations=0 elapsed=5.7013640780000001 total-domain-bits=137687443.63319042 remaining-domain-bits=137687443.63319042 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.459014999999994 final-bound=0 first-incumbent-elapsed=27.563192844390869 samples=3`
  `[SearchEngine] New best cost: 76.459 at iteration 0 source=root-heuristic-restricted event-epoch=1792715027.8449514 program-deadline=1792715024.7817693`
- First incumbent: 27.563192844 s, cost 76.459; best cost 76.459015.

### I546_c033_n42_confirm2
- In progress: 2026-10-22T20:27:03.453521-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I546_c033_n42_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I546_c033_n42_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I546_c033_n42_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 42 --ignore-checkpoints --run-dir logs/gemma_output_runs/I546_c033_n42_confirm2`
- Completed: 2026-10-22T20:31:13.095296-04:00; exit 0; runner elapsed 249.640 s.
  `[SearchEngine volume] iterations=0 elapsed=5.7076237340000002 total-domain-bits=137687443.63319042 remaining-domain-bits=137687443.63319042 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.459014999999994 final-bound=0 first-incumbent-elapsed=27.6547691822052 samples=3`
  `[SearchEngine] New best cost: 76.459 at iteration 0 source=root-heuristic-restricted event-epoch=1792715278.5998569 program-deadline=1792715275.4450982`
- First incumbent: 27.654769182 s, cost 76.459; best cost 76.459015.

### I547_c033_n41_regression
- In progress: 2026-10-22T20:31:13.164191-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I547_c033_n41_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I547_c033_n41_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I547_c033_n41_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 41 --ignore-checkpoints --run-dir logs/gemma_output_runs/I547_c033_n41_regression`
- Completed: 2026-10-22T20:35:21.889013-04:00; exit 0; runner elapsed 248.723 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1215376609999996 total-domain-bits=134375658.09119266 remaining-domain-bits=134375658.09119266 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=72.387917000000002 final-bound=0 first-incumbent-elapsed=26.599437713623047 samples=3`
  `[SearchEngine] New best cost: 72.3879 at iteration 0 source=root-heuristic-restricted event-epoch=1792715526.5856259 program-deadline=1792715524.4861991`
- First incumbent: 26.599437714 s, cost 72.3879; best cost 72.387917.

### I548_c033_oracle
- In progress: 2026-10-22T20:35:21.957187-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I548_c033_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I548_c033_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I548_c033_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T20:35:26.651185-04:00; exit 0; runner elapsed 4.693 s.
- Oracle verdict: 48/48 cases passed.

### I549_c033_search_gap
- In progress: 2026-10-22T20:35:26.716298-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I549_c033_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I549_c033_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I549_c033_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T20:35:26.736397-04:00; exit 0; runner elapsed 0.019 s.

### I550_c033_native__private_layout
- In progress: 2026-10-22T20:35:26.794030-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I550_c033_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I550_c033_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I550_c033_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T20:35:27.017624-04:00; exit 0; runner elapsed 0.222 s.

### I551_c033_native__write_after_read_cache
- In progress: 2026-10-22T20:35:27.093488-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I551_c033_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I551_c033_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I551_c033_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T20:35:27.288611-04:00; exit 0; runner elapsed 0.193 s.

### I552_c033_native__incremental_propagators
- In progress: 2026-10-22T20:35:27.362968-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I552_c033_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I552_c033_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I552_c033_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T20:35:27.576620-04:00; exit 0; runner elapsed 0.212 s.

### I553_c033_native__start_rank_window
- In progress: 2026-10-22T20:35:27.655564-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I553_c033_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I553_c033_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I553_c033_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T20:35:27.859389-04:00; exit 0; runner elapsed 0.201 s.

### I554_c033_native__selection_reachability
- In progress: 2026-10-22T20:35:27.933194-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I554_c033_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I554_c033_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I554_c033_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T20:35:28.174940-04:00; exit 0; runner elapsed 0.240 s.

### I555_c033_native__relaxation_bound
- In progress: 2026-10-22T20:35:28.271001-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I555_c033_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I555_c033_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I555_c033_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T20:37:16.736297-04:00; exit 0; runner elapsed 108.464 s.

### I556_c033_native__prune_egraph
- In progress: 2026-10-22T20:37:16.814072-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I556_c033_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I556_c033_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I556_c033_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T20:37:17.010258-04:00; exit 0; runner elapsed 0.194 s.

### I557_c034_costload_build
- In progress: 2026-10-22T21:01:33.741002-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I557_c034_costload_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I557_c034_costload_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I557_c034_costload_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T21:03:36.028312-04:00; exit 0; runner elapsed 122.286 s.

### I558_c034_n43
- In progress: 2026-10-22T21:03:36.118946-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I558_c034_n43.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I558_c034_n43.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I558_c034_n43.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I558_c034_n43`
- Completed: 2026-10-22T21:09:59.202648-04:00; exit 1; runner elapsed 383.082 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1410073030000003 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792717797.3597174 program-deadline=1792717793.5782239`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 81.6109 (no supervisor gap summary); best cost 81.6109.

### I559_c034_graphs_build
- In progress: 2026-10-22T21:22:05.322503-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I559_c034_graphs_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I559_c034_graphs_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I559_c034_graphs_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T21:24:07.283513-04:00; exit 0; runner elapsed 121.959 s.

### I560_c034_n43
- In progress: 2026-10-22T21:24:07.367461-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I560_c034_n43.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I560_c034_n43.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I560_c034_n43.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I560_c034_n43`
- Completed: 2026-10-22T21:25:05.545224-04:00; exit 1; runner elapsed 58.176 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1512306470000002 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792718703.7396362 program-deadline=1792718699.963383`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 81.6109 (no supervisor gap summary); best cost 81.6109.

### I561_c034_nolock_build
- In progress: 2026-10-22T21:29:28.261968-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I561_c034_nolock_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I561_c034_nolock_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I561_c034_nolock_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T21:31:30.718532-04:00; exit 0; runner elapsed 122.455 s.

### I562_c034_n43
- In progress: 2026-10-22T21:31:30.804119-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I562_c034_n43.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I562_c034_n43.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I562_c034_n43.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I562_c034_n43`
- Completed: 2026-10-22T21:32:28.952145-04:00; exit 1; runner elapsed 58.146 s.
  `[SearchEngine volume] iterations=0 elapsed=6.093754949 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792719147.1490369 program-deadline=1792719143.3692262`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 81.6109 (no supervisor gap summary); best cost 81.6109.

### I563_c034_names_build
- In progress: 2026-10-22T21:50:36.283710-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I563_c034_names_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I563_c034_names_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I563_c034_names_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T21:52:38.926639-04:00; exit 0; runner elapsed 122.640 s.

### I564_c034_n43
- In progress: 2026-10-22T21:52:39.025194-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I564_c034_n43.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I564_c034_n43.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I564_c034_n43.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I564_c034_n43`
- Completed: 2026-10-22T21:57:24.863363-04:00; exit 0; runner elapsed 285.836 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1300903309999999 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=81.610916000000003 final-bound=0 first-incumbent-elapsed=28.057661056518555 samples=3`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792720415.1029 program-deadline=1792720411.5452507`
- First incumbent: 28.057661057 s, cost 81.6109; best cost 81.610916.

### I565_c034_persist_build
- In progress: 2026-10-22T22:02:33.566199-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I565_c034_persist_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I565_c034_persist_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I565_c034_persist_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T22:04:36.276347-04:00; exit 0; runner elapsed 122.708 s.

### I566_c034_n43
- In progress: 2026-10-22T22:04:36.362477-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I566_c034_n43.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I566_c034_n43.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I566_c034_n43.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I566_c034_n43`
- Completed: 2026-10-22T22:09:18.717778-04:00; exit 0; runner elapsed 282.354 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0374908920000001 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=81.610916000000003 final-bound=0 first-incumbent-elapsed=27.899813652038574 samples=3`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792721132.2475009 program-deadline=1792721128.8476989`
- First incumbent: 27.899813652 s, cost 81.6109; best cost 81.610916.

### I567_c034_n43_confirm1
- In progress: 2026-10-22T22:09:18.789580-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I567_c034_n43_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I567_c034_n43_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I567_c034_n43_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I567_c034_n43_confirm1`
- Completed: 2026-10-22T22:14:08.163683-04:00; exit 0; runner elapsed 289.372 s.
  `[SearchEngine volume] iterations=0 elapsed=6.0230769009999996 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=81.610916000000003 final-bound=0 first-incumbent-elapsed=28.047106742858887 samples=3`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792721414.8569117 program-deadline=1792721411.3098171`
- First incumbent: 28.047106743 s, cost 81.6109; best cost 81.610916.

### I568_c034_n43_confirm2
- In progress: 2026-10-22T22:14:08.234051-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I568_c034_n43_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I568_c034_n43_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I568_c034_n43_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I568_c034_n43_confirm2`
- Completed: 2026-10-22T22:18:54.651421-04:00; exit 0; runner elapsed 286.415 s.
  `[SearchEngine volume] iterations=0 elapsed=6.1042873369999997 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=81.610916000000003 final-bound=0 first-incumbent-elapsed=28.008164167404175 samples=3`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792721704.2355571 program-deadline=1792721700.7274032`
- First incumbent: 28.008164167 s, cost 81.6109; best cost 81.610916.

### I569_c034_n42_regression
- In progress: 2026-10-22T22:19:56.899933-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I569_c034_n42_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I569_c034_n42_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I569_c034_n42_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 42 --ignore-checkpoints --run-dir logs/gemma_output_runs/I569_c034_n42_regression`
- Completed: 2026-10-22T22:24:13.070622-04:00; exit 0; runner elapsed 256.167 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3461697539999999 total-domain-bits=137687443.63319042 remaining-domain-bits=137687443.63319042 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=76.459014999999994 final-bound=0 first-incumbent-elapsed=26.92028284072876 samples=3`
  `[SearchEngine] New best cost: 76.459 at iteration 0 source=root-heuristic-restricted event-epoch=1792722051.5622065 program-deadline=1792722049.1419349`
- First incumbent: 26.920282841 s, cost 76.459; best cost 76.459015.

### I570_c034_oracle
- In progress: 2026-10-22T22:24:13.149703-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I570_c034_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I570_c034_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I570_c034_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T22:24:18.022310-04:00; exit 0; runner elapsed 4.871 s.
- Oracle verdict: 48/48 cases passed.

### I571_c034_search_gap
- In progress: 2026-10-22T22:24:18.084813-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I571_c034_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I571_c034_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I571_c034_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T22:24:18.104626-04:00; exit 0; runner elapsed 0.018 s.

### I572_c034_native__private_layout
- In progress: 2026-10-22T22:24:18.175446-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I572_c034_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I572_c034_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I572_c034_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T22:24:18.363987-04:00; exit 0; runner elapsed 0.187 s.

### I573_c034_native__write_after_read_cache
- In progress: 2026-10-22T22:24:18.444823-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I573_c034_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I573_c034_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I573_c034_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T22:24:18.647574-04:00; exit 0; runner elapsed 0.201 s.

### I574_c034_native__incremental_propagators
- In progress: 2026-10-22T22:24:18.726702-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I574_c034_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I574_c034_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I574_c034_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T22:24:18.944162-04:00; exit 0; runner elapsed 0.216 s.

### I575_c034_native__start_rank_window
- In progress: 2026-10-22T22:24:19.025645-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I575_c034_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I575_c034_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I575_c034_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T22:24:19.248891-04:00; exit 0; runner elapsed 0.221 s.

### I576_c034_native__selection_reachability
- In progress: 2026-10-22T22:24:19.336335-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I576_c034_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I576_c034_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I576_c034_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T22:24:19.565630-04:00; exit 0; runner elapsed 0.228 s.

### I577_c034_native__relaxation_bound
- In progress: 2026-10-22T22:24:19.679329-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I577_c034_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I577_c034_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I577_c034_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T22:26:07.006034-04:00; exit 0; runner elapsed 107.325 s.

### I578_c034_native__prune_egraph
- In progress: 2026-10-22T22:26:07.082519-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I578_c034_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I578_c034_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I578_c034_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T22:26:07.301876-04:00; exit 0; runner elapsed 0.218 s.

### I579_c035_precedence_build
- In progress: 2026-10-22T22:31:11.778849-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I579_c035_precedence_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I579_c035_precedence_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I579_c035_precedence_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T22:33:14.284441-04:00; exit 0; runner elapsed 122.504 s.

### I580_c035_n44
- In progress: 2026-10-22T22:33:14.372601-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I580_c035_n44.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I580_c035_n44.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I580_c035_n44.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I580_c035_n44`
- Completed: 2026-10-22T22:39:55.654459-04:00; exit 1; runner elapsed 401.279 s.
  `[SearchEngine volume] iterations=0 elapsed=5.8671127079999996 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792723193.8590519 program-deadline=1792723190.0174272`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 80.8816 (no supervisor gap summary); best cost 80.8816.

### I581_c035_covering_build
- In progress: 2026-10-22T22:50:04.782291-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I581_c035_covering_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I581_c035_covering_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I581_c035_covering_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T22:52:06.947624-04:00; exit 0; runner elapsed 122.163 s.

### I582_c035_n44
- In progress: 2026-10-22T22:52:07.041248-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I582_c035_n44.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I582_c035_n44.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I582_c035_n44.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I582_c035_n44`
- Completed: 2026-10-22T22:57:17.348811-04:00; exit 0; runner elapsed 310.306 s.
  `[SearchEngine volume] iterations=0 elapsed=5.7518866510000004 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.881568999999999 final-bound=0 first-incumbent-elapsed=28.184388637542725 samples=3`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792723984.1643789 program-deadline=1792723980.4800007`
- First incumbent: 28.184388638 s, cost 80.8816; best cost 80.881569.

### I583_c035_bases_build
- In progress: 2026-10-22T23:04:27.812549-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I583_c035_bases_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I583_c035_bases_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I583_c035_bases_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T23:06:30.432028-04:00; exit 0; runner elapsed 122.618 s.

### I584_c035_n44
- In progress: 2026-10-22T23:06:30.535225-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I584_c035_n44.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I584_c035_n44.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I584_c035_n44.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I584_c035_n44`
- Completed: 2026-10-22T23:11:40.677156-04:00; exit 0; runner elapsed 310.138 s.
  `[SearchEngine volume] iterations=0 elapsed=5.6534919720000003 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.881568999999999 final-bound=0 first-incumbent-elapsed=28.037186145782471 samples=3`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792724847.47703 program-deadline=1792724843.9398539`
- First incumbent: 28.037186146 s, cost 80.8816; best cost 80.881569.

### I585_c035_n44_confirm1
- In progress: 2026-10-22T23:11:40.751027-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I585_c035_n44_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I585_c035_n44_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I585_c035_n44_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I585_c035_n44_confirm1`
- Completed: 2026-10-22T23:16:46.594479-04:00; exit 0; runner elapsed 305.842 s.
  `[SearchEngine volume] iterations=0 elapsed=5.6013199680000003 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.881568999999999 final-bound=0 first-incumbent-elapsed=28.032699823379517 samples=3`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792725157.5967855 program-deadline=1792725154.0640979`
- First incumbent: 28.032699823 s, cost 80.8816; best cost 80.881569.

### I586_c035_n44_confirm2
- In progress: 2026-10-22T23:16:46.676347-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I586_c035_n44_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I586_c035_n44_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I586_c035_n44_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I586_c035_n44_confirm2`
- Completed: 2026-10-22T23:21:50.298755-04:00; exit 0; runner elapsed 303.620 s.
  `[SearchEngine volume] iterations=0 elapsed=5.5563610260000003 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.881568999999999 final-bound=0 first-incumbent-elapsed=28.016229867935181 samples=3`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792725463.7923217 program-deadline=1792725460.2761037`
- First incumbent: 28.016229868 s, cost 80.8816; best cost 80.881569.

### I587_c035_n43_regression
- In progress: 2026-10-22T23:22:58.944890-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I587_c035_n43_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I587_c035_n43_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I587_c035_n43_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 43 --ignore-checkpoints --run-dir logs/gemma_output_runs/I587_c035_n43_regression`
- Completed: 2026-10-22T23:27:39.969345-04:00; exit 0; runner elapsed 281.023 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4301069460000004 total-domain-bits=141002167.41865814 remaining-domain-bits=141002167.41865814 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=81.610916000000003 final-bound=0 first-incumbent-elapsed=27.430076360702515 samples=3`
  `[SearchEngine] New best cost: 81.6109 at iteration 0 source=root-heuristic-restricted event-epoch=1792725834.8855722 program-deadline=1792725831.955508`
- First incumbent: 27.430076361 s, cost 81.6109; best cost 81.610916.

### I588_c035_oracle
- In progress: 2026-10-22T23:27:40.044518-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I588_c035_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I588_c035_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I588_c035_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-22T23:27:44.797462-04:00; exit 0; runner elapsed 4.751 s.
- Oracle verdict: 48/48 cases passed.

### I589_c035_search_gap
- In progress: 2026-10-22T23:27:44.866488-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I589_c035_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I589_c035_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I589_c035_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-22T23:27:44.889094-04:00; exit 0; runner elapsed 0.021 s.

### I590_c035_native__private_layout
- In progress: 2026-10-22T23:27:44.960657-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I590_c035_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I590_c035_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I590_c035_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-22T23:27:45.143388-04:00; exit 0; runner elapsed 0.181 s.

### I591_c035_native__write_after_read_cache
- In progress: 2026-10-22T23:27:45.225298-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I591_c035_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I591_c035_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I591_c035_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-22T23:27:45.406763-04:00; exit 0; runner elapsed 0.180 s.

### I592_c035_native__incremental_propagators
- In progress: 2026-10-22T23:27:45.489298-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I592_c035_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I592_c035_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I592_c035_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-22T23:27:45.707001-04:00; exit 0; runner elapsed 0.214 s.

### I593_c035_native__start_rank_window
- In progress: 2026-10-22T23:27:45.797041-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I593_c035_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I593_c035_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I593_c035_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-22T23:27:46.013768-04:00; exit 0; runner elapsed 0.215 s.

### I594_c035_native__selection_reachability
- In progress: 2026-10-22T23:27:46.100940-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I594_c035_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I594_c035_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I594_c035_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-22T23:27:46.376925-04:00; exit 0; runner elapsed 0.274 s.

### I595_c035_native__relaxation_bound
- In progress: 2026-10-22T23:27:46.497548-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I595_c035_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I595_c035_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I595_c035_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-22T23:29:33.248521-04:00; exit 0; runner elapsed 106.749 s.

### I596_c035_native__prune_egraph
- In progress: 2026-10-22T23:29:33.337595-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I596_c035_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I596_c035_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I596_c035_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-22T23:29:33.563074-04:00; exit 0; runner elapsed 0.223 s.

### I597_c036_parents_build
- In progress: 2026-10-22T23:51:24.878414-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I597_c036_parents_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I597_c036_parents_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I597_c036_parents_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-22T23:53:28.550375-04:00; exit 0; runner elapsed 123.669 s.

### I598_c036_n46
- In progress: 2026-10-22T23:53:28.652760-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I598_c036_n46.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I598_c036_n46.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I598_c036_n46.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 46 --ignore-checkpoints --run-dir logs/gemma_output_runs/I598_c036_n46`
- Completed: 2026-10-23T00:06:15.178056-04:00; exit 0; runner elapsed 766.524 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1450666280000004 total-domain-bits=150947892.70465142 remaining-domain-bits=150947892.70465142 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.895331999999996 final-bound=0 first-incumbent-elapsed=28.112761974334717 samples=3`
  `[SearchEngine] New best cost: 80.8953 at iteration 0 source=root-heuristic-restricted event-epoch=1792728088.4048862 program-deadline=1792728084.7921355`
- First incumbent: 28.112761974 s, cost 80.8953; best cost 80.895332.

### I599_c036_n47
- In progress: 2026-10-23T00:06:15.249773-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I599_c036_n47.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I599_c036_n47.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I599_c036_n47.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I599_c036_n47`
- Completed: 2026-10-23T00:13:49.324595-04:00; exit 1; runner elapsed 454.073 s.
  `[SearchEngine volume] iterations=0 elapsed=5.369020345 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792728828.3388112 program-deadline=1792728823.7366383`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 80.4027 (no supervisor gap summary); best cost 80.4027.

### I600_c036_n46_confirm1
- In progress: 2026-10-23T00:14:25.026194-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I600_c036_n46_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I600_c036_n46_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I600_c036_n46_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 46 --ignore-checkpoints --run-dir logs/gemma_output_runs/I600_c036_n46_confirm1`
- Completed: 2026-10-23T00:20:10.758511-04:00; exit 0; runner elapsed 345.729 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1000466749999998 total-domain-bits=150947892.70465142 remaining-domain-bits=150947892.70465142 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.895331999999996 final-bound=0 first-incumbent-elapsed=28.176634073257446 samples=3`
  `[SearchEngine] New best cost: 80.8953 at iteration 0 source=root-heuristic-restricted event-epoch=1792728923.5508747 program-deadline=1792728919.874253`
- First incumbent: 28.176634073 s, cost 80.8953; best cost 80.895332.

### I601_c036_n46_confirm2
- In progress: 2026-10-23T00:20:10.828745-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I601_c036_n46_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I601_c036_n46_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I601_c036_n46_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 46 --ignore-checkpoints --run-dir logs/gemma_output_runs/I601_c036_n46_confirm2`
- Completed: 2026-10-23T00:25:59.929690-04:00; exit 0; runner elapsed 349.099 s.
  `[SearchEngine volume] iterations=0 elapsed=5.0938542289999997 total-domain-bits=150947892.70465142 remaining-domain-bits=150947892.70465142 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.895331999999996 final-bound=0 first-incumbent-elapsed=28.105761051177979 samples=3`
  `[SearchEngine] New best cost: 80.8953 at iteration 0 source=root-heuristic-restricted event-epoch=1792729269.217308 program-deadline=1792729265.6115589`
- First incumbent: 28.105761051 s, cost 80.8953; best cost 80.895332.

### I602_c036_n44_regression
- In progress: 2026-10-23T00:26:00.007837-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I602_c036_n44_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I602_c036_n44_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I602_c036_n44_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 44 --ignore-checkpoints --run-dir logs/gemma_output_runs/I602_c036_n44_regression`
- Completed: 2026-10-23T00:30:57.177461-04:00; exit 0; runner elapsed 297.168 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3953330560000001 total-domain-bits=144383514.37867966 remaining-domain-bits=144383514.37867966 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.881568999999999 final-bound=0 first-incumbent-elapsed=27.505820989608765 samples=3`
  `[SearchEngine] New best cost: 80.8816 at iteration 0 source=root-heuristic-restricted event-epoch=1792729616.942318 program-deadline=1792729613.9365077`
- First incumbent: 27.505820990 s, cost 80.8816; best cost 80.881569.

### I603_c036_oracle
- In progress: 2026-10-23T00:30:57.257146-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I603_c036_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I603_c036_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I603_c036_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-23T00:31:01.961907-04:00; exit 0; runner elapsed 4.703 s.
- Oracle verdict: 48/48 cases passed.

### I604_c036_search_gap
- In progress: 2026-10-23T00:31:02.036397-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I604_c036_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I604_c036_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I604_c036_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-23T00:31:02.061696-04:00; exit 0; runner elapsed 0.024 s.

### I605_c036_native__private_layout
- In progress: 2026-10-23T00:31:02.131911-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I605_c036_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I605_c036_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I605_c036_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-23T00:31:02.334643-04:00; exit 0; runner elapsed 0.200 s.

### I606_c036_native__write_after_read_cache
- In progress: 2026-10-23T00:31:02.416299-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I606_c036_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I606_c036_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I606_c036_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-23T00:31:02.633755-04:00; exit 0; runner elapsed 0.216 s.

### I607_c036_native__incremental_propagators
- In progress: 2026-10-23T00:31:02.722368-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I607_c036_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I607_c036_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I607_c036_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-23T00:31:02.941852-04:00; exit 0; runner elapsed 0.218 s.

### I608_c036_native__start_rank_window
- In progress: 2026-10-23T00:31:03.028988-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I608_c036_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I608_c036_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I608_c036_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-23T00:31:03.229348-04:00; exit 0; runner elapsed 0.199 s.

### I609_c036_native__selection_reachability
- In progress: 2026-10-23T00:31:03.313643-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I609_c036_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I609_c036_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I609_c036_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-23T00:31:03.570366-04:00; exit 0; runner elapsed 0.254 s.

### I610_c036_native__relaxation_bound
- In progress: 2026-10-23T00:31:03.689981-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I610_c036_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I610_c036_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I610_c036_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-23T00:32:51.889969-04:00; exit 0; runner elapsed 108.197 s.

### I611_c036_native__prune_egraph
- In progress: 2026-10-23T00:32:51.975250-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I611_c036_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I611_c036_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I611_c036_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-23T00:32:52.205036-04:00; exit 0; runner elapsed 0.226 s.

### I612_c037_cycles_build
- In progress: 2026-10-23T00:54:37.065227-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I612_c037_cycles_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I612_c037_cycles_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I612_c037_cycles_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-23T00:56:40.660723-04:00; exit 0; runner elapsed 123.593 s.

### I613_c037_n47
- In progress: 2026-10-23T00:56:40.765809-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I613_c037_n47.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I613_c037_n47.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I613_c037_n47.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I613_c037_n47`
- Completed: 2026-10-23T00:57:41.473059-04:00; exit 1; runner elapsed 60.705 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4219820890000001 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792731459.8892446 program-deadline=1792731455.7370641`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 80.4027 (no supervisor gap summary); best cost 80.4027.

### I614_c037_rounds_build
- In progress: 2026-10-23T01:01:48.047070-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I614_c037_rounds_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I614_c037_rounds_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I614_c037_rounds_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-23T01:03:51.255550-04:00; exit 0; runner elapsed 123.206 s.

### I615_c037_n47
- In progress: 2026-10-23T01:03:51.360775-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I615_c037_n47.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I615_c037_n47.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I615_c037_n47.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I615_c037_n47`
- Completed: 2026-10-23T01:04:52.097973-04:00; exit 1; runner elapsed 60.736 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3679665139999999 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792731890.5274467 program-deadline=1792731886.5401487`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 80.4027 (no supervisor gap summary); best cost 80.4027.

### I616_c037_selptr_build
- In progress: 2026-10-23T01:14:36.888646-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I616_c037_selptr_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I616_c037_selptr_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I616_c037_selptr_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-23T01:16:40.155756-04:00; exit 0; runner elapsed 123.265 s.

### I617_c037_n47
- In progress: 2026-10-23T01:16:40.261874-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I617_c037_n47.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I617_c037_n47.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I617_c037_n47.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I617_c037_n47`
- Completed: 2026-10-23T01:17:40.876731-04:00; exit 1; runner elapsed 60.613 s.
  `[SearchEngine volume] iterations=0 elapsed=5.4545243780000003 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792732659.2844317 program-deadline=1792732655.1399968`
- Missing volume or gap report: FAILED evaluation.
- First incumbent: elapsed unavailable, cost 80.4027 (no supervisor gap summary); best cost 80.4027.

### I618_c037_costkey_build
- In progress: 2026-10-23T01:29:55.394368-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I618_c037_costkey_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I618_c037_costkey_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I618_c037_costkey_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-23T01:31:59.340926-04:00; exit 0; runner elapsed 123.943 s.

### I619_c037_n47
- In progress: 2026-10-23T01:31:59.445965-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I619_c037_n47.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I619_c037_n47.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I619_c037_n47.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I619_c037_n47`
- Completed: 2026-10-23T01:38:07.182920-04:00; exit 0; runner elapsed 367.735 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3743759689999999 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.402702000000005 final-bound=0 first-incumbent-elapsed=28.194376707077026 samples=3`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792733577.7390163 program-deadline=1792733574.0446517`
- First incumbent: 28.194376707 s, cost 80.4027; best cost 80.402702.

### I620_c037_n47_confirm1
- In progress: 2026-10-23T01:38:22.377660-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I620_c037_n47_confirm1.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I620_c037_n47_confirm1.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I620_c037_n47_confirm1.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I620_c037_n47_confirm1`
- Completed: 2026-10-23T01:44:31.339757-04:00; exit 0; runner elapsed 368.960 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3505715260000004 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.402702000000005 final-bound=0 first-incumbent-elapsed=28.172918796539307 samples=3`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792733960.7010727 program-deadline=1792733957.0281639`
- First incumbent: 28.172918797 s, cost 80.4027; best cost 80.402702.

### I621_c037_n47_confirm2
- In progress: 2026-10-23T01:44:31.421242-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I621_c037_n47_confirm2.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I621_c037_n47_confirm2.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I621_c037_n47_confirm2.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 47 --ignore-checkpoints --run-dir logs/gemma_output_runs/I621_c037_n47_confirm2`
- Completed: 2026-10-23T01:50:35.156671-04:00; exit 0; runner elapsed 363.733 s.
  `[SearchEngine volume] iterations=0 elapsed=5.3235408660000001 total-domain-bits=154261549.1440661 remaining-domain-bits=154261549.1440661 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.402702000000005 final-bound=0 first-incumbent-elapsed=28.27629566192627 samples=3`
  `[SearchEngine] New best cost: 80.4027 at iteration 0 source=root-heuristic-restricted event-epoch=1792734329.7909622 program-deadline=1792734326.014678`
- First incumbent: 28.276295662 s, cost 80.4027; best cost 80.402702.

### I622_c037_n46_regression
- In progress: 2026-10-23T01:50:35.229203-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I622_c037_n46_regression.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I622_c037_n46_regression.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I622_c037_n46_regression.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len 46 --ignore-checkpoints --run-dir logs/gemma_output_runs/I622_c037_n46_regression`
- Completed: 2026-10-23T01:56:20.117370-04:00; exit 0; runner elapsed 344.887 s.
  `[SearchEngine volume] iterations=0 elapsed=5.1050091660000003 total-domain-bits=150947892.70465142 remaining-domain-bits=150947892.70465142 log2-remaining-fraction=0 eliminated-bits=-0 remaining-nodes=1 exhausted=false incomplete-branches=0`
  `[Program gap] primal-dual-integral=30 normalized=1 final-gap=1 final-incumbent=80.895331999999996 final-bound=0 first-incumbent-elapsed=27.611914873123169 samples=3`
  `[SearchEngine] New best cost: 80.8953 at iteration 0 source=root-heuristic-restricted event-epoch=1792734692.3960319 program-deadline=1792734689.2841299`
- First incumbent: 27.611914873 s, cost 80.8953; best cost 80.895332.

### I623_c037_oracle
- In progress: 2026-10-23T01:56:20.193064-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I623_c037_oracle.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I623_c037_oracle.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I623_c037_oracle.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_planner_oracle`
- Completed: 2026-10-23T01:56:25.028154-04:00; exit 0; runner elapsed 4.834 s.
- Oracle verdict: 48/48 cases passed.

### I624_c037_search_gap
- In progress: 2026-10-23T01:56:25.098689-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I624_c037_search_gap.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I624_c037_search_gap.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I624_c037_search_gap.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python -m tests.test_search_gap`
- Completed: 2026-10-23T01:56:25.125201-04:00; exit 0; runner elapsed 0.025 s.

### I625_c037_native__private_layout
- In progress: 2026-10-23T01:56:25.192899-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I625_c037_native__private_layout.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I625_c037_native__private_layout.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I625_c037_native__private_layout.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --private-layout`
- Completed: 2026-10-23T01:56:25.390808-04:00; exit 0; runner elapsed 0.196 s.

### I626_c037_native__write_after_read_cache
- In progress: 2026-10-23T01:56:25.473769-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I626_c037_native__write_after_read_cache.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I626_c037_native__write_after_read_cache.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I626_c037_native__write_after_read_cache.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --write-after-read-cache`
- Completed: 2026-10-23T01:56:25.655707-04:00; exit 0; runner elapsed 0.180 s.

### I627_c037_native__incremental_propagators
- In progress: 2026-10-23T01:56:25.735486-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I627_c037_native__incremental_propagators.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I627_c037_native__incremental_propagators.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I627_c037_native__incremental_propagators.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --incremental-propagators`
- Completed: 2026-10-23T01:56:25.932329-04:00; exit 0; runner elapsed 0.194 s.

### I628_c037_native__start_rank_window
- In progress: 2026-10-23T01:56:26.013012-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I628_c037_native__start_rank_window.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I628_c037_native__start_rank_window.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I628_c037_native__start_rank_window.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --start-rank-window`
- Completed: 2026-10-23T01:56:26.225729-04:00; exit 0; runner elapsed 0.210 s.

### I629_c037_native__selection_reachability
- In progress: 2026-10-23T01:56:26.310584-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I629_c037_native__selection_reachability.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I629_c037_native__selection_reachability.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I629_c037_native__selection_reachability.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --selection-reachability`
- Completed: 2026-10-23T01:56:26.562970-04:00; exit 0; runner elapsed 0.251 s.

### I630_c037_native__relaxation_bound
- In progress: 2026-10-23T01:56:26.671175-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I630_c037_native__relaxation_bound.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I630_c037_native__relaxation_bound.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I630_c037_native__relaxation_bound.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --relaxation-bound`
- Completed: 2026-10-23T01:58:13.566022-04:00; exit 0; runner elapsed 106.893 s.

### I631_c037_native__prune_egraph
- In progress: 2026-10-23T01:58:13.638898-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I631_c037_native__prune_egraph.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I631_c037_native__prune_egraph.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I631_c037_native__prune_egraph.patch`
- Command: `/home/aaron/Projects/tensor_graphs/tensor_graphs_cpp/test --prune-egraph`
- Completed: 2026-10-23T01:58:13.866034-04:00; exit 0; runner elapsed 0.224 s.

### I632_c038_standin_build
- In progress: 2026-10-23T02:26:47.310339-04:00
- Metadata `logs/targets/prefill_decode_incumbent_seq_len/I632_c038_standin_build.json`; log `logs/targets/prefill_decode_incumbent_seq_len/I632_c038_standin_build.log`; patch `logs/targets/prefill_decode_incumbent_seq_len/I632_c038_standin_build.patch`
- Command: `/home/aaron/Projects/tensor_graphs/.venv/bin/python build.py --targets bindings test bench --log-level INFO`
- Completed: 2026-10-23T02:28:52.133095-04:00; exit 0; runner elapsed 124.821 s.
