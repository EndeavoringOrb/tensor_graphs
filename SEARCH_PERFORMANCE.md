# Search performance log

## EmbeddingGemma 2 full text search

Measurements use `.venv/bin/python tests/embeddinggemma-2/run.py text --config full`
on the same workspace and machine. Wall time includes model setup, planning,
search, and execution.

| Stage | Wall time | Change |
| --- | ---: | --- |
| Baseline | 297.21 s | Original search implementation. |
| Optimization 1 | 278.45 s | Cache a completed selection-variable scan until the selection-domain revision changes. This avoids rescanning fixed selection variables while starts and offsets are assigned. |
| Optimization 2 | 283.87 s | Tried a single forward pass over sorted offset obstacles. The same output was produced, but the measured run was 5.42 s slower than optimization 1, so this change was removed. |
| Optimization 3 | 278.99 s | Tried incremental maintenance of the fixed-offset allocation index. The later profile run with this change took 270.12 s end to end (254.18 s search), but this does not isolate a clear gain from run-to-run variation. |
| Optimization 4 | 272.78 s | Added no-op guards for irrelevant memory-propagator inputs. Profile time stayed at 218.35 s, so this did not help materially. |
| Optimization 5 | 256.84 s | Check offset-domain bounds before alias/read analysis when neither allocation can be forced to move. Saved 15.94 s (5.8%) versus optimization 4; write-after-read time fell from 221.89 s to 201.14 s. |
| Optimization 6 | 264.37 s | Tried another bound check for unfixed offsets. It was 7.53 s slower than optimization 5, and write-after-read rose to 208.58 s, so the extra check was removed. |
| Optimization 7 | 94.25 s | Query ordered spatial indexes for fixed offset conflicts instead of scanning every temporal neighbor. The same numerical result was produced; search time was 78.46 s and write-after-read time fell to 40.81 s. |
| Optimization 8 | 90.56 s | Removed forward pruning against unfixed offsets; later assignments still check fixed pairs. Same numerical result, 3.69 s faster than optimization 7. |
| Optimization 9 | 74.93 s | Register the variable types each propagator handles, then skip unrelated calls. Search time fell to 59.06 s; output remained unchanged. |
| Optimization 10 | 62.88 s | Query the fixed-allocation index for possible spatial conflicts even when the changed offset is still a range. Write-after-read time fell from 36.81 s to 25.15 s; output remained unchanged. |
| Optimization 11 | >90 s (stopped) | Removed write-after-read pruning for unfixed offsets. Search generated more than 109,000 nodes without a solution (versus 25,243 nodes in optimization 10), so the change was reverted. |
| Optimization 12 | 57.70 s | Use the fixed-allocation index records directly instead of repeating selection, offset, and allocation lookups for each spatial candidate. Search fell from 51.26 s to 41.65 s; ranged-offset write-after-read fell from 28.79 s to 19.25 s. Output remained unchanged. |
| Optimization 13 | 49.32 s | Cache each class’s start bounds and use them in temporal overlap checks. Ranged-offset write-after-read fell to 11.23 s; output remained unchanged. |
| Optimization 14 | 49.45 s | Pass the changed start variable directly into input pruning and hoist its repeated bound lookup. No measurable gain. |
| Optimization 15 | 39.44 s | Coalesce duplicate pending variables in the propagation worklist. Search fell to 23.50 s with the same 25,243 nodes and output. |
| Optimization 16 | 39.00 s | Avoid copying input-prune selection domains unless a candidate is removed. No meaningful gain. |
| Optimization 17 | 39.55 s | Add scheduler guards for start propagators that already skip optional selections. No meaningful gain. |
| Optimization 18 | 34.23 s | Replace per-call input occurrence hash maps and sets with reusable counters and fixed small storage. Input-prune time fell from 7.45 s to 3.25 s; output remained unchanged. |
| Optimization 19 | 34.35 s | Skip consumer-precedence calls with no consumers. This data set rarely reached that case, so there was no measurable gain. |
| Optimization 20 | 34.02 s | Dispatch directly through propagator lists indexed by variable type. Propagation overhead fell from 3.36 s to 2.81 s. |
| Optimization 21 | 32.87 s | Cache canonical child classes and variable IDs for input-prune alternatives. Search fell to 16.97 s; output remained unchanged. |
| Optimization 22 | 32.67 s | Avoid repeating propagation-state initialization checks on changed starts. No meaningful gain. |
| Optimization 23 | 32.75 s | Lazily copy consumer selection domains only when removing candidates. No meaningful gain. |
| Optimization 24 | 29.10 s | Sample per-propagator timing on one call in sixteen and scale the duration estimate. Call counts remain exact; the profile times are estimates. Output remained unchanged. |

The final run finished search in 13.19 s and the full command in 29.10 s. It
produced `text: cosine=0.9999768` and `max_abs=0.0016230`. Profiled propagator
call counts are exact; per-propagator time totals are estimated from the one in
sixteen samples.

## EmbeddingGemma 2 full audio search

Build with `.venv/bin/python build.py --t bindings --profile --opencl 0`, then
run `.venv/bin/python tests/embeddinggemma-2/run.py audio --config full`.
Wall times below are for the audio command, including model setup, planning,
search, and execution. Build time is recorded separately because a changed C++
header forces a full bindings rebuild.

| Stage | Before | After | Search time / nodes | Change |
| --- | ---: | ---: | ---: | --- |
| Baseline | — | 121.39 s | Not captured / about 40,600 nodes | Original start branching and verbose profiling diagnostics. Audio output: cosine `0.9994730`, max absolute error `0.0046229`. |
| Trial (reverted) | 121.39 s | >90 s, stopped | 150,000 iterations without an incumbent | Split start ranges into halves. This delayed fixed-start conflict checks and expanded far more nodes, so the change was reverted. |
| Optimization 1 | 121.39 s | 107.59 s | 72.87 s / 58,301 nodes | Suppress per-conflict debug formatting and stream flushes in profiling builds; aggregate propagator contradiction counts and timings remain. Audio output matched the baseline. |
| Trial 2 (reverted) | 107.59 s | 109.95 s | 75.47 s / 58,301 nodes | Cache each class's common input children by selection domain. It was 2.36 s slower than optimization 1, and profiled input-prune time rose from 15.79 s to 16.95 s; reverted. |
| Optimization 2 | 107.59 s | 100.82 s | 65.96 s / 58,301 nodes | Run consumer-start precedence only when a start is fixed. This defers some pruning but cuts its profiled time from 16.68 s to 3.08 s; node count and audio output stayed the same. |
| Optimization 3 | 100.82 s | 98.93 s | 64.29 s / 58,301 nodes | Cache the fixed class-to-selection/start variable pairs as a contiguous list for start-unique checks. Start-unique time fell from 11.13 s to 9.98 s; output and node count stayed the same. |
| Optimization 4 | 98.93 s | 81.50 s | 46.64 s / 58,301 nodes | Run input-prune start precedence only when a required operation's start is fixed. Its profiled time fell from 15.95 s to 0.006 s; iterations, node count, and audio output stayed the same. |
| Trial 3 (reverted) | 81.50 s | 80.91 s | 46.30 s / 58,301 nodes | Deduplicate worklist additions once per changed variable rather than after each propagator. The overhead profile was unchanged (17.38 s versus 17.37 s), so the 0.59 s wall-time difference was not a reliable gain; reverted. |
| Trial 4 (reverted) | 81.50 s | 81.51 s | 46.61 s / 58,301 nodes | Avoid copying each start domain until after checking whether it contains the fixed value. Start-unique time was effectively unchanged (8.74 s versus 8.69 s); reverted. |
| Optimization 5 | 81.50 s | 80.34 s | 45.75 s / 58,301 nodes | Move start-unique's existing fixed-start and fixed-selection checks into the propagator dispatch guard. Calls fell from about 117 million to 15,319; the tree and audio output stayed the same. |
| Optimization 6 | 80.34 s | 79.78 s | 45.05 s / 58,301 nodes | Skip memory-no-overlap dispatch for start changes until that class's offset is fixed. Calls fell from about 117 million to 89,347, and sampled propagator time fell from 13.48 s to 9.70 s; output and tree stayed the same. |
| Optimization 7 | 79.78 s | 79.34 s | 44.63 s / 58,301 nodes | Compute each changed start's selection, start, offset, and consumer guard facts once and reuse them across propagators. Propagation overhead was similar (21.11 s versus 21.22 s); output and tree stayed the same. |
| Optimization 8 | 79.34 s | 72.86 s | 38.06 s / 58,301 nodes | Omit both cycle propagators when the planner has proved every bucket graph is a DAG. A selected subset cannot introduce a cycle; search and output stayed the same. |
| Trial 5 (reverted) | 72.86 s | 114.05 s | 79.18 s / 58,301 nodes | Index start domains by endpoint and contained value to visit only domains that can lose a fixed start. It preserved the result and tree, but start-unique time rose to 43.57 s; the candidate collection cost outweighed the skipped scans. |
| Optimization 9 | 72.86 s | 72.34 s | 37.36 s / 58,301 nodes | Cache the required start variables by selection-domain revision. Start-unique time fell from 9.36 s to 8.27 s; the same tree and audio output were preserved. |
| Trial 6 (reverted) | 72.34 s | 72.47 s | 37.75 s / 58,301 nodes | Check temporal separation before walking view chains in memory no-overlap. The wall time and memory-propagator time both increased slightly, so the change was removed. |
| Trial 7 (reverted) | 72.47 s | 78.37 s | 42.70 s / 58,301 nodes | Cache view relationships by class pair and selection revision. Hash lookups cost more than the repeated view checks; memory-no-overlap time rose to 13.38 s. |
| Optimization 10 | 72.34 s | 70.08 s | 35.62 s / 58,301 nodes | Add a consumer-precedence dispatch guard for fixed starts. Calls fell from 117.3 million to 15,318 and the propagator time fell from 2.52 s to 0.055 s; the tree and output stayed the same. |
| Trial 8 (reverted) | 70.08 s | 123.93 s | 89.30 s / 58,301 nodes | Scan precomputed temporal neighbors for ranged offsets and look up fixed allocations by class. Those lists were broader than the spatial range query; memory-no-overlap time rose to 46.98 s. |
| Optimization 11 | 70.08 s | 69.53 s | 34.55 s / 58,301 nodes | Dispatch write-after-read-start only for fixed, required starts. Calls fell from 117.4 million to 15,319, and its profile time fell from 4.79 s to 2.47 s; output and tree stayed the same. |
| Trial 9 (reverted) | 69.53 s | 69.36 s | 34.89 s / 58,301 nodes | Require a fixed positive selection as well as a fixed start before dispatching write-after-read-start. Calls and profile time were unchanged, so this stricter guard was removed. |
| Optimization 12 | 69.53 s | 68.82 s | 34.04 s / 58,301 nodes | Build Euler intervals for the selected view-parent forest, so memory conflict checks can answer view-ancestor questions in constant time. Memory-no-overlap time fell from 9.60 s to 9.28 s; output and tree stayed the same. |
| Trial 10 (reverted) | 68.82 s | 110.54 s | 75.60 s / 58,301 nodes | Walk endpoint and small-domain start-index buckets directly instead of copying a combined candidate list. Start-unique time still rose to 44.03 s, so the index was removed. |
| Trial 11 (reverted) | 68.82 s | 69.12 s | 34.26 s / 58,301 nodes | Skip write-after-read-start when fewer than two offsets are fixed. The call still takes about 2.48 s overall and total time was unchanged; the extra guard was removed. |
| Trial 12 (reverted) | 68.82 s | 69.53 s | 34.93 s / 58,301 nodes | Maintain ordered sets of unresolved selection variables instead of scanning all classes for the smallest domain. The extra set maintenance made the measured run 0.71 s slower; reverted. |
| Trial 13 (reverted) | 68.82 s | 69.08 s | 34.47 s / 58,301 nodes | Cache selection variable IDs in their original scan order to avoid repeated class-map lookups. The branch and audio times did not improve; reverted. |
| Optimization 13 | 68.82 s | 65.42 s (repeat: 66.34 s) | 30.75 s / 58,301 nodes | Index fixed allocation intervals in a treap augmented with each subtree's maximum end. Ranged-offset checks can skip allocations that cannot overlap; memory-no-overlap fell from 9.28 s to 7.07 s. Both runs preserved the search tree and audio output. |
| Optimization 14 | 65.42 s | 65.14 s (repeat: 65.20 s) | 30.52 s / 58,301 nodes | Use the same interval tree for fixed-offset checks, querying only intervals that overlap the current allocation. The fixed-offset profile fell from 1.79 s to 0.18 s; tree and audio output stayed the same. |
| Trial 15 (reverted) | 65.14 s | 65.30 s | 30.75 s / 58,301 nodes | Skip copying a wide start range when the fixed value is interior and cannot be removed. Start-unique time was unchanged, so this special case was removed. |
| Trial 16 (reverted) | 65.14 s | 65.26 s | 30.51 s / 58,301 nodes | Sort the cached required-start list by variable ID to improve memory locality. Start-unique time stayed near 8.2 s, so sorting was removed. |
| Trial 17 (reverted) | 65.20 s | 65.48 s | 31.00 s / 58,301 nodes | Stop the smallest-domain selection scan at the first size-two domain. This did not lower branch or total time, so the early exit was removed. |
| Optimization 15 | 361.0 s | 225.0 s | 134.66 s / 102,381 nodes | Add Pass 4 cycle reduction to remove dead-end 2-cycle enodes between memory spaces where one class has no external consumers. This eliminated all remaining 637 cyclic components, proved bucket 0 is a DAG, and allowed omitting both cycle propagators (saving ~120 s in propagation). Audio output: cosine `0.9995105`, max_abs `0.0041135`. |
| Optimization 16 | 225.0 s | 216.0 s | 126.14 s / 102,381 nodes | Avoid eager domain copying and sort active start variables in StartUniquePropagator; guard WriteAfterReadStartPropagator against scanning classes when fewer than 2 offsets are fixed, and iterate over fixed offset index directly. Profiled StartUnique time fell from 38.83 s to 32.14 s, and WriteAfterReadStart fell from 24.68 s to 19.38 s. Output: cosine `0.9995105`, max_abs `0.0041135`. |
| Optimization 17 | 216.0 s | 199.0 s | 111.79 s / 102,381 nodes | Pre-index static class lookups (page sizes, mem_spaces, and var IDs) into flat contiguous vectors; precalculate and sort preallocated obstacles per memory space; merge offset obstacles into disjoint intervals for linear scan; and add size-2 early exit to chooseSelection. Branch time dropped from 34.91 s to 19.94 s (preferred-offset fell from 27.29 s to 16.94 s, selection from 5.30 s to 2.43 s, sched-start from 1.81 s to 0.43 s). Output: cosine `0.9995105`, max_abs `0.0041135`. |
| Optimization 18 | 199.0 s | 182.72 s | 111.55 s / 100,975 nodes | Optimize e-graph saturation and rebuild: add visited sets to RemoveContiguous and RemoveRedundantReshape (preserving full sweeps in FusionRule so multi-depth fusions are fully observed when children change); avoid eager child vector copies in rebuild; cache constant data byte hashes; run congruence closure to fixpoint inside rebuild; and early-exit saturation when 0 rules match in an iteration. Saturation time dropped from 49.20 s to 34.01 s (iterations fell from 4 to 2). Discovered plan execution cost improved from 2365.43 ms to 2244.8 ms (all 206,355 fusion applications preserved). Output: cosine `0.9995105`, max_abs `0.0041135`. |
| Optimization 19 | 182.72 s | 174.00 s | 106.12 s / 100,937 nodes | Partition FusionRule patterns into single-node (all pattern inputs are leaf variables/constants) and multi-node patterns. Apply a visited set exclusively to single-node patterns so e-nodes are evaluated against single-node patterns only once, while continuing to evaluate multi-node patterns on every sweep so multi-depth fusions are fully observed when child classes change. Saturation time fell from 34.01 s to 31.90 s (eliminating all redundant single-node re-matches in sweep 2), search time dropped from 111.55 s to 106.12 s (100,937 nodes), and total wall time dropped to 174.00 s. Audio output: cosine `0.9995105`, max_abs `0.0041137`. |

The first optimization cut 13.80 s (11.4%) from the end-to-end audio run. The
profile reports about 117 million calls each to input-prune, consumer-start,
start-unique, memory-no-overlap, and write-after-read-start propagators. Memory
no-overlap spent 10.13 s in write-after-read checks, while cycle avoidance spent
6.15 s. Per-propagator time totals are estimates sampled once per sixteen calls;
call counts are exact.

A repeat using the retained source completed in 107.92 s (73.21 s in search,
58,301 nodes) and produced the same cosine and maximum absolute error. The
most recent profile build, after optimization 14, took 98.94 s to recompile
the changed binding. Earlier builds took 98.27 s for a changed binding, 1.63 s
for an unchanged build, and 0.72 s for a cached build.

With optimization 2 enabled, the exact requested build and audio commands both
succeeded. The audio output remained `cosine=0.9994730`, `max_abs=0.0046229`.
Optimization 3 also preserved that output.
Optimization 4 preserved it as well and reduced the full audio wall time to
81.50 s.
Optimization 5 preserved it and measured 80.34 s.
Optimization 6 preserved it and measured 79.78 s.
Optimization 7 preserved it and measured 79.34 s.
Optimization 8 preserved it and measured 72.86 s; the log confirmed that all
buckets were acyclic.
Optimization 9 preserved it and measured 72.34 s.
Optimization 10 preserved it and measured 70.08 s.
Optimization 11 preserved it and measured 69.53 s.
Optimization 12 preserved it and measured 68.82 s with the conservative
write-after-read-start dispatch guard.
Optimization 13 preserved it and measured 65.42 s; a repeat measured 66.34 s.
Optimization 14 preserved it and measured 65.14 s; a repeat measured 65.20 s.
Optimization 15 reduced the full audio wall time from 361.0 s to 225.0 s (search time from 270.87 s to 134.66 s) by removing 637 residual cross-space copy-cycle enodes, enabling DAG reachability and eliminating CycleAvoidancePropagator and PearceKellyCyclePropagator. Audio output remained `cosine=0.9995105`, `max_abs=0.0041135`.
Optimization 16 reduced the full audio wall time from 225.0 s to 216.0 s (search time from 134.66 s to 126.14 s) by avoiding eager domain copies and sorting active starts in StartUniquePropagator, and guarding WriteAfterReadStartPropagator against scanning classes when fewer than 2 offsets are fixed. Brancher profile instrumentation revealed that preferredOffset takes 27.29 s out of 34.90 s in branch time. Audio output remained `cosine=0.9995105`, `max_abs=0.0041135`.
Optimization 17 reduced the full audio wall time from 216.0 s to 199.0 s (search time from 126.14 s to 111.79 s) by pre-indexing static class lookups into flat contiguous vectors, precalculating sorted preallocated obstacles per memory space, merging offset obstacles into disjoint intervals for single-pass scanning, and adding size-2 early exit to chooseSelection. Branch time dropped from 34.91 s to 19.94 s. Audio output remained `cosine=0.9995105`, `max_abs=0.0041135`.
Optimization 18 reduced the full audio wall time from 199.0 s to 182.72 s (search time from 111.79 s to 111.55 s, nodes 102,381 to 100,975) by optimizing e-graph saturation and rebuild. Visited sets were added to `RemoveContiguous` and `RemoveRedundantReshape` (leaving `FusionRule` unrestricted across sweeps so multi-depth fusions are fully observed when child classes change); `EGraph::rebuild` avoids copying child vectors for nodes whose children are already canonical, caches constant byte hashes to avoid re-reading gigabytes of raw weights, and iterates congruence closure to fixpoint inside `rebuild` rather than through redundant outer saturation loops. Finally, saturation early-exits immediately when 0 rewrite rules match in an iteration. Rewrite saturation time dropped from 49.20 s to 34.01 s, and iterations fell from 4 to 2. The discovered plan execution cost improved from 2365.43 ms to 2244.8 ms (all 206,355 fusion applications preserved). Audio output remained `cosine=0.9995105`, `max_abs=0.0041135`.
Optimization 19 reduced the full audio wall time from 182.72 s to 174.00 s (search time from 111.55 s to 106.12 s, nodes 100,975 to 100,937). `FusionRule` patterns are partitioned into single-node patterns (where all pattern inputs are leaf variables/constants) and multi-node patterns. A visited set is applied exclusively to single-node patterns so e-nodes are evaluated against single-node patterns only once, while continuing to evaluate multi-node patterns on every sweep so multi-depth fusions are fully observed when child classes change. Rewrite saturation time dropped from 34.01 s to 31.90 s (eliminating all redundant single-node re-matches in sweep 2), and search time fell to 106.12 s. Audio output remained `cosine=0.9995105`, `max_abs=0.0041137`.


