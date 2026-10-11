# KernelBench iteration-speed plan

## Evidence and scope

Reviewed the production journal capture at `/tmp/kernelbench-review-journal.log` (34,321 lines, October 11, 2026, approximately 00:00–10:51 EDT), the production review in `docs/reviews/kernelbench-production-2026-10-11.md`, and the current build, worker, promotion, and Gemma validation code. This uses the existing production capture, not a new live production inspection. Current source includes improvements made after some of these log entries; distinguish observed historical waste from work still needed. No production changes or benchmarks were performed.

Matched `Starting ... (timeout ...)` to `Finished in ...` by service Python PID, using the service's serial command execution. Durations below are rounded subprocess wall times, not exclusive CPU/GPU utilization, queue latency, or a complete accounting of service uptime.

| Stage | Completed commands | Total time | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Full goal runner | 36 | 16,336 s (4.54 h) | 469.7 s | 489.3 s |
| Native runner, all actions | 1,122 | 4,236 s (1.18 h) | 0.2 s | 201.5 s |
| Promotion bindings build | 36 | 3,594 s (1.00 h) | 134.2 s | 137.9 s |
| Kernel benchmark bindings build | 47 | 2,796 s (0.78 h) | 58.8 s | 92.7 s |
| Explicit lint commands | 38 | 7.3 s | 0.2 s | 0.6 s |
| Git commands | 89 | 8.8 s | 0.0 s | 1.4 s |

Full goal runs and builds account for about 84% of these recorded command durations. The 0.2-second native median hides expensive discovery/evaluation outliers; split native actions before deciding whether process startup deserves attention. Git cloning itself and lint are not the major costs.

Representative promotion, journal lines 33508–34261:

- Baseline native build: 134.9 s. Stable CUDA compile occupies approximately 22 s, followed by approximately 112 s for the main bindings object; linking is short. CUDA and target C++ compilation currently run sequentially.
- Baseline goal: 468.4 s. It tries a nonexistent job-local cache, then plans from scratch. Planning starts at 10:05:56; full saturation finishes at 10:06:00 and the logged partial saturation phases finish around 10:06:02; memory materialization starts at 10:12:49. The approximately 407-second gap warrants profiling cost evaluation, pruning, extraction, and graph construction. The logs do not prove which internal phase dominates.
- Candidate native build: 12.7 s, reusing stable CUDA and `bindings.o` for this CUDA candidate.
- Candidate goal: 461.2 s, again using a fresh cache path.
- Combined measured build/goal time: 1,077.2 s, approximately 18 minutes before queueing and other overhead.

The journal has 32 stable CUDA compilation messages, 51 stable CUDA cache hits, and only 16 target-object cache-hit messages. Existing incremental caching works, but workspace lifetimes and broad compilation dependencies limit its benefit.

## 1. Measure phases and make the existing early exits effective

**First implementation: `kernel_bench/jobs.py`, `kernel_bench/promotion.py`, `build.py`, and the goal runner/native planner.**

Persist structured per-job timings for queue wait, source preparation, lint, code generation, individual compilations, linking, native load, discovery, planning, reference generation, correctness, warmup, timing, and artifact writes. Record cache identity, hit/miss and reason, source/base revision, native action, selected candidate instructions, and peak memory. Add timings inside the long planning gap, including per-bucket cost evaluation, pruning, extraction, and graph construction. Keep detailed compiler output as artifacts and emit concise stage summaries.

The current `promotion.py` already supports `planning_command` and exits before full measurements when the candidate is unselected. Both shipped promotion JSON recipes already provide it; verify the active operator-owned production configuration also does. The historical review found 10 of 17 candidate promotions unselected. Those are a substantial avoidable workload, but do not count all of their runtime as guaranteed savings: selection preflight itself still compiles and plans.

Expose existing executed-workload information to candidate authors so they can prioritize configurations that contribute to positive-weight buckets. Validate registration arity and immutable configuration data before expensive compilation where practical. Preserve current baseline interface checks and batch infrastructure-error handling; add a real post-build export/version preflight so a stale binary cannot fail hundreds of configurations.

**Acceptance:** stage totals reconcile with job elapsed time; an unselected candidate never reaches full correctness/timing; invalid submissions and incompatible native modules fail with an actionable stage and reason.

## 2. Persist and share builds across jobs and service restarts

**Highest-confidence build improvement: replace disposable build lifetimes with immutable cached variants.**

Currently candidate evaluation reuses one goal workspace within a worker's `TemporaryDirectory`, baseline calibration uses another workspace, startup discovery uses a temporary workspace, and every promotion clones into a new temporary directory. Evaluation cache reuse therefore disappears on restart, and promotion repeatedly pays for the same baseline native objects.

Create a service-owned cache outside the Git checkout, with separate build entries for immutable discovery snapshots and current promotion integration revisions. Keep job outputs and writable staging separate. Build a baseline once, then derive candidate variants using copy-on-write/reflink where available or copied files; never let a candidate mutate shared baseline files. The promotion code can switch from the archived discovery commit to current deployed HEAD: key and verify the actual integration revision, not just `goal_id`.

Cache keys must include source/dependency content, generated source identity, build/linter schema, compiler executable identity/version, flags and target CPU features, CUDA toolkit/architecture, Python ABI and pybind11 headers, enabled backends, and native interface version. Hardware and driver identity additionally belong in execution/measurement cache keys. Preserve DEBUG versus WARNING log-level variants initially: these are currently different compile commands, and unifying flags should be a separately validated change.

Current cache entries use absolute object paths and command strings, dependency mtimes, and link keys containing resolved paths. Copying `.build_cache.json` into a different directory will not provide reliable hits, and snapshot extraction resets mtimes. Prefer stable build directories plus content-based manifests; if copying entries, implement and test dependency/path normalization explicitly. Validate all dependency content, including removed files and toolchain/system headers. Missing or malformed dependency metadata must trigger a rebuild with a clear reason.

Publish completed entries atomically after native load/export validation; use per-key locks, bounded disk retention, and pin entries while jobs use them. Cache baseline lint by source and linter identity rather than an unqualified marker file. Do not reuse a result solely because a directory or `.baseline_built` marker exists.

**Acceptance:** a second promotion on the same integration base reuses baseline objects; restarting the worker preserves reuse; changes to a dependency, flags, compiler, Python ABI, or interface cause the appropriate miss. Aim to eliminate the approximately 135-second cold baseline build for a warm entry, then measure actual end-to-end savings.

## 3. Reuse compiled goal plans, including selection-preflight output

**Largest observed opportunity: stop rebuilding the same execution plans.**

`promotion.py::measure` uses a new `{label}.dirty-cache` for each run. The current selection path builds/plans the candidate, removes it, builds/plans baseline, then re-adds and builds/plans candidate again. Borderline paired runs toggle the candidate file and invoke `measure` again with new cache names, potentially adding four more builds/plans. Paired timing is valuable; repeating compilation and planning is not required to obtain fresh samples.

Create two stable variants per promotion: baseline and candidate. Compile each native binary once and compile each execution plan once. Run preflight against the candidate variant and carry that exact compiled cache forward to candidate correctness and timing. Reuse the baseline plan across jobs when its complete identity matches. Each subsequent paired round should load the corresponding binary/cache in a fresh subprocess and perform fresh execution measurements, without toggling source files or rebuilding.

A compiled-plan key needs the native source/registry identity, complete logical graph and constant/model content identity, root, bucket regions and weights, planner settings/version, memory topology, hardware, and exact effective cost corpus. `compiler_records` alone is insufficient: promotion supplements it with database measurements and filters candidate rows differently for baseline and candidate. Hash the effective remapped records for each variant. New cost observations can change selection even when kernel source is unchanged.

Existing `Session.loadCache` validates cache version, root and kernels, and `Session.compile` checks buckets/weights; inspect and extend these checks for settings and cost-corpus provenance before relying on shared caches. It can truncate an invalid cache, so give subprocesses writable job-local copies of immutable shared entries. Never seed a candidate with a baseline plan unchanged: retaining the baseline plan could conceal a newly selectable kernel. Candidate planning must consider the candidate registry and costs.

Keep reference correctness evidence and compiled plans reusable, but obtain fresh baseline/candidate timing samples for acceptance. Old speedup results may guide exploration; they cannot substitute for current paired evidence near the 1% gate. Do not skip zero-weight bucket correctness merely because selection/timing contribution is zero.

**Acceptance:** candidate preflight and full candidate validation report the same selection and plan identity; paired rounds show cache hits and no build/planning phase; changing weights, graph, constants, costs, topology or planner settings invalidates reuse. Compare cold and warm selected kernels, correctness, and weighted timing within expected measurement noise.

## 4. Reduce compilation scope and schedule independent compilations together

After persistent caching, improve genuinely changed builds in `build.py` and native registration layout:

1. Preserve the existing stable/generated CUDA split. Start independent CUDA and C++ object compilations together using a bounded executor; link only once all required objects succeed. The representative cold build suggests an upper-bound saving around the shorter 22-second compile, before contention. Measure memory pressure and wall time rather than assuming more parallel tasks are faster.
2. Split stable CPU registrations from submitted CPU kernels. CPU kernels currently enter a generated header included by the main target, so a CPU candidate can invalidate a large bindings translation unit. Move registration into separately compiled translation units with explicit registration entry points as needed. Audit static-initializer retention and duplicate registration when linking.
3. Separate heavyweight pybind/core code from candidate registration so one candidate changes only its object plus link. Consider bounded stable CPU/CUDA shards if stable source edits remain expensive. Cache both baseline and candidate objects so paired variants do not thrash one object pathname.
4. Evaluate a compiler content cache such as ccache/sccache only after path and toolchain keys are correct; measure actual C++ and nvcc support/hit rates on this installation. Consider a precompiled header for stable core headers if translation-unit splitting still leaves expensive parsing. Avoid introducing several cache systems at once without attribution.

Keep optimized release flags for performance measurements. A fast compile/check mode can provide preliminary feedback but must not qualify a candidate for promotion. Whole-file capability guards and distinct threaded/unthreaded kernels remain required; build improvements must not introduce runtime fallback branches.

**Acceptance:** changing one CPU or CUDA candidate compiles its own translation unit and links, without recompiling stable kernels or heavyweight bindings. No-op builds execute no compiler/linker. Verify registrations, UIDs, capabilities and correctness against the existing build.

## 5. Cache deterministic preparation and investigate remaining planner time

Cache independent reference logits keyed by model file content, tokenizer/configuration, fixed token inputs, dtype, explicit RoPE semantics, reference implementation and library versions. The runner regenerates these in every full run, but representative model-weight loading is only about 0.1 s; reference caching is secondary to planning, not a justification for a persistent heavyweight process yet.

Use the phase measurements from step 1 to optimize the approximately 407-second planning gap. If reusable compiled plans still miss frequently, consider caching the saturated graph separately from kernel matching/cost extraction. Key saturation by graph/constants, rewrites and settings; invalidate if candidate reference expressions add rewrite semantics. Recompute candidate matching, costs and selection. `session.hpp` already shares full-bucket saturation with partial buckets, so preserve that existing reuse rather than proposing it as new work.

Profile repeated equivalent shape/stride/device cost queries and memoize pure computations with complete keys. Determine whether time is spent in cost lookup, search/extraction, or graph bufferization before changing algorithms or `--min-compile-time`. Do not blindly lower the planning budget: a cheaper plan that chooses worse kernels changes both validation and production performance.

Classify malformed constant datasets once per immutable goal and stop recalibrating/retrying unchanged invalid inputs. The capture contains 164 invalid-constant messages and an earlier 330-error native-interface incident. Repair goal data and harness semantics, then create corrected snapshots; do not cache bad correctness answers as successful evidence.

**Acceptance:** warm reference preparation avoids model reference execution; its identity changes invalidate it. Planner optimizations preserve logical definition, executable coverage and correctness, with selection/performance differences explicitly measured.

## 6. Improve queue latency without contaminating measurements

The worker currently serializes builds and measurements under a shared database lock. Keep that behavior initially. Once cache wins are measured, evaluate preparing/linting the next job while the current job runs. Full native compilation can compete for CPU and memory with CPU or mixed CPU/GPU timing, so overlap compilation with benchmarks only if controlled measurements establish acceptable interference; otherwise schedule it outside timed sections.

Deduplicate identical in-flight build and deterministic validation work using the full source/goal/configuration identity, retaining per-submission events. Make discovery/calibration resumable and avoid rebuilding native-equivalent snapshots for UI-only changes; do not erase logical-goal identity or source provenance to obtain reuse. Expose expected queue and stage durations to authors.

Keep fresh native subprocesses for crash isolation. The 0.2-second native median and 7.3-second total lint cost do not support prioritizing a permanent interpreter or skipping lint. Skip unrelated OpenCL generation only as a small cleanup after the major work.

## Delivery order and verification

1. Add stage timings and verify deployed selection-preflight configuration and infrastructure early exits.
2. Add persistent immutable baseline/candidate build variants and cache integrity checks.
3. Reuse candidate preflight plans, baseline plans, and both variants during fresh paired measurements.
4. Parallelize independent object compilation, then isolate CPU candidate registrations.
5. Profile remaining planning misses; add deterministic reference/saturation/query caches where measured benefit supports them.
6. Consider queue scheduling changes after measurement isolation is established.

Validate with representative selected CPU and CUDA candidates, an unselected candidate, an identical resubmission, a worker restart, a native interface/dependency change, a cost-corpus change, and a borderline paired promotion. Test cache corruption, concurrent requests for one key, and interruption during publication. Run the relevant existing build-cache, KernelBench contract/promotion/deployment tests and native correctness checks for implemented changes; no new tests are needed for this documentation-only plan.

Report cold/warm p50 and p95 submission-to-result latency, queue wait, build and planning duration, cache hit rate and miss reasons, unselected early exits, disk usage, and paired timing dispersion. Keep benchmark settings and workload fixed for comparisons. Expected gains are hypotheses until measured: prioritize eliminating repeated approximately 135-second baseline builds and approximately 470-second goal preparation runs; do not add these savings twice when preflight and plan reuse eliminate the same work.
