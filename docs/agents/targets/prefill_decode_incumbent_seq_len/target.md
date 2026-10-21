# Reach a first prefill+decode plan for longer sequences within 30 seconds

Increase the sequence length N for which the planner compiles Gemma-3-270m's full set of prefill and decode buckets and produces a valid, correct plan (a first incumbent, extracted and written) inside a 30-second compile budget. Optimality is not gated: this target is about getting *any* sound plan for 2N+1 buckets quickly, so it rewards work whose cost per bucket is small and grows slowly with N: shared or parametric saturation, reuse across near-identical buckets, cheap reductions, and fast primal heuristics. The long-term goal is N = 32768 (65537 buckets; Gemma-3-270m's context length). The rules in `docs/agents/targets/README.md` apply.

## Objective

- **Metric:** N\*, the largest `--seq-len N` whose evaluation run passes (`[Gemma prefill+decode] ok=true`) under the 30 s budget, in a run with identical source and build.
- **Goal:** N = 32768. Keep raising N\* while there are plausible, sound ideas.
- **Tie-breaker (when N\* does not move):** the time to first incumbent at N\* (`[Program gap] first-incumbent-elapsed=`) in the 30 s run, lower is better. Example: the accepted implementation passes N = 3 with its first incumbent at 29 s and times out at N = 4; a change that still times out at N = 4 is kept only if its N = 3 run finds its first incumbent earlier.
- **Recorded, not gated:** final incumbent cost (`[Program gap] final-incumbent=`), compile elapsed, per-phase times (saturation, bucket reduction, root propagation, heuristic build, first validation), peak memory.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings test bench --log-level INFO
.venv/Scripts/python.exe tests/search_experiment.py --record prefill_decode_incumbent_seq_len:<machine> <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 30 --seq-len <N> --ignore-checkpoints --run-dir logs/gemma_output_runs/<run_id>
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

`<machine>` is `box` (Linux CUDA box, OpenCL off) or `laptop` (Windows arm64, OpenCL on); each machine has its own `progress_<machine>.md`, baseline and N\*. Results are only comparable on the same machine. Use `.venv/bin/python` on Linux.

The evaluator (`superviseLogits` in `tests/test_gemma_output.py`) runs three phases, as described in `docs/agents/targets/prefill_decode_seq_len/target.md`:

0. **Calibrate (unbudgeted).** Measures every cost configuration this N looks up that `benchmarks/records.bin` lacks.
1. **Compile (budgeted, 30 s).** Builds an `LLMSession` with `max_sequence_length=N` and the 2N+1 buckets of `prefillDecodeBuckets(N)` plus the full bucket. The search stops at `TG_SEARCH_DEADLINE = start + 30 - SEARCH_MARGIN_SECONDS`, keeping its incumbent; the plan is then extracted and written. Everything from process launch (imports, model loading, saturation, reduction, propagation, heuristics, search, extraction, plan writing) counts against the 30 s; the process is killed at 30 s. A compile that has no feasible plan at the deadline fails.
2. **Check (unbudgeted, exact).** A fresh process loads the plan (planning again fails the run) and runs every compiled graph, comparing each graph's needed logits rows against the reference. The check stays exact: every graph, no sampling.

- **Budget cap:** no run for this target, evaluation, gate or diagnostic, uses a compile budget above 30 s. A size that does not finish in 30 s is a miss; do not rerun it with a longer budget to see how far it got. Shorter budgets and phase timings from the compile log of a killed run are fine for diagnostics.
- One configuration per process, fresh `--run-dir`, no compiled-plan cache, `--ignore-checkpoints` on every recorded run, default build options, current hardware.
- **Fixed:** everything listed as fixed in `docs/agents/targets/prefill_decode_seq_len/target.md` (the supervisor and checks in `tests/test_gemma_output.py`, `tests/search_gap.py`, tolerances, `calibrate`, `pinPlatformEnvironment`, `OPENCL_BY_PLATFORM`, `bench.cpp`, saturation's equivalent-candidate cost filter, the oracle reference enumerator and existing oracle cases, the cost model and existing records, the bucket definitions), plus the 30 s budget and the exact check.
- **Agent-editable:** saturation, rewrite rules, cycle strip passes, bucket reduction, the planner, propagators, bounds, branching, primal heuristics, extraction, plan writing, kernels, and `SEARCH_MARGIN_SECONDS` (keep enough margin that extraction and plan writing finish inside the budget at the N being evaluated).

## Result lines

| Line | Meaning |
|---|---|
| `[Gemma calibrate] round=<k> pending=... benchmarked=... failed=...` | One line per calibration round |
| `[Gemma compile] supervisor elapsed=... budget=30 exit=... timed_out=...` | Compile process outcome |
| `[Program gap] ... final-incumbent=... first-incumbent-elapsed=...` | First incumbent time and final plan cost (gap values are recorded, not gated) |
| `[Gemma logits] graph=<g> dirty=[s,e) rows=[r0,r1) ok=... max_abs_err=...` | One line per compiled graph |
| `[Gemma prefill+decode] ok=... seq_len=N buckets=2N+1 compile_elapsed=...` | Summary; also written to `<run-dir>/results.json`. `ok=true` is a pass |

A run is **failed** if calibration cannot complete, the compile exceeds 30 s, crashes, has no feasible plan at the deadline or writes no plan, the gap samples are invalid, the check plans again or its graphs differ from `expectedGraphRanges(N)`, the check exceeds its timeout, or any logits row is outside tolerance.

## Finding N\*

The agent chooses which sizes to evaluate; there is no fixed ladder. Any order is fine (e.g. 1, 2, 3, 8, 10, 16, ...): jump ahead when phase timings say a size is clearly in reach, bisect when a jump fails. N\* only counts sizes with a recorded passing run. Record every run, including misses.

Start at N = 1. Before optimizing, record how each compile phase scales with N (from the compile logs of 30 s runs at a few small N, including killed ones), so the work goes to whatever grows fastest.

## Acceptance gate

1. **Correctness:** full oracle passes, `tests.test_search_gap` passes, native suites pass (`tensor_graphs_cpp/test` with `--private-layout`, `--write-after-read-cache`, `--incremental-propagators`, `--start-rank-window`, `--selection-reachability`, `--relaxation-bound`), and every evaluation run passes the logits check.
2. **Improvement**, one of:
   - **N\* rises:** a passing run at a larger N. Label it a *scaling* change.
   - **N\* stays, earlier incumbent:** the change's 30 s run at N\* passes with an earlier first incumbent than the accepted implementation's run at N\*. Label it a *speed* change.
3. **No regression:** a run at the accepted implementation's N\* still passes (for a speed change this is the same run as in 2). Plan quality is not gated: getting incumbents at large N comes first.

Because run variance is low, a single passing run is enough for a gate or tie-breaker reference. Keep the accepted implementation's reference run at N\* in the comparison table; re-measure when the build or hardware changes.

## Target-specific rules

- Report a budget miss honestly. Do not shrink, merge away or drop buckets, and do not prune saturation alternatives to fit the budget (the shared rule: saturation keeps every sound alternative).
- Work may be reordered (e.g. bound computations deferred until after the first incumbent) and the gap is not gated, but bounds must stay sound (shared exactness rules) and anything deferred must still respect the search deadline.
- Exploiting near-identical buckets is encouraged (sharing saturated or reduced e-graphs, plans or estimates across buckets that differ only in a size), but it must come from properties the planner can check (e.g. bucket specifications equal up to a sequence length), never from hard-coded bucket indices, and a shared result must be exactly what the bucket would have computed itself, or a sound superset, with the argument in the record.
- Calibration and the check are unbudgeted; a very slow calibration or check at large N is reported, not worked around by changing the evaluator, buckets, thresholds or the check.

## Starting points (verify before relying on them)

- `docs/agents/targets/prefill_decode_seq_len/progress_box.md` (Linux box, accepted B053): at N = 1 the first incumbent arrives at 11.3 s; at N = 2 (5 buckets) at 45.7 s: saturation ~12.6 s, saturation to search start ~19.5 s (bucket reduction: e-node infos, pruning, cycle passes), root propagation ~13 s (decode-bucket relaxations ~9 s each), heuristic build ~7 s, then validation. So N = 1 should pass and N = 2 fail at 30 s; confirm with the baseline.
- At 65537 buckets the 30 s budget leaves under 0.5 ms per bucket, so per-bucket saturation, reduction, relaxation and heuristic passes cannot reach the goal however fast each one is. Expect to need work proportional to the number of *distinct* bucket shapes (decode buckets `[t-1, t)` differ only in the KV length; prefill buckets `[0, t)` only in t), e.g. saturating and reducing one representative per family and instantiating the rest.
- Buckets with identical specifications already share their saturated and reduced e-graph (`identical_to` in `Planner::planAll`, B051).
- The root relaxation and its splits are only needed for the bound; the root primal heuristic currently uses relaxation region estimates and cheap from-scratch estimates (`cheapEstimates`) for its starts. At N = 2, cyclic decode components (one per layer) made region-based estimates expensive and initially produced invalid plans (see B038 to B042).
- Plan validation (`tryPlan`) checks a complete plan through all propagators; per-selection propagators were already given a whole-plan fast path (B010).
- The exact check runs every graph from a prefix, so its cost grows roughly with N²; watch its time against the check timeout as N grows.
