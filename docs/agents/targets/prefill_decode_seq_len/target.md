# Plan longer prefill+decode sequences to within 5% of optimal in 120 seconds

Increase the sequence length N for which the planner compiles Gemma-3-270m's full set of prefill and decode buckets and proves its plan within 5% of optimal, inside a 120-second compile budget. The bucket count grows as 2N+1, so this rewards work that scales with the number of buckets: shared saturation, cycle handling, bounds that stay valid across near-identical buckets, and fast first incumbents. The rules in `docs/agents/targets/README.md` apply.

## Objective

- **Metric:** N\*, the largest `--seq-len N` whose run passes (`ok=true`) with final primal-dual gap below 0.05 (`gap-target-met=true`), in a run with identical source and build.
- **Goal:** no fixed goal. Keep raising N\* while there are plausible, sound ideas.
- **Frontier size N_f:** the smallest size on the search ladder (the sizes tried in "Finding N\*" below) that the accepted implementation does not meet. While even N = 0 misses the target, N\* is undefined and N_f = 0.
- **Progress metric (when N\* does not move):** the `normalized` primal-dual integral at N_f (`[Program gap] normalized=`), lower is better. It is smoother than the final gap, and it rewards both earlier incumbents (the gap is 1 until the first one) and faster bounds. The final gap at N_f breaks ties.
- **Recorded, not gated:** eliminated domain bits (`[SearchEngine volume]`), time to first incumbent, compile elapsed time. Volume is not comparable across candidates here, because saturation rules and cycle stripping change how many domain bits exist and the total grows with N. The integral already counts time to first incumbent.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO
.venv/Scripts/python.exe tests/search_experiment.py --record prefill_decode_seq_len <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --run-dir logs/gemma_output_runs/<run_id>
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

`tests.test_gemma_output --config prefill+decode --timeout 120` (`superviseLogits`) runs three phases, each in its own process:

0. **Calibrate (unbudgeted).** Repeats rounds of a `saturate_only` session for the same N (cost-call logging on, written to `<run-dir>/calibrate<k>_calls.bin` through `TG_COST_CALLS_PATH`) followed by `tensor_graphs_cpp/bench(.exe) --calls <that file>`, which measures every logged configuration that `benchmarks/records.bin` lacks and appends it. Exception: `bench` skips reference kernels whose estimated time exceeds `--max-ref-estimate-ms` (default 10000 ms; calibration uses the default). They are not counted as pending, get no record, and the planner and saturation use their interpolated estimate. It stops when a round finds nothing pending, so every other cost lookup the compile makes has an exact record. New records can change which fused candidates saturation keeps, which is why it repeats. Saturation's equivalent-candidate filter compares exact or interpolated costs, so a skipped reference kernel is still dropped when an interchangeable kernel is cheaper; only candidates whose kernel has no records at all are kept unconditionally. It fails if a round cannot benchmark any of its pending configurations, or after 5 rounds. Logs: `<run-dir>/calibrate<k>.log` and `calibrate<k>_bench.log`, plus `[Gemma calibrate] round=... pending=... benchmarked=...` lines.
1. **Compile (budgeted).** Cost-call logging is off. Builds an `LLMSession` with `max_sequence_length=N` and the token buckets of `prefillDecodeBuckets(N)`: prefill `[0, t)` and decode `[t-1, t)`, both needing logits row `t-1`, for `t = 1..N`. The session adds the full bucket, giving 2N+1 buckets (prefill 1 and decode 1 coincide, as in `tests/verify_saturation.py`). **N = 0** is the special case that compiles only the full bucket of a one-token sequence (1 graph). `expectedGraphRanges(N)` defines the exact set. The search runs with `--optimal` (until exhausted) and `TG_SEARCH_GAP=1`, and stops at `TG_SEARCH_DEADLINE = start + 120 - SEARCH_MARGIN_SECONDS`, keeping its incumbent. The plan is then extracted and written to the fresh run directory. Everything (process start, imports, model loading, saturation, search, extraction, plan writing) counts against the 120 s from launching the process; the process is killed at 120 s.
2. **Check (unbudgeted).** A fresh process opens the session from that plan. A run that plans again fails. It runs **every compiled graph directly** (`LLMSession.run_graph_logits`, bypassing bucket selection): for a graph with dirty tokens `[s, e)` it first runs the full graph on `S[:s]` (zero padded) to set up the persistent state, then the graph itself on `S[:e]`. It saves the graph's needed logits rows.

- One configuration per process, no compiled-plan cache from earlier runs (every run uses a fresh `--run-dir`), current hardware, default build options.
- **Per-machine settings:** `OPENCL_BY_PLATFORM` in `tests/test_gemma_output.py` pins `TG_DISABLE_OPENCL` for every phase. OpenCL is on for the Windows arm64 laptop and off for the Linux CUDA box. Results are only comparable on the same machine: keep a separate baseline, comparison table and N\* per machine in `progress.md`.
- **Cost records:** `benchmarks/records.bin` only grows, by calibration measuring configurations it lacks. Existing records are never edited or removed. A candidate that makes the planner look up new configurations gets them measured before its timed compile, so it is not judged on interpolated costs (except reference kernels skipped by the estimate threshold). Calibration time is not part of the budget.
- **Caching & Checkpoints Policy:** Adding caches or checkpoints (such as serialized `RelaxationBucket` dumps, e-graph checkpoints, propagator caches, or evaluation shortcuts) is explicitly permitted and encouraged for rapid local experimentation and development loops. However, **the benchmarked final run cannot use any precomputed cache or checkpoints**. Official benchmark runs (`tests/search_experiment.py --record ...`) must always execute with `--ignore-checkpoints` / `TG_IGNORE_CHECKPOINTS=1` (and fresh `--run-dir`), verifying that the entire compilation, saturation, and search execute cleanly from scratch within the 120 s budget without relying on offline artifacts.
- **Fixed:** the supervisor and checks in `tests/test_gemma_output.py` (`superviseLogits`, `runEveryGraph`, `checkLogits`, `prefillDecodeBuckets`, `expectedGraphRanges`, `referenceTokens`, `referenceLogits`), `tests/search_gap.py`, the tolerances in `tests/inference_benchmark.py`, `calibrate`, `pinPlatformEnvironment`, `OPENCL_BY_PLATFORM`, `tensor_graphs_cpp/bench.cpp` (including the `--max-ref-estimate-ms` default), saturation's equivalent-candidate cost filter (`rewrite.hpp`), the oracle reference enumerator and existing oracle cases, the cost model and existing records, and the bucket definitions. Changing any of them needs separate review and a new baseline.
- **Agent-editable:** saturation, rewrite rules, cycle strip passes, the planner, propagators, bounds, branching, primal heuristics, extraction, kernels, and `SEARCH_MARGIN_SECONDS` in `tests/test_gemma_output.py` (the time kept after the search for extraction and plan writing, defaulting to 3.0 s; the agent can fine-tune this margin: e.g. if extraction only needs ~1 s, shrinking the margin recovers wasted seconds and gives more time to search, provided the 120 s compile budget is respected).

## Result lines

| Line | Meaning |
|---|---|
| `[Gemma calibrate] round=<k> pending=... benchmarked=... failed=...` | One line per calibration round |
| `[Gemma compile] supervisor elapsed=... budget=120 exit=... timed_out=...` | Compile process outcome |
| `[Program gap] ... final-gap=... normalized=...` | Primal-dual gap of the compile's search samples over [0, 120 s] (`docs/core/search_gap.md`) |
| `[Gemma logits] graph=<g> dirty=[s,e) rows=[r0,r1) ok=... max_abs_err=...` | One line per compiled graph |
| `[Gemma prefill+decode] ok=... seq_len=N buckets=2N+1 final-gap=... gap-target-met=...` | Summary; also written to `<run-dir>/results.json` |

A run is **failed** if:
- calibration cannot complete (a configuration that cannot be benchmarked, or more than 5 rounds);
- the compile exceeds the budget, crashes, finds no feasible plan before the search deadline, or writes no plan;
- the gap samples are invalid (bound above incumbent, bound decreasing, incumbent increasing);
- the check plans again, or its compiled graphs differ from `expectedGraphRanges(N)`;
- any logits row is outside tolerance.

A passing run with `final-gap >= 0.05` does not meet the target for that N.

## Acceptance gate

1. **Correctness:** full oracle passes, `tests.test_search_gap` passes, native suites pass (`tensor_graphs_cpp/test.exe` with `--private-layout`, `--write-after-read-cache`, `--incremental-propagators`, `--start-rank-window`), every evaluation run passes the logits check, and the official benchmarked run passes with `--ignore-checkpoints` (recomputing from scratch without offline caches).
2. **Improvement**, one of:
   - **N\* rises:** a passing run at the new N\* meets the target (`gap-target-met=true`). Label it a *scaling* change.
   - **N\* stays, frontier improves:** a run at N_f has lower normalized integral than the accepted implementation. Label it a *frontier* change. The final gap at N_f breaks ties.
3. **No regression at or below N\*:** a run at the accepted implementation's N\* still meets the target. The integral there is not gated, so a scaling change that adds a small fixed cost is not rejected for it.

Because run variance is low (< 0.5 s spread at the 120 s scale), a single passing run is sufficient to establish a gate or frontier reference. Keep the accepted implementation's reference run at N_f (and at N\*) in the comparison table; re-measure when the build, hardware or N_f changes.

## Finding N\*

Start at N = 0 (full bucket only), then N = 1, then double (2, 4, 8, ...) until a run misses the target, and bisect between the last passing and first failing N. The long-term goal is N = 131072, so prefer changes whose cost grows slowly with the bucket count. Confirm N\* with a passing run. Record every run, including misses. When N\* rises, the next ladder size becomes N_f: measure the accepted implementation there once before evaluating the next frontier change.

## Target-specific rules

- Calibration benchmarks every pending configuration except reference kernels estimated above 10 s (such as the reference DOT at 16 x 262144). A very slow calibration is reported, not worked around by changing buckets, the threshold or the evaluator.

- Report a budget miss honestly. Do not shrink buckets, drop buckets or prune saturation alternatives to fit the budget.
- Bounds and pruning may exploit that the 2N+1 buckets are near-identical, but must stay valid on arbitrary small multi-bucket oracle cases (`tests/planner_oracle_cases/` supports up to 4 buckets).

## Starting points (verify before relying on them)

- Every remaining cyclic component in prefill+decode buckets is a genuine two-way choice: a partial-reuse `x = SCATTER(r)` versus `r = CONTIGUOUS(SLICE(x))`, or a CPU ⇄ OpenCL `COPY_TO` pair. `cycles.hpp` now strips only enodes that can never be in an acyclic plan (`dominated_child`). The rest must be handled by the search.
- `RelaxationBoundComputer` (`core/plan/relaxation_bound.hpp`) and the root primal heuristic (`primal_heuristic.hpp`) switch off for any bucket with a cycle, leaving the search with no root bound and no early incumbent. Each component has at most 5 classes after stripping, so unrolling each component into acyclic levels (level 0: external alternatives only; level l: a zero-cost alias of level l-1 plus internal alternatives whose children come from level l-1) would keep the relaxation exact.
- `docs/agents/targets/gap_integral/progress.md` records earlier bound and heuristic work on decode buckets.
