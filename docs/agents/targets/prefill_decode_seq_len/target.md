# Plan longer prefill+decode sequences to within 5% of optimal in 120 seconds

Increase the sequence length N for which the planner compiles Gemma-3-270m's full set of prefill and decode buckets and proves its plan within 5% of optimal, inside a 120-second compile budget. The bucket count grows as 2N+1, so this rewards work that scales with the number of buckets: shared saturation, cycle handling, bounds that stay valid across near-identical buckets, and fast first incumbents. The rules in `docs/agents/targets/README.md` apply.

## Objective

- **Metric:** N\*, the largest `--seq-len N` whose run passes (`ok=true`) with final primal-dual gap below 0.05 (`gap-target-met=true`), in each of three runs with identical source and build.
- **Goal:** no fixed goal. Keep raising N\* while there are plausible, sound ideas.
- **Secondary (tie-breaks and regressions):** final gap at N\* and at N\*+1, compile elapsed time at N\*, normalized primal-dual integral at N\*.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO
.venv/Scripts/python.exe tests/search_experiment.py --record prefill_decode_seq_len <run_id> -m tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len <N> --run-dir logs/gemma_output_runs/<run_id>
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

`tests.test_gemma_output --config prefill+decode --timeout 120` (`superviseLogits`) runs two processes:

1. **Compile (budgeted).** Builds an `LLMSession` with `max_sequence_length=N` and the token buckets of `prefillDecodeBuckets(N)`: prefill `[0, t)` and decode `[t-1, t)`, both needing logits row `t-1`, for `t = 1..N`. The session adds the full bucket, giving 2N+1 buckets (prefill 1 and decode 1 coincide, as in `tests/verify_saturation.py`). **N = 0** is the special case that compiles only the full bucket of a one-token sequence (1 graph). `expectedGraphRanges(N)` defines the exact set. The search runs with `--optimal` (until exhausted) and `TG_SEARCH_GAP=1`, and stops at `TG_SEARCH_DEADLINE = start + 120 - SEARCH_MARGIN_SECONDS`, keeping its incumbent. The plan is then extracted and written to the fresh run directory. Everything (process start, imports, model loading, saturation, search, extraction, plan writing) counts against the 120 s from launching the process; the process is killed at 120 s.
2. **Check (unbudgeted).** A fresh process opens the session from that plan. A run that plans again fails. It runs **every compiled graph directly** (`LLMSession.run_graph_logits`, bypassing bucket selection): for a graph with dirty tokens `[s, e)` it first runs the full graph on `S[:s]` (zero padded) to set up the persistent state, then the graph itself on `S[:e]`. It saves the graph's needed logits rows.

The supervisor then computes the reference with Hugging Face transformers (float32, eager attention, `use_cache=False`) on `S` = the first N tokens, with BOS, of `models/google/gemma-3-270m/README.md`. It compares every saved row, over all vocabulary entries, with `tests.inference_benchmark.compare_row`: `|ours - ref| <= 2e-2 + 1e-3 |ref|` per logit, logsumexp error `<= 2e-2`, and `ref[argmax(ours)] >= max(ref) - 2e-2`.

- One configuration per process, no compiled-plan cache from earlier runs (every run uses a fresh `--run-dir`), current hardware, default build options.
- Precomputing anything outside the timed compile process (offline solver runs, cached plans, files from earlier runs) is forbidden.
- **Fixed:** the supervisor and checks in `tests/test_gemma_output.py` (`superviseLogits`, `runEveryGraph`, `checkLogits`, `prefillDecodeBuckets`, `expectedGraphRanges`, `referenceTokens`, `referenceLogits`), `tests/search_gap.py`, the tolerances in `tests/inference_benchmark.py`, the oracle reference enumerator and existing oracle cases, the cost model and its records, and the bucket definitions. Changing any of them needs separate review and a new baseline.
- **Agent-editable:** saturation, rewrite rules, cycle strip passes, the planner, propagators, bounds, branching, primal heuristics, extraction, kernels, and `SEARCH_MARGIN_SECONDS` in `tests/test_gemma_output.py` (the time kept after the search for extraction and plan writing).

## Result lines

| Line | Meaning |
|---|---|
| `[Gemma compile] supervisor elapsed=... budget=120 exit=... timed_out=...` | Compile process outcome |
| `[Program gap] ... final-gap=... normalized=...` | Primal-dual gap of the compile's search samples over [0, 120 s] (`docs/core/search_gap.md`) |
| `[Gemma logits] graph=<g> dirty=[s,e) rows=[r0,r1) ok=... max_abs_err=...` | One line per compiled graph |
| `[Gemma prefill+decode] ok=... seq_len=N buckets=2N+1 final-gap=... gap-target-met=...` | Summary; also written to `<run-dir>/results.json` |

A run is **failed** if:
- the compile exceeds the budget, crashes, finds no feasible plan before the search deadline, or writes no plan;
- the gap samples are invalid (bound above incumbent, bound decreasing, incumbent increasing);
- the check plans again, or its compiled graphs differ from `expectedGraphRanges(N)`;
- any logits row is outside tolerance.

A passing run with `final-gap >= 0.05` does not meet the target for that N.

## Acceptance gate

1. **Correctness:** full oracle passes, `tests.test_search_gap` passes, native suites pass (`tensor_graphs_cpp/test.exe` with `--private-layout`, `--write-after-read-cache`, `--incremental-propagators`, `--start-rank-window`), and every evaluation run passes the logits check.
2. **Primary improvement:** the new N\* is larger than the accepted implementation's, confirmed by three passing runs at the new N\* that all meet the target.
3. **No regression below N\*:** at the accepted implementation's N\*, all three runs still meet the target, and the median final gap is no higher than the accepted implementation's median plus the spread between its own repeated runs.

A change that keeps N\* but lowers the median final gap at N\*+1, or the median compile time at N\*, may be retained when all other gates pass. Label it as such.

## Finding N\*

Start at N = 0 (full bucket only), then N = 1, then double (2, 4, 8, ...) until a run misses the target, and bisect between the last passing and first failing N. The long-term goal is N = 131072, so prefer changes whose cost grows slowly with the bucket count. Confirm the final N\* with three runs. Record every run, including misses: the final gap at the first failing N shows how far the next step is.

## Target-specific rules

- Report a budget miss honestly. Do not shrink buckets, drop buckets or prune saturation alternatives to fit the budget.
- Bounds and pruning may exploit that the 2N+1 buckets are near-identical, but must stay valid on arbitrary small multi-bucket oracle cases (`tests/planner_oracle_cases/` supports up to 4 buckets).

## Starting points (verify before relying on them)

- Every remaining cyclic component in prefill+decode buckets is a genuine two-way choice: a partial-reuse `x = SCATTER(r)` versus `r = CONTIGUOUS(SLICE(x))`, or a CPU ⇄ OpenCL `COPY_TO` pair. `cycles.hpp` now strips only enodes that can never be in an acyclic plan (`dominated_child`). The rest must be handled by the search.
- `RelaxationBoundComputer` (`core/plan/relaxation_bound.hpp`) and the root primal heuristic (`primal_heuristic.hpp`) switch off for any bucket with a cycle, leaving the search with no root bound and no early incumbent. Each component has at most 5 classes after stripping, so unrolling each component into acyclic levels (level 0: external alternatives only; level l: a zero-cost alias of level l-1 plus internal alternatives whose children come from level l-1) would keep the relaxation exact.
- `docs/agents/targets/gap_integral/progress.md` records earlier bound and heuristic work on decode buckets.
