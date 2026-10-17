# Improve the planner's primal-dual gap integral

Improve how fast the planner finds good plans **and** proves good lower bounds on Gemma decode planning. The score is the primal-dual gap integral defined in `docs/core/search_gap.md`. Every change must keep at least one optimal plan, never report a lower bound above the optimum, and must not regress the search-volume score or time to first incumbent.

## Execution mandate

When invoked, treat the evaluation command below as fixed and run an autonomous optimization loop. Establish (or reuse) the baseline, then implement and evaluate improvements. A timeout or rejected candidate ends one evaluation, not the task. Continue without asking permission for routine edits, rebuilds, evaluations, oracle runs or diagnostics. If a concrete blocker prevents further work, report it with the work completed rather than asking whether to start the next routine step.

Work toward a lower `normalized` primal-dual integral. There is no fixed target: keep iterating while there are plausible, sound ideas. Record every retained change and every rejected trial.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO
.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 600
.venv/Scripts/python.exe -m tests.test_planner_oracle
.venv/Scripts/python.exe -m tests.test_search_gap
```

Native suites (run after native changes): `tensor_graphs_cpp/test.exe` with `--private-layout`, `--write-after-read-cache`, `--incremental-propagators` and `--start-rank-window`.

- One configuration per process (decode), `--optimal`, no compiled-plan cache, the current hardware and default build options.
- The budget covers worker startup, imports, model loading, preprocessing and search. Precomputing anything outside the timed program (offline solver runs, cached results, files produced by earlier runs) is forbidden: all work must happen inside the evaluated process.
- Read before editing: `docs/core/search_gap.md`, `docs/core/search_volume.md`, `docs/core/planner_oracle.md`, `docs/core/propagators.md`, `docs/reviews/ortools-cpsat-lessons-2026-10-12.md` and `docs/reviews/walkthrough.md`.
- Keep the evaluator, supervisor (`tests/test_gemma_output.py`), gap scorer (`tests/search_gap.py`), volume evaluator, oracle reference enumerator, existing oracle cases, cost model, initial domains, bucket setup and encoding fixed. Changing any of them needs separate review and a new baseline.

## Metrics

Each timed run prints, and `tests/search_experiment.py` records:

| Metric | Source line | Direction |
|---|---|---|
| **Primary: normalized primal-dual integral** | `[Program gap] normalized=` | lower is better |
| Final gap, final incumbent, final certified bound | `[Program gap]` | informative |
| Eliminated bits | `[SearchEngine volume] eliminated-bits=` | must not decrease |
| Time to first incumbent (program seconds) | `first-incumbent-elapsed=` / `New best cost ... event-epoch` | must not regress |
| First-incumbent cost and deadline incumbent cost | gap samples / progress lines | must not increase |
| Deadline overshoot | `[Program deadline] overshoot=` | must be ≤ 1 s (supervisor-enforced) |

A run without a `[Program gap]` line, without a volume report, with invalid gap samples (bound above incumbent, bound decreasing, incumbent increasing), with a crash, or killed by the supervisor is a **failed** candidate.

## Acceptance gate

Run each candidate once. If promising, run two confirmations (three runs total) with identical source and build. Retain the change, labelled an **optimization**, only if all of the following hold:

1. **Correctness:** full oracle passes (including the certified-bound validity check), `tests.test_search_gap` passes, and native suites pass.
2. **Primary improvement:** the median normalized integral of the three runs is lower than the accepted implementation's median, by more than the spread observed between the accepted implementation's own repeated runs. Repeat uncertain comparisons rather than accepting noise.
3. **No volume regression:** eliminated bits are no lower than the accepted implementation's in every run. This metric is deterministic and should match to the printed digits unless the change intentionally alters root reductions.
4. **No time-to-first-incumbent regression:** the median over three runs is no greater than the accepted implementation's median plus 1 second, and no single run exceeds the accepted implementation's slowest run plus 2 seconds.
5. **No cost regression:** first-incumbent cost and deadline incumbent cost are no higher than the accepted implementation's.

A change that only improves a non-primary metric (for example, a pure speedup that lowers time to first incumbent) may be retained if the primary median does not get worse and all other gates pass.

## Soundness rules

- **Lower bounds** must be valid for every plan in the node's box: state the inequality and why it holds. The oracle checks the root bound and every certified bound sampled during exhaustive search against the true optimum. Add a small adversarial case (`tests/planner_oracle_cases/`) that would break the rule if it were wrong. Use `capacity_pages`, multiple engines and buckets, optional children and shared children as needed.
- **Pruning and dominance** must keep at least one optimal plan (equal optima may be dropped). Give the argument in the record. Bounds that depend on the incumbent (for example, reduced-cost fixing) may only exclude assignments that cannot be strictly better than the incumbent.
- **Primal heuristics** (better first dives, LNS, local search) may use any rule, but any plan they report must be validated by the normal leaf evaluation. They must not alter the bound semantics.
- **Fail loud:** invalid states throw or report a conflict; no silent fallbacks.

## Generality rule

Exploiting structure common in today's neural networks is welcome: repeated identical blocks, near-identical decode buckets, residual chains, attention with a KV cache, MLP/GLU, MoE routing, weight-layout alternatives, CPU/GPU copies. Express it through graph properties the planner sees (costs, engines, memory spaces, sizes, alternatives, reachability, structural repetition). Do **not** hard-code instance facts: e-class/e-node IDs, Gemma-specific tensor sizes or capacities, operator names chosen because they appear in this graph, or search orders, thresholds and iteration counts tuned to land events at particular times in this 600-second run. Rules must remain correct (oracle) on arbitrary small cases.

## Iteration

1. State a specific hypothesis, the expected effect on the incumbent curve or the bound curve, and its soundness argument.
2. Make one focused change. Add oracle cases for new bound or pruning rules.
3. Rebuild, run the oracle and the unit and native tests, then the timed evaluation. Use `tests/search_experiment.py --record gap <run_id> ...` for every build, test, evaluation and diagnostic.
4. Compare against the accepted implementation and the original gap baseline. Run confirmations when promising.
5. Retain or reject. Preserve rejected trials' patches and logs (the runner saves a patch per run) and restore the accepted source before the next trial.

Diagnostic runs (shorter timeouts, profiling builds with `--profile`, temporary logging) are encouraged but are not evaluations. Label them as diagnostics and remove the instrumentation before an evaluation.

Watch out for stale builds: copying a backup file over a source file keeps its old timestamp, so the build may report "up-to-date". Touch restored files before rebuilding and confirm the build log shows a compile.

## Logs and resumable progress

Keep `docs/agents/targets/gap_integral/progress.md` and `logs/gap_improvements/` sufficient to resume without conversation history. The runner refuses to overwrite a run id and appends an in-progress entry and a completion entry for each run: command, revision, patch, exit status, volume, deadline, gap and first-incumbent lines, and the oracle verdict. After each run add a short note covering:
- the hypothesis label;
- the metrics compared with the accepted implementation and the original gap baseline;
- the gate verdict and its reason.

Keep a **comparison table** (original gap baseline, accepted implementation, candidate, confirmations: normalized integral, final gap, final bound, final incumbent, first-incumbent time, eliminated bits) and a **resume checkpoint**:
- the accepted implementation and its patch;
- the working-tree state and build state;
- any running process and its log;
- the next concrete step.

Update both after every change and run. Reconcile interrupted runs from their saved logs before continuing.

## Starting points (from earlier research; verify before relying on them)

- Root lower bound is weak: about 0.53 natively, while an engine-load relaxation proves about 30.36 per decode bucket (`docs/reviews/ortools-cpsat-lessons-2026-10-12.md`, R166). `EngineWorkloadPropagator` only counts mandatory work, and nearly every class is optional at the root.
- The first incumbent (75.01) is far above relaxation solutions (about 41.7 for one bucket ignoring memory, R168). The first dive is feasibility-first.
- After the first incumbent, the search explores only a few thousand nodes near it and never improves. Large neighbourhood search around the incumbent, or a cost-aware dive, may help.
- The objective weights 16 decode buckets at 1/16 each (the 17th "full" bucket has weight 0), so bound pruning of single-bucket changes requires tight bounds. Improving the certified bound itself is still directly rewarded by the integral.
