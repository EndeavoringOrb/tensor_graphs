# Optimization targets

Each target is one folder `docs/agents/targets/<name>/` with two files:

| File | Contents | Who edits |
|---|---|---|
| `target.md` | Objective, fixed evaluation contract, metric, acceptance gate, target-specific rules | The user (fixed while an agent works on it) |
| `progress.md` | Resumable record: baseline, comparison table, one entry per run, resume checkpoint | The agent, after every change and run |

Run every build, test and evaluation through the runner so each run gets a log, metadata and a source patch:

```powershell
.venv/Scripts/python.exe tests/search_experiment.py --record <name> <run_id> <command...>
```

`--record <name>` appends to `docs/agents/targets/<name>/progress.md` and writes `logs/targets/<name>/<run_id>.{log,json,patch}`. `gap_integral`, `search_volume` and `inference_speed` (also reachable as `gap`, `search` and `inference`) keep their older log folders under `logs/`.

## Adding a target

1. Copy `_template/` to `docs/agents/targets/<name>/` (lowercase, underscores).
2. Fill in `target.md`. The evaluation command and the pass/fail checks must be executable, fixed, and print machine-readable result lines.
3. Leave `progress.md` as the skeleton. The first agent run records the baseline.
4. Add a row to the table below.

## Targets

| Target | Objective | Evaluation |
|---|---|---|
| [gap_integral](gap_integral/target.md) | Lower normalized primal-dual gap integral on Gemma decode planning | `tests.test_gemma_output --config decode --optimal --timeout 600` |
| [prefill_decode_seq_len](prefill_decode_seq_len/target.md) | Largest sequence length N whose 2N+1 prefill+decode buckets are planned to within 5% of optimal in 120 s | `tests.test_gemma_output --config prefill+decode --optimal --timeout 120 --seq-len N` |
| [search_volume](search_volume/target.md) | Eliminate at least half of the search's domain bits (`remaining-domain-bits <= total-domain-bits / 2`) on Gemma full and decode planning | `tests.test_gemma_output --config full\|decode --optimal --timeout 60` |
| [inference_speed](inference_speed/target.md) | Higher prefill and decode tokens per second, matching the transformers reference | `tests.inference_benchmark --run-dir logs/inference_runs/<run_id>` |
| [saturation_seq_len](saturation_seq_len/target.md) | Largest sequence length whose 2N+1 buckets saturate with exact gold equivalence in under 30 s | `tests/verify_saturation.py --seq-len N` |

## Rules shared by every target

A target's `target.md` may tighten these rules but not loosen them.

**Execution mandate.** When invoked on a target, treat its evaluation contract as fixed and run an autonomous loop: establish or reuse the baseline, then implement and evaluate improvements. A timeout or rejected candidate ends one evaluation, not the task. Do not ask permission for routine edits, rebuilds, evaluations, oracle runs or diagnostics. If a concrete blocker stops progress, report it with the work completed.

**Exactness.**
- Lower bounds must hold for every plan in the node's box. State the inequality and why it holds.
- Pruning, dominance and cycle stripping must keep at least one optimal plan (equal optima may be dropped). Give the argument in the record.
- Add a small adversarial case to `tests/planner_oracle_cases/` that would break each new bound or pruning rule if it were wrong. `tests.test_planner_oracle` checks propagation, search, certified bounds and every cycle strip pass (alone and as the pipeline) against independent enumeration.
- Saturation keeps every sound alternative. Make it faster or parallel; never prune alternatives for speed.
- Optimize the graph the model builds. Partial reuse and KV-style caching must come from saturation, never from hand-inserted slices or caches in the model graph.
- Primal heuristics may use any rule, but every plan they report goes through the normal leaf evaluation.
- Fail loud: invalid states throw or report a conflict, with no silent fallbacks.

**Generality.** Exploit structure common in neural networks (repeated blocks, near-identical buckets, residual chains, KV caches, CPU/GPU copies) through graph properties the planner sees. Do not hard-code instance facts: e-class/e-node IDs, model-specific sizes, operator names chosen because they appear in this graph, or thresholds tuned to land events at particular times in one run.

**One evaluation at a time.** Timed evaluations run alone on the machine: no builds, oracle runs, other evaluations or heavy diagnostics in parallel. Budgets and throughput are sensitive to load and thermals. Untimed phases of an evaluation (calibration, checks) may overlap only with nothing timed.

**Evidence.** Run each candidate once; if promising, run two confirmations (three runs total) with identical source and build. Diagnostic runs (shorter budgets, profiling builds, temporary logging) are welcome but are not evaluations. Label them, and remove the instrumentation before an evaluation. Restoring a backup file keeps its old timestamp, so touch it and confirm the build log shows a compile.

**Records.** Keep `progress.md` and the run logs sufficient to resume without conversation history:
- a note per run: hypothesis, metrics against the accepted implementation and the baseline, gate verdict and reason;
- a comparison table;
- a resume checkpoint: accepted implementation and its patch, working-tree and build state, running processes, next concrete step.

Reconcile interrupted runs from their saved logs before continuing.
