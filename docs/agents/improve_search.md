# Improve planner search

Improve selectors, branching, propagation, and admissible lower bounds for Gemma
full and decode planning. Preserve at least one optimal solution, including
when several encodings represent the same plan. This is an empirical correctness
gate, not a proof of soundness.

## Fixed evaluation contract

Read `docs/core/search_volume.md`, `docs/core/planner_oracle.md`, and the Gemma
search research in `docs/reviews/` before editing. Agree on a fixed configuration,
timeout, build options, hardware, and repeat count. Record a baseline first.
Use one configuration per process and disable compiled-plan caching.

```powershell
.venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO
.venv/Scripts/python.exe -m tests.test_gemma_output --config full --optimal --timeout 60
.venv/Scripts/python.exe -m tests.test_gemma_output --config decode --optimal --timeout 60
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

The timeout covers worker startup, imports, model loading, preprocessing, and
search. It is independent of `--min-compile-time`. Native code stops at a complete
frontier boundary and reports volume once on timeout; a supervisor kills a worker
that exceeds the budget plus two seconds of reporting grace. A missing report,
crash, watchdog kill, or failed oracle is a failed candidate. Never give those
runs an exhausted-frontier score. Record program elapsed time and deadline
overshoot alongside the score; reporting grace is not additional search budget.
Boundary checks can overshoot during a long propagation step. The supervisor rejects overshoot above 50 ms. For finer comparisons, reject
any overshoot or rerun at a longer fixed budget; do not compare them as
exact fixed-time measurements. If setup consumes the budget, increase the common
budget for baseline and all candidates.

The objective is to minimize `log2-remaining-fraction`, equivalently maximize
`eliminated-bits`. The numerator is the union of stored unresolved disjoint
frontier boxes. It is not a count of feasible plans. Empty frontier has infinite
eliminated bits, and must pass the correctness gate. Keep initial domains,
costs, bucket setup, incumbent policy, and encoding fixed across comparisons.

## Iteration

1. Describe a specific hypothesis and why it is sound. Identify the constraints
   or lower-bound inequality that justify any pruning. Consider optional nodes,
   views, sparse domains, equal optima, and shared constraints across buckets.
2. Make one focused change. Add a small counterexample case when introducing a
   new rule; see the oracle schema documentation. Do not change the reference
   semantics merely to make the candidate pass.
3. Rebuild the extension, run each fixed timeout evaluation, then run the whole
   exhaustive oracle suite. Reference enumeration is cached by case contents and
   reference source; native candidate checks always run again.
4. Save commands, build revision/options, logs, score, elapsed time, overshoot,
   and oracle verdict. Repeat promising results to distinguish noise from gains.
5. Retain a change only after correctness passes and comparable runs improve.
   Explain rejected hypotheses too. Seek independent reviewer counterexamples.

Keep the evaluator, supervisor, reference enumerator, existing cases, objective,
and workload definition trusted. Changes to those need separate review and a
new baseline. Do not optimize by reducing domains without a sound justification,
changing the problem, hiding work outside search, skipping cases, or fabricating
metrics. Reviewers should freely add small adversarial JSON cases; private cases
help detect overfitting. Small exhaustive checks cannot prove correctness on
large inputs: supply a mathematical argument for each new pruning rule.
