# Improve planner search

Improve selectors, branching, propagation, and admissible lower bounds for Gemma
full and decode planning. Preserve at least one optimal solution, including
when several encodings represent the same plan. This is an empirical correctness
gate, not a proof of soundness.

## Execution mandate

When invoked with a benchmark command, treat that command as the fixed
evaluation configuration for an autonomous optimization task. Run it to
establish the baseline, then implement and evaluate search improvements.

Do not stop after the baseline or after a benchmark timeout. A timeout ends
one evaluation, not this task. Continue the iteration cycle below without
asking permission for routine edits, rebuilds, evaluations, or oracle checks.

Unless the user specifies otherwise, use the supplied configuration and
timeout, current hardware and build options, one baseline run, and two
confirmation runs for promising candidates. Record these settings before
editing. Create `docs/agents/SEARCH_IMPROVEMENTS.md` if it does not exist.

Strive for `remaining-domain-bits <= total-domain-bits / 2` within the fixed
evaluation budget, with all correctness checks and the non-regression gate below
passing, and confirmation runs meeting the target. A smaller improvement is
progress, not completion. Continue
the optimization loop toward this target; three rejected hypotheses are not a
stopping condition. An exhausted frontier meets the target only if correctness
passes. Do not change initial domains or the evaluation contract to meet it.
Report retained changes, rejected trials,
measurements, and correctness results. If a concrete blocker prevents further
work, report it and the work completed instead of asking whether to start
the next routine step.

## Fixed evaluation contract

Read `docs/core/search_volume.md`, `docs/core/planner_oracle.md`, and the Gemma
search research in `docs/reviews/` before editing. Fix the configuration,
timeout, build options, hardware, and repeat count using the user's instructions
and the defaults above. Record a baseline first.
Use one configuration per process and disable compiled-plan caching.

```powershell
.venv/Scripts/python.exe build.py --targets bindings --log-level INFO
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
Boundary checks can overshoot during a long propagation step. The supervisor rejects overshoot above 1 second. For finer comparisons, reject
any overshoot or rerun at a longer fixed budget; do not compare them as
exact fixed-time measurements. If setup consumes the budget, increase the common
budget for baseline and all candidates.

The objective is to minimize `log2-remaining-fraction`, equivalently maximize
`eliminated-bits`. The numerator is the union of stored unresolved disjoint
frontier boxes. It is not a count of feasible plans. Empty frontier has infinite
eliminated bits, and must pass the correctness gate. Keep initial domains,
costs, bucket setup, incumbent policy, and encoding fixed across comparisons.
The target is to eliminate at least half the total domain bits, equivalently
`eliminated-bits >= total-domain-bits / 2`. This is a target on the logged bit
quantities, not merely halving the remaining volume.

## Comparison and non-regression gate

Track eliminated bits (higher is better), time to first feasible incumbent
(lower is better), first-incumbent cost (lower is better), and best incumbent
cost at the fixed deadline (lower is better). Keep the original baseline as a
permanent comparison anchor; also compare each candidate with the currently
accepted implementation under the same evaluation contract.

Retain a candidate only when correctness passes and comparable runs improve
at least one of these metrics without regressing any other. Eliminated bits
must not decrease, and neither incumbent cost may increase. Time to first
incumbent must be no greater than the original baseline's time plus 1 second;
the 1-second allowance is always measured from the original baseline, never
added again for successive candidates. Also avoid regressing against the
currently accepted implementation, except for timing differences within that
original-baseline allowance. Apply the gate to the candidate evaluation and
both confirmation runs; repeat uncertain comparisons rather than retaining a
change on noise. The timing allowance is separate from deadline overshoot.

Measure time to first incumbent from program evaluation start, including setup,
using the timestamp of the first feasible incumbent event. Record search-only
elapsed time separately if available. Use the same timing origin in all runs.
Do not substitute iteration count or a later progress timestamp for the event
time. If existing logs only bracket the event, record the interval explicitly;
if it cannot resolve the 1-second gate, obtain a precise measurement before
accepting a candidate. Do not invent precision from rounded progress logs.

When no incumbent is found before the deadline, record both incumbent costs
and time as unavailable (comparison value +infinity), explicitly stating
"no incumbent within budget". This is a regression against any baseline that
found an incumbent, regardless of bit gains. Reassess previously retained
changes against this gate when resuming; preserve their historical results and
mark any nonconforming change as requiring repair or rejection.

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
4. Save every run and update the resumable record below before starting the
   next run. Repeat promising results to distinguish noise from gains.
5. Retain a change only after correctness and the comparison/non-regression
   gate pass; label accepted changes as "optimizations".
   Explain rejected hypotheses too as "trials". Log the explanations in docs/agents/SEARCH_IMPROVEMENTS.md

## Logs and resumable progress

Keep `docs/agents/SEARCH_IMPROVEMENTS.md` and the referenced files under `logs/`
sufficient to resume optimization without the conversation history. Read the
existing record before starting; resume pending work and reuse comparable
recorded baselines rather than silently restarting the experiment.

Capture complete stdout and stderr for every build, baseline, candidate,
confirmation, oracle, and diagnostic run in a uniquely named file under
`logs/search_improvements/`. Preserve failed, timed-out, interrupted, and
rejected runs too. Never overwrite logs or save only a tail of the output.
Console output may be abbreviated only after the complete output is saved.

Before each run, add a record with its identifier, log path, exact command,
working directory, start time, configuration, timeout, build options, hardware,
and source state. Record the revision and preserve a patch of any uncommitted
candidate changes under `logs/search_improvements/`, including relevant new
files, so each run can be tied to the exact implementation evaluated. Mark the
run as in progress, then update it immediately on completion with its end time,
exit status, score, elapsed time, deadline overshoot, correctness/oracle verdict,
time to first incumbent, first-incumbent cost, best incumbent cost at the
deadline, and any failure or interruption. Mark unavailable metrics explicitly; do not
infer success from missing output.

Copy every `[SearchEngine volume]` report into `SEARCH_IMPROVEMENTS.md`, linked
to its run and complete log. Record all reported fields: `iterations`, `elapsed`,
`total-domain-bits`, `remaining-domain-bits`, `log2-remaining-fraction`,
`eliminated-bits`, `remaining-nodes`, `exhausted`, and `incomplete-branches`.
Preserve the reported precision. Also record the target
`total-domain-bits / 2`, the remaining-bit ratio
`remaining-domain-bits / total-domain-bits`, and whether the target was met.
If no volume report was produced, explicitly mark it missing and the run failed
under the evaluation contract; never substitute invented bit measurements.

For each hypothesis, document the before behavior and baseline run links, the
change and its rationale/soundness argument, and the after results with links
to every run and saved patch. Explain the measured difference and whether the
change was retained, rejected, reverted, or remains pending. Preserve rejected
trial records when reverting code.

Include a comparison table in `SEARCH_IMPROVEMENTS.md` covering the original
baseline, currently accepted implementation, candidate, and every confirmation:
eliminated/remaining bits, time to first incumbent (with timing origin and any
uncertainty), first-incumbent cost, and best incumbent cost at the deadline.
Link the incumbent events in the saved logs, record deltas for every metric,
and state the gate verdict and reason. A bit improvement alone cannot justify
retention when incumbent discovery or cost regresses.

Keep a current resume checkpoint in `SEARCH_IMPROVEMENTS.md`: the retained
implementation and baseline, current working-tree changes and build state,
completed and pending checks, any running process and its log location, and
the next concrete optimization step. Update it after each change and run, and
before stopping or handing off. If a previous run was interrupted, reconcile
its saved output and process status before continuing; leave its verdict
incomplete unless the required checks actually finished.

Keep the evaluator, supervisor, reference enumerator, existing cases, objective,
and workload definition trusted. Changes to those need separate review and a
new baseline. Do not optimize by reducing domains without a sound justification,
changing the problem, hiding work outside search, skipping cases, or fabricating
metrics. Reviewers should freely add small adversarial JSON cases; private cases
help detect overfitting. Small exhaustive checks cannot prove correctness on
large inputs: supply a mathematical argument for each new pruning rule.
