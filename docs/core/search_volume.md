# Remaining search volume evaluation

Enable measurement for Gemma with:

```powershell
.venv/Scripts/python.exe -m tests.test_gemma_output --config full --optimal --search-volume
```

For a whole-program evaluation budget:

```powershell
.venv/Scripts/python.exe -m tests.test_gemma_output --config full --optimal --timeout 60
```

`--timeout` enables volume tracking and starts the budget before worker startup
and imports. It includes model loading and preprocessing. Use one configuration,
`--optimal`, and no compiled cache. Native search stops without needing an
incumbent at the first complete frontier boundary after the deadline and reports
volume. The supervisor allows two seconds for reporting, then kills the worker.
Missing reports, worker failures, and deadline overshoot greater than 1 second fail
evaluation. This is a cooperative deadline, not an exact interrupt inside a
propagator; accepted runs can overshoot by up to 1 second. The deadline uses a wall
clock timestamp across processes; avoid changing the system clock during runs.
Program elapsed time is measured separately with a monotonic clock.

The native extension must be rebuilt after changing C++ headers. Other callers
can set `TG_SEARCH_VOLUME=1` before constructing their search engine.

## Definition

The reference hyperbox is the Cartesian product of all search variable domains
at entry to `SearchEngine::solve`, **before root propagation**. Preprocessing and
variable creation have already happened; their pruning is not included. All
variable types are counted: CACHED, SELECTED, START, and OFFSET.

For a box H with domains D_i:

```
box_bits(H) = log2(|H|) = sum_i log2(|D_i|)
remaining_domain_bits = log2(sum_H |H|)
log2_remaining_fraction = remaining_domain_bits - total_domain_bits
eliminated_bits = -log2_remaining_fraction
```

`log2_remaining_fraction` is the requested union/total-domain ratio expressed in
bits: zero means all the reference volume remains; -10 means 1/1024 remains.
`eliminated_bits` is the corresponding positive score, where larger is better.
For example, if the initial volume is 16 and the frontier contains two disjoint
boxes of volume 2 each, remaining volume is 4, the ratio is 1/4, and the scores
are -2 and +2 bits respectively.

An exhausted frontier reports `remaining-domain-bits=-inf`,
`log2-remaining-fraction=-inf`, `eliminated-bits=inf`, and `exhausted=true`.
The solver can exhaust an infeasible problem too; exhaustion does not by itself
mean a feasible plan was found. A root with an empty initial domain has no
defined reference ratio and fails explicitly.

## Union accounting

The frontier contains queued nodes plus the currently popped active node at a
branch-point measurement. Ancestors and previously visited nodes are not counted.
Nodes whose valid stored lower bound meets or exceeds the incumbent are omitted,
including stale queue entries that have not yet been physically removed.

Branch children must restrict the same variable, be nonempty subsets of their
parent domain, and be disjoint. The evaluator checks these properties while
measurement is enabled. This makes all frontier boxes disjoint by construction,
so summation is their union volume without a general multidimensional union
algorithm. Masks and interval intersections are counted exactly, including mask
holes and different mask bases. Cardinalities use 64-bit integers; volumes are
combined in log space rather than materialized.

Lazy right branches use the propagated parent box restricted by their branch
delta. They have not yet received their own propagation, so the metric counts
that larger unresolved box. Restoring a node refreshes its stored volume after
propagation. The evaluator does not replay queued nodes to discover additional
pruning just for measurement. Thus it measures the stored unresolved frontier,
not the unknown number of feasible or improving assignments inside it.

`incomplete-branches` counts branch splits whose children do not cover the entire
parent domain. Such splits still give disjoint boxes, but the omitted assignments
must have a separate justification before interpreting all measured elimination
as a proof of global optimality. Existing packing policies and propagator
soundness are also outside this evaluator's verification. In particular, this
metric does not certify the correctness of arbitrary pruning rules.

## API and cost

`plan::evaluateSearchVolume(total_domain_bits, remaining_box_bits)` is the numeric
evaluation function for a supplied list of disjoint boxes. The caller is
responsible for disjointness when using this function directly.

`SearchEngine::evaluateSearchVolume(active_node)` gathers the built-in selector's
remaining boxes, filters incumbent-pruned entries, and includes the active node
when supplied. At iteration boundaries, omit that argument. Do not pass a node
that is also queued. Custom selectors must implement `appendRemainingBoxBits`;
unsupported selectors fail explicitly.

Domain volumes are tracked in a stable sum tree: initialization is O(variables),
each domain change or restoration is O(log variables), and each stored node adds
one double. A frontier evaluation scans O(queued nodes) and uses O(queued nodes)
temporary storage. Measurement is opt-in so ordinary searches do not maintain
the tree or scan the frontier.

The engine emits `[SearchEngine volume]` records on timeout and on normal exit. Each record contains elapsed
seconds, both volume logs, the ratio in bits, eliminated bits, remaining node
count, exhaustion status, and incomplete-branch count. A contradictory root also
emits an exhausted record. Logs require an INFO-or-more-verbose build.

For an optimization loop, hold initial domains, costs, bucket configuration,
incumbent policy, and elapsed-time budget fixed across candidates. Raw volume
includes irrelevant address/rank encodings for optional selections; this is a
measure of the encoded search space, not a count of distinct execution plans.
