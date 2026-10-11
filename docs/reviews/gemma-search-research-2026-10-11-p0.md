# Gemma optimal search and bucket scaling research

Research only. No implementation changes or benchmarks were run. Recommendations are hypotheses grounded in the current source and the primary sources linked below; speedups need measurement on the two requested Gemma configurations.

## Current model and concrete gaps

- `tests/test_gemma_output.py`: both configurations use sequence length 16. `--optimal` sets `min_compile_time=-1`, requesting search exhaustion. Compilation caching is disabled unless a cache filename is supplied.
- `tensor_graphs_cpp/bindings.cpp:406`: decode adds one dirty-token/output-token bucket for each position. This is currently decode buckets plus the full bucket, rather than the proposed decode and prefill family.
- `core/session.hpp:231`: the full bucket has weight zero when other buckets exist, but still needs a feasible plan, including cache production/freshness. With only the full bucket it has weight one.
- `core/plan/search_engine.hpp:562`: objective is the weighted sum of simulated bucket makespans. START variables are dispatch ranks, not elapsed time. The simulator waits for dependencies and engines.
- `core/plan/search_state.hpp:930`: bucket bound is the maximum of critical path, engine workload, and start-run bounds; total bound is their weighted sum.
- `propagators/critical_path.hpp`: recursion returns zero for a class whose selection still contains zero. A parent alternative can require an expensive child, but the child's cost disappears until propagation makes that child globally mandatory. This is safe but unnecessarily weak.
- `propagators/engine_workload.hpp`: only fixed positive selections contribute workload. Mandatory classes with multiple implementations contribute nothing.
- Both these bound propagators only reject an entire state. They do not remove individual candidates that cannot beat the incumbent.
- `brancher.hpp:673`: choose the first bucket with a suitable selection, then shallowest root depth, then small domain. This is not a global comparison across buckets. Implementation preference combines static memory and cost estimates. Cache choices use candidate order and prefer zero.
- `search_node.hpp` and `search_engine.hpp:968`: optimization priority includes a fixed `+1` for right branches and `-0.01*depth` for left branches. Lazy right branches inherit the parent bound. Queue order is not strictly by lower bound.
- `propagators/cached_offset.hpp`: cross-bucket offset equality propagates only from a fixed address. `cached_offset_allocation.hpp` also intersects bounds when allocating, then fixes a sequential address; it does not maintain domain intersection on every unfixed restriction.
- `propagators/start_unique.hpp`: removes already fixed dispatch ranks from other mandatory starts. There is no Hall-interval overload reasoning.
- `domain.hpp`: domains use a 32-bit mask or an interval. Removing an interior value from a wide interval cannot be represented. This limits pruning of large schedule/address domains.

Existing reachability, parent support, precedence, views, cache exclusion/requirement/freshness, cycle pruning, and write-after-read propagation should be extended rather than duplicated. Existing embeddinggemma logs are a different workload and older source snapshots; they are not a timing baseline for these commands.

## First priority: strengthen bounds before selections are fixed

### 1. Conditional critical path through optional children

Maintain the bound for executing a class, separately from whether that class is mandatory:

`execLB(C) = min over active nonzero alternatives a of [cost(a) + max over children D of execLB(D)]`.

The zero/unselected value must not be considered inside this recurrence. It is valid to descend into an optional child because evaluating alternative `a` assumes all of its children execute. Use the root execution bound, and the maximum execution bound of any other globally mandatory class, as bucket bounds.

Example: root alternatives are `A(cost 1) -> X(cost at least 100)` and `B(cost 20) -> Y(cost at least 30)`. X and Y are optional before root choice. The current bound can be 1; the conditional bound is 50. With incumbent 60, A can be removed immediately using its conditional bound 101.

Use the existing all-buckets-DAG result for a simple reverse topological pass. For cyclic buckets, use a proven conservative SCC/fixed-point relaxation; do not apply the DAG recurrence with arbitrary memoization assumptions. Shared children are safe under maximum-path reasoning. Summing child subtree costs is not safe for DAG sharing.

Make it incremental: alternative removal or child-bound increase updates consumers, maintaining alternative maxima and class minima. Restore these values or invalidate affected data on backtracking.

### 2. Mandatory minimum workload per engine

For mandatory class C and engine e, contribute:

`m(C,e) = min over active alternatives a of [cost(a) if a occupies e, else 0]`.

Then `workLB(e) = sum over mandatory C of m(C,e)`. This remains valid with alternative engine assignments because a choice that avoids e contributes zero. Match simulator treatment of views, INPUT, CACHE, empty engine sets, and physical engines exactly.

Example: ten mandatory operations each have two CPU implementations costing 3 or 5. Before any implementation is fixed, the current workload contribution is zero; the new CPU bound is 30.

Stronger extension for flexible engines: for nonnegative engine weights alpha summing to one, each mandatory class contributes the minimum weighted engine occupancy over its alternatives. Their sum is also a makespan lower bound. Engine-subset bounds or a small LP can improve the relaxation when individual engine minima are all zero. Count each mandatory e-class once; optional descendants need a sharing-aware selection relaxation, not summed independent subtree estimates.

### 3. Incumbent-driven candidate removal and bucket budgets

Let U be the incumbent, L_b the bucket bounds, and w_b its positive weight. Bucket b has an optimistic improvement ceiling:

`T_b = (U - sum over j != b of w_j*L_j) / w_b`.

If assuming a candidate in b gives a valid conditional bucket bound at least T_b, remove it. A bound increase in any bucket tightens the budget for every other bucket. This is a direct, sound form of cross-bucket propagation without assuming equal implementations.

Example: two unit-weight buckets, incumbent 100, other bucket bound 40. Candidate bound 65 in this bucket rules out that candidate even if its current unconditional bound is only 20.

For shared cache decisions, evaluate both polarities across affected buckets: `L(cache=0)` and `L(cache=1)`. If either is at least U, remove that polarity. A failed local probe should therefore prune cache-dependent choices in every bucket through existing cache propagation.

Avoid scanning every bucket after every bound increase. Maintain weighted slack and enqueue candidates whose stored slack certificates have expired. Zero-weight buckets have feasibility propagation but no objective ceiling. New incumbents must explicitly wake objective filtering, even if no variable domain changed.

Use conservative floating-point comparisons or a well-defined integer cost representation before introducing aggressive equality pruning.

### 4. Release/workload/tail bounds

Critical path and total engine work miss forced idle gaps. On a unary engine, any mandatory operation set S with valid earliest release bounds r_i and mandatory downstream tail bounds q_i yields:

`makespan >= min(r_i in S) + sum(processing_i in S) + min(q_i in S)`.

Start with groups having high releases or tails. Exclude each operation itself from its head/tail and use guaranteed dependencies, not optional paths. More generally, a window cannot contain more mandatory engine energy than its capacity permits.

OR-Tools implements disjunctive and cumulative overload propagation, and Schutt et al. explain time-table/edge-finding with nogood learning. These are reference algorithms, not direct drop-in rank propagators: introduce separate duration-based earliest/latest time bounds because existing START values are dispatch ranks.

## Second priority: remove unnecessary schedule and allocation search

### 5. Prove when scheduling cannot improve the objective

If every positive-cost operation uses the same unary engine, the graph is acyclic, and all other operations introduce no extra delay, every legal topological order has the same makespan: the sum of selected costs. Threaded kernels may still occupy the same modeled CPU engine; inspect the actual mapped engine sets.

Once selections are fixed in such a bucket, solve scheduling/allocation as feasibility only. On finding one feasible arrangement, that bucket is solved for those selections and shared decisions. There is no benefit to enumerating another dispatch permutation or buffer layout with the same objective.

Offsets do not enter `evaluateMakespan`; they matter through feasibility. Even for multiple engines, after a fixed dispatch order admits a valid allocation, alternative addresses cannot improve that order's cost. Stop that allocation subproblem after a feasible witness. A decomposition is needed to retain completeness across other orders and selections.

### 6. Stronger rank and memory propagation

Dispatch ranks: add bounds-consistent all-different over mandatory operations. Three operations restricted to two ranks conflict before any rank is fixed. A Hall interval occupied by k mandatory operations can be removed from outside operations. Use interval endpoints rather than visiting every possible rank.

Rank symmetry: investigate dense ranks for active operations, because the simulator uses order rather than numeric rank gaps. Establish compatibility with start-run bounds and memory constraints before enforcing normalization. Independently reorder operations only when engine effects, dependencies, readers, views, and memory feasibility commute.

Memory: propagate unavoidable live ranges without waiting for fixed addresses. Required production and last-reader bounds imply compulsory lifetime segments. If their distinct non-alias buffer sizes exceed usable memory at a rank, fail; if assuming a candidate overloads memory, prune it. Persistent caches consume capacity throughout. Do not double-count views or optional readers.

For allocations whose offsets must overlap, derive `A's required readers before B` OR the symmetric order even while both starts are unfixed. Remove an impossible order using start/reader bounds, then propagate the surviving order. Current memory documentation explicitly defers some cases with two unfixed starts.

Represent order decisions directly when they are more informative than choosing a rank value. Represent disjoint forbidden address intervals if profiling shows wide range domains repeatedly exploring known holes.

## Third priority: branch and select using observed consequences

### 7. Separate incumbent construction from optimality proof

Keep memory-aware dives to obtain feasible incumbents. After an incumbent exists, use lower bound as the primary queue key and depth/locality as tie-breakers. Replace fixed additive left/right/depth biases with lexicographic fields. Raw cost units should not compete with arbitrary constants.

Probe both children for a small shortlist of impactful decisions. Measure weighted bound increase, domain reduction, conflicts, and propagation time. Learn separate pseudocosts for choosing/removing implementations and cache polarities. Use reliable historical estimates to avoid expensive probes at every node.

Compare decisions across buckets: high objective weight, large candidate bound spread, critical resource pressure, and global cache fanout are useful signals. For cache choices, evaluate both branches instead of always preferring zero. Reuse neighboring bucket incumbents as value preferences.

Refalo's impact-based search learns from domain reduction. Here a useful adaptation also learns bound gain per unit propagation time. Domain impact alone may favor large reductions in irrelevant optional classes. Keep conflict weights attached to actual scopes/variables, rather than only a propagator name.

### 8. Learn explained failures

Current bounded conflict signatures are diagnostics, not reusable constraints. Start with a narrow set of explanation-producing rules: impossible selection/precedence, cache budget, shared address intersection, and conditional cost bounds.

Learn, for example, `not(cache_X=1 and implementation_Y=k)` when those assumptions alone establish the contradiction. Memory failures should explain the relevant selections/orders/addresses, not ban an implementation that merely failed under one layout.

Transfer a learned constraint to another bucket only if its assumptions and constraints map identically or have a proved implication. Cost cuts additionally need appropriate cost/weight bounds. Nogoods derived from a valid incumbent remain useful as incumbents improve; assumptions about changed cost data require invalidation.

## Scaling to 262144 buckets: exploit the shared interface

### 9. Shared offset domains before fixing

For cached base X, maintain `D_shared(X) = intersection of offset domains of every required bucket copy`. Push every narrowing to its copies. Bounds intersection is already useful; preserve mask holes where applicable.

Example: copies allow [10,30] and [20,40]. With caching required, both become [20,30], although neither was fixed. If caching remains optional and the conditional intersection becomes empty, remove `cache_X=1` and let existing cache exclusion prune all dependent implementations.

A single canonical offset variable per cached base can remove equality copies. Be careful: uncached copies retain independent offsets, and cache sizes must cover the maximum footprint across buckets.

### 10. Decompose the Cartesian product of bucket choices

Let g contain shared cache decisions and persistent allocation decisions. Conditional on g, check that all remaining constraints are bucket-local. Then:

`optimum = min over feasible g of sum_b w_b * optimum_b(g)`.

Zero-weight buckets supply feasible witnesses. Cache freshness/production obligations stay in g or in the corresponding local constraints. Buckets share physical addresses across calls but are not concurrent schedules; reserve persistent ranges consistently.

Search g in a master problem. Solve buckets independently, returning feasible plans, lower bounds, exact local optima, and infeasibility explanations. Cache results by the relevant subset of g rather than the entire global assignment. If one bucket proves a shared cache combination impossible, learn a master cut and prune all combinations with that cause.

This is a direct application opportunity for logic-based Benders decomposition. Hooker's work emphasizes strong subproblem relaxations and informative cuts. The repo-specific factorization above follows from the current variable coupling; verify it before implementation. It avoids enumerating every combination of equally good independent bucket schedules.

### 11. Certify bucket families instead of assuming identical choices

Fingerprint candidate structure using logical/base identities, kernel identities, shapes, strides, regions, engine sets, and cache obligations. E-class IDs or candidate indices alone are not portable between bucket egraphs.

Reuse exact solutions for fully equivalent problems, aggregating weights. For buckets with the same feasible set but different costs, share structure and feasibility nogoods, then solve the changed objective. Position changes can alter attention extent, shapes, dirty regions, cache paths, kernel eligibility, and alignment, so neighboring sequence lengths do not imply the same optimum.

For a length interval, a uniform dominance certificate can eliminate implementation B if A is at least as good for every length and has compatible children, engines, storage, aliasing, and cache effects. A cost crossover splits an interval; kernel eligibility boundaries do too. Similar checks can reuse a solution when a region translation leaves the complete optimization problem equivalent.

Adjacent-bucket solutions and learned branch scores are always useful as hints. Fixing choices requires equivalence, dominance, or a conditional bound certificate. Build common templates with sparse bucket differences; lazily expand uncertified families. Exact optimality still requires accounting for every bucket, individually or through a valid family certificate.

## Costs that become material with many buckets

- `updateLowerBoundBucket` currently resums all bucket bounds each time one changes. Maintain the weighted sum by delta, with correct restoration and periodic numerical reconciliation. With B buckets initialized one at a time, repeated whole-vector sums can become quadratic.
- Several brancher/revision caches rebuild across all buckets when one selection changes. Track per-bucket revisions and affected shared-variable subscribers.
- `CacheFreshnessPropagator::ensureIndex` loops cache bases, buckets, and selection maps. Build reverse indexes in one pass over selections.
- Bucket duplicate reporting in `planAll` compares earlier buckets. Hash canonical bucket descriptions for grouping.
- A full egraph, domain set, and indexes per bucket will be expensive even after search improves. Share immutable templates and materialize differences. This supports the proposed propagators and decomposition.
- The model's atomic attention path constructs `[heads, seq_len, seq_len]` tensors. At length 131072, one FLOAT32 attention matrix alone is 64 GiB per head. Include early memory infeasibility propagation and distinguish plans that avoid materialization from plans that require it. Search improvements alone cannot make an infeasible materializing plan feasible.

## Exactness questions to resolve alongside pruning

`setOffsetDecision` chooses `offset=preferred` versus `[preferred+1,max]`. If preferred exceeds the current minimum, this drops smaller addresses unless another valid rule already eliminated them. Require an explicit obstacle or symmetry proof; a heuristic preference alone is insufficient.

`CachedOffsetAllocationPropagator` fixes caches by arrival-order sequential packing. This is a policy restriction unless canonical packing is proved to preserve a feasible optimum under preallocations, bucket bounds, and temporary allocation requirements. Clarify whether `--optimal` means optimal under that policy or over every otherwise feasible allocation. New pruning should preserve the intended model.

## Suggested order and later validation

1. Conditional critical path, mandatory minimum workload, and incremental weighted bound updates.
2. Candidate filtering with incumbent slack; shared offset intersection and cache-polarity probes.
3. Queue keys and global branch scoring; instrument bound components, candidate removals, probe time, first-incumbent time, and proof time separately.
4. Hall rank propagation and compulsory memory bounds if rank/offset branches dominate the remaining proof.
5. Feasibility subproblems for allocation and same-engine scheduling; cache/interface bucket decomposition.
6. Bucket templates, portable explanations, and interval dominance certificates for long sequence lengths.

Later validation should compare exact optimum and feasibility against small exhaustive models, including rollback, shared descendants, optional paths, views, empty engine sets, zero-weight buckets, and incumbent updates. Use current sequence-16 full/decode workloads for performance, then increasing bucket counts to distinguish fewer nodes from cheaper propagation. Stronger relaxations should be scheduled adaptively based on measured bound gap and pruning benefit.

## Primary references

- [OR-Tools disjunctive propagators](https://github.com/google/or-tools/blob/stable/ortools/sat/disjunctive.h): overload checking, precedence, edge finding, and related scheduling propagation.
- [OR-Tools cumulative energy propagation](https://github.com/google/or-tools/blob/stable/ortools/sat/cumulative_energy.h): energy-based relaxation and overload checking.
- [Schutt, Feydy, Stuckey: Explaining Time-Table-Edge-Finding](https://arxiv.org/abs/1208.3015): strong resource propagation with explanations for nogood learning.
- [Refalo: Impact-Based Search Strategies](https://citeseerx.ist.psu.edu/document?doi=6fa94b422157e949c809910abb91d74c457bf338&repid=rep1&type=pdf): learning search impact from domain reduction; search indexing exposed the paper text, although direct PDF access failed.
- [Hooker: Logic-based Benders Decomposition for Planning and Scheduling](https://johnhooker.tepper.cmu.edu/bendersKER.pdf): master/subproblem decomposition, relaxations, and inference-based cuts.
- [Fast and Optimal Extraction for Sparse Equality Graphs](https://doi.org/10.1145/3689801): exact extraction exploiting sparse graph structure. Potential reference for local extraction subproblems; its extraction objective does not directly solve this repo's scheduling/cache/allocation objective.

These references motivate the techniques. They do not establish performance gains on Gemma; the conditional recurrences, bucket budgets, and decomposition proposal above are derived from this repository's model.
