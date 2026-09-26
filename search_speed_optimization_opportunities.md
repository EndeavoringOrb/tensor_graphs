# Search speed optimization opportunities

Investigation date: 2026-09-26 (b1acd87400ffa076eefe77608344278050db3754). Scope: recommendations only; no implementation changes.

## Main findings

The supplied nine-bucket Gemma run spends 397.57 seconds inside instrumented propagators by the displayed report. `TopologicalOrderPropagator` accounts for 213.75 seconds (53.8%) and `SelectionPropagator` for 152.56 seconds (38.4%). Together they account for 92.1%. This excludes branching, node restoration outside propagators, allocation, and logging overhead. The later iteration-900 line shows no feasible incumbent (`Best: inf`), so incumbent-based pruning cannot help yet.

The first priorities are to scope propagation to changed buckets, avoid rebuilding graph structures and repeated reachability searches, and obtain a verified feasible incumbent early without removing alternative branches.

## Objective and optimality constraint

`SearchEngine::evaluateMakespan` currently sums selected operation costs per engine and returns the maximum engine total. It sorts operations by their `START` values but does not simulate dependency waits or elapsed execution time. `START` variables are integer ordering slots; precedence uses `+1`, and engine exclusion forbids equal slots. This distinction matters for every proposed scheduling bound.

The historical `ExtractorJacksonCarlierRule` includes critical-path and preemptive Schrage bounds on elapsed makespan. Those cannot be copied into the current workload objective without a new admissibility argument. Its incremental data structures and cheap-before-expensive bound evaluation are independently useful.

For example, a CPU operation costing 10 followed by a dependent GPU operation costing 10 has current objective 10, but elapsed makespan 20. Pruning it using a critical-path bound of 20 against a workload incumbent of 15 would discard an improvement.

`thinking.txt` describes a weighted execution-time objective and a unified search with freely interleaved decisions. The recommendations below keep that unified search. They distinguish improvements valid for today's workload objective from bounds that require an explicit change to elapsed makespan.

## Priorities

| Priority | Opportunity | Expected benefit | Optimality requirement |
| --- | --- | --- | --- |
| 1 | Bucket-local events and precomputed graph tables | Removes repeated scans of unaffected buckets from both dominant propagators | Wake every actual dependency, including shared cache changes and rollback |
| 2 | Reuse selected-graph reachability and topology | Removes repeated DFS and hash-table reconstruction | Invalidate or restore summaries when selected edges change |
| 3 | Verified incumbent completion and locality-aware node priority | Enables objective pruning; reduces time spent replaying ancestors | Keep all unexplored alternatives; heuristic failure proves nothing |
| 4 | Mandatory-child propagation and required-operation workload bounds | Narrows selection domains and raises early lower bounds | Count only unavoidable work, once per canonical e-class |
| 5 | Candidate-specific cost filtering | Deletes alternatives that cannot beat the incumbent | Use admissible conditional bounds with correct bucket weights |
| 6 | Bidirectional precedence, engine Hall intervals, memory compulsory overlap | Detects infeasibility before fixing every start/offset | Propagate only constraints true for every completion |
| 7 | Conflict learning, compact state, scheduling symmetries | Reduces repeated work and equivalent assignments | Explain conflicts; prove representative coverage |
| Conditional | Critical paths and Jackson-Carlier/Schrage | Stronger elapsed-makespan bounds | First align leaf evaluation, bounds, and runtime scheduling semantics |

These are priorities inferred from source and the supplied profile, not measured speedups. As an illustration, making the two dominant propagators 5 times faster would reduce instrumented time to about 104.5 seconds, approximately 3.8 times faster within that component. Whole-search speedup depends on the unmeasured costs and any change in node count.

## 1. Make propagation proportional to the change

### Bucket-local events first

Evidence: `SearchEngine::runPropagators` already has a dirty-variable worklist and `Propagator::watches(VarType)`. However, it collapses each changed variable to its type; each awakened propagator then loops over every bucket. A selection decision in bucket 0 therefore wakes selection/topology scans in buckets 1 through 8 as well. The recent commits `de76b2c` and `67b044e` already eliminated full empty-domain scans and added type-level scheduling; do not repeat those optimizations.

Proposed progression:

1. Queue `(propagator, bucket)` work and retain changed variable IDs. Recompute only affected bucket bounds.
2. Precompute reverse cache dependencies: one shared `CACHED` variable can affect several buckets, but only specific CACHE/SCATTER alternatives within them.
3. Distinguish removal of a selection value, becoming required, becoming fixed, start lower/upper-bound changes, and offset changes. For example, a start change needs precedence-bound updates, not rebuilding the selected graph or rechecking every candidate for cycles.
4. Run cheap selection/support propagation to a local fixed point before expensive graph passes, then repeat until the whole worklist is empty.

All initial constraints still run. `SearchState::backtrackTo` currently marks restored domains dirty; keep that behavior or restore derived state from the same trail. Widening domains during rollback must reintroduce removed supports. An incumbent update is another event even though no variable domain changed.

The existing type-based implementation is a useful reference for differential tests. Event-driven incremental filtering is also illustrated in Choco's [constraint implementation tutorial](https://choco-solver.org/tutos/constraints/).

### Dense, immutable graph tables and reusable scratch storage

Evidence: propagators repeatedly call `findConst`, `getChildren`, and hash-map lookups; allocate visited sets; and reconstruct selected parents, views, and engine groups. Selection, topology, memory, and `HeuristicBrancher::chooseSchedule` duplicate portions of this work.

After pruning, build compact per-bucket tables containing canonical and deduplicated children, reverse candidate users, selected/start/offset variable IDs, engine indices, view flags, sizes, and costs. Keep original enode-index mappings stable. Use reusable vectors and generation-stamped visitation arrays instead of a new `unordered_set` for each traversal. Maintain selected parents/topological order once per selection version and share them with scheduling and memory analysis.

Replace the dirty-variable hash set with a vector plus a per-variable queued flag if allocation/hash overhead is measurable. Retain separate flags for different consumers if they consume events independently. Do not create a second copy of every large table per search node.

### Topological cycle checks

Evidence: `TopologicalOrderPropagator::propagate` calls `has_fixed_path(child, candidate_class)` for each child of each surviving candidate, rebuilding a frontier and visited set each time. The brancher performs similar path tests. The source explains a plausible hotspot; the aggregate profile does not yet isolate its share of the 103 ms per call.

Useful alternatives, in increasing complexity:

- Compute a fixed-graph topological rank once and use it to reject impossible reachability queries cheaply; ambiguous queries still need a real reachability test.
- Group queries by source or destination so one traversal answers many candidate checks.
- Maintain fixed-edge transitive reachability as bitsets, updating only ancestors/descendants affected by a newly fixed choice. A full closure uses roughly `N*N/8` bytes per bucket, so benchmark grouped traversals before committing to it.
- When only start bounds change, reuse the graph and run only precedence propagation.

Keep the current selected-cycle contradiction test. Reachability summaries must describe forced edges, not the union of mutually exclusive alternatives. Rollback must undo inserted edges and invalidate or restore closure; a closure from a sibling branch is unsafe.

### Selection reachability

Evidence: `SelectionPropagator` reconstructs root reachability through every currently possible alternative on every invocation, then scans all structurally reachable classes to fix unsupported ones to zero.

Scope this pass to changed buckets immediately. Then consider decremental root reachability using reverse candidate edges and dirty regions. Root connectivity needs care: simple incoming-support reference counts incorrectly retain an unreachable cycle because its nodes support one another. Use a sound root-reachability/SCC algorithm or periodically rebuild affected regions. Reusable bucket-local BFS is a valuable intermediate implementation.

## 2. Get a feasible incumbent and avoid unnecessary replay

The displayed run has `Best: inf`. `CostLowerBoundPropagator` can report a bound but cannot prune against a finite target. The historical Jackson–Carlier rule likewise returned immediately when there was no incumbent.

Use a bounded completion attempt from a promising partial state: choose compatible implementations, complete cache choices, produce a topological schedule, and assign offsets. Validate the entire completed plan before installing its cost as the incumbent. A memory-aware choice may succeed sooner than the cheapest operation-by-operation choice. Completion failure must leave the original search alternatives available.

This can remain a heuristic attached to the unified search; it need not introduce an outer extraction/scheduling/allocation decomposition. Run it at the root and selected promising nodes, with a measured budget, rather than at every node. A compatible cached plan is another incumbent candidate, but the user's log reports an invalid cache, so it cannot be assumed usable.

Evidence for replay cost: `restoreNode` backtracks to the lowest common ancestor, then reapplies decisions and runs propagators at every intervening node. The supplied report already has 1,275 lower-bound evaluations before the subsequent iteration-900 line. This indicates more propagation calls than popped search iterations, although the log alone does not attribute each call to restoration.

Current priority is `right_lb + 1` for deferred alternatives and `left_lb - depth*0.01` for preferred children. This encourages diving but does not guarantee staying on the current path. Consider a priority component based on restoration distance, with a stronger early preference for reaching a feasible completion. Keep heuristic priority separate from admissible lower bounds.

Measure replayed ancestors, propagation during restoration, and restored trail entries. If replay dominates after bucket-local propagation, consider sparse checkpoints or versioned propagated deltas. Replaying only decisions is simpler but expensive; replaying cached propagation is safe only when its assumptions and incumbent threshold remain valid. Do not restore a snapshot from another logical node merely because its trail length matches.

## 3. Stronger selection propagation

### Required children before fixing an implementation

Currently, only a fixed nonzero selection forces its children to be selected. If a required class has alternatives `f(a,b)` and `g(a,c)`, then `a` is required even before choosing between them. Intersect canonical child sets of the remaining alternatives and remove zero from every common child. Repeat as alternatives disappear. Skip optional parent classes while zero remains possible.

Maintain reverse supports so a child becoming impossible immediately removes all parent alternatives requiring it. The topology propagator already performs that check during its broad scan; move the same deduction to the affected candidates. This is both faster and easier to combine with common-child propagation.

These implications can expose substantial unavoidable work to the cost bound. Deeper AND/OR mandatory-descendant analysis is a later extension; shared descendants and alternative-dependent cycles need explicit treatment.

### Conditional precedence support

For a selected parent alternative and one of its required children, remove child implementation values whose start interval cannot precede the parent's interval. Conversely, remove a parent alternative if some child has no compatible nonzero implementation remaining. Start with simple pairwise support; keep a cached supporting alternative and search again only when that support disappears.

Pairwise support is a relaxation: preserving a value because it has local support can miss a contradiction, but removing a value with no possible support is sound. Do not infer that a union of optional edges must all hold simultaneously.

## 4. Bounds that are admissible for the current objective

### Include required but unfixed operations

`CostLowerBoundPropagator` currently counts only fixed nonzero selections. This explains why many unresolved required operations contribute nothing. It also computes the same bound inside `propagate` and again in `runPropagators` after the fixed point.

Let `R_b` be the distinct required canonical classes in bucket `b` (selection domain excludes zero). Let `D_c` be a class's remaining implementation values. Define `load(c,a,e)` as the implementation's cost if it uses engine `e`, otherwise zero. For nonnegative costs and bucket weights:

```text
W[b,e] = sum over c in R_b of min over a in D_c load(c,a,e)
L[b]   = max over engines e of W[b,e]
L      = sum over buckets b of weight[b] * L[b]
```

Every completion incurs at least each per-engine minimum, so this is admissible. Fixed selections are naturally included once. Do not add the old fixed-work bound again. Shared children are charged once by canonical ID; do not sum recursive subtree costs, which can double-count a shared DAG.

Maintain per-class contributions and per-bucket totals incrementally; reuse the result between propagation and lower-bound reporting. A domain restriction can increase its minimum or make a class required. Backtracking reverses both. Use cost semantics consistent with the evaluator, including missing/invalid costs; today's `INF -> 1` evaluation fallback should not be silently mixed with a bound that treats `INF` as infeasible.

### Account for flexible engine choices

Per-engine minima can be zero when each required task can choose CPU or GPU. A cheap stronger family of bounds uses nonnegative engine weights `alpha[e]` summing to one:

```text
L[b,alpha] = sum over c in R_b of
                min over a in D_c sum over e alpha[e] * load(c,a,e)
L[b]      = max over tested alpha vectors of L[b,alpha]
```

Proof: maximum engine workload is at least any weighted average of engine workloads. Minimizing each required class independently only relaxes the problem. Engine unit vectors recover the previous bound; equal weights capture work movable between engines. With two engines, a small fixed set of CPU/GPU weights is inexpensive. Example: two required tasks each cost 10 on either engine. Per-engine minima give zero; equal weights give 10, the optimal balanced workload.

A fractional assignment relaxation can strengthen this further. Use a certified lower bound or feasible dual solution; an arbitrary approximate primal solution of a minimization relaxation is not a pruning certificate. Test the cheap weighted bounds first.

### Filter individual candidates against the incumbent

For `selected[c] = a`, replace that class's minimum contribution by the candidate's contribution and include any additional descendants proven required under this assumption, with deduplication. Remove `a` if the resulting global bound cannot improve the incumbent. This prunes before branching rather than merely rejecting a whole fixed prefix.

For a bucket-local candidate, the cutoff must account for other buckets:

```text
sum over k != b weight[k] * L[k] + weight[b] * L[b | c=a] >= incumbent
```

Do not compare an unweighted bucket bound to a weighted total or add overlapping bounds together. For a zero-weight bucket, objective filtering provides no restriction but feasibility still matters. Negative weights would invalidate these monotone lower bounds and should be excluded from the model contract.

Start with cheap contribution tests; reserve full assumption/propagation probes for candidates near the cutoff or with high influence. An incumbent improvement must wake this filtering even if no selection domain changed.

## 5. Stronger schedule and memory propagation

### Precedence in both directions

Current topology propagation only pushes `min(parent_start) >= min(child_start) + 1`. Also propagate `max(child_start) <= max(parent_start) - 1`, using a reverse topological pass. This preserves the existing unit-slot model and becomes useful as soon as branching or engine constraints tighten an upper bound. Keep affected-edge queues instead of rescanning all selected edges.

### Engine exclusion as AllDifferent

Current engine propagation detects duplicate fixed starts and skips occupied slots at the lower boundary of another interval. Also skip occupied upper-bound suffixes and detect Hall intervals among definitely selected operations on each engine: if `k` starts are confined to fewer than `k` slots, fail; if they occupy exactly `k` slots, exclude those slots from the other starts. For example, three tasks each restricted to `[4,5]` are impossible without fixing any task. Two tasks in `[4,5]` force a third task with domain `[4,8]` to start at least at 6. This is established AllDifferent filtering; see [The AllDifferent Constraint with Precedences](https://cquimper.github.io/publications/2011-03-16-tw.pdf).

Apply this per engine, and include a multi-engine operation in every engine it requires. Optional operations require guarded constraints. Current interval domains cannot represent interior holes: retain a safe interval hull, introduce an appropriate sparse representation, or branch on the disjunction. Do not delete the whole interval merely because it contains blocked interior values.

Weighted-duration edge-finding is not a direct replacement here: current starts are dispatch slots, not times in cost units. Unit-slot overload and precedence reasoning are immediately applicable; elapsed-time scheduling constraints belong with the objective change in section 6.

### Memory: compulsory overlap before fixed starts

Current memory analysis rebuilds selected view aliases and consumer graphs, then compares allocation pairs. It largely waits for producer starts to be fixed. Cache the graph/alias part per selection version and partition allocations by memory space. Sweep live intervals to generate overlapping pairs instead of comparing every pair when most lifetimes do not overlap.

For stronger pruning, derive intervals during which an allocation must be live in every completion. With producer latest start `LS` and a valid lower bound `EE` on its lifetime end, `[LS, EE)` is compulsory when `LS < EE`. Sum unique physical allocation sizes over compulsory overlap; reject a memory-capacity overflow. Views share their base allocation and must not be counted again. Include reserved/preallocated storage exactly once.

For two allocations proven simultaneously live, spatial non-overlap implies `offsetA + sizeA <= offsetB OR offsetB + sizeB <= offsetA`. If one orientation is impossible from current bounds, propagate the other even when neither offset is fixed. If spatial disjointness is impossible, a temporal ordering disjunction can be explored instead, preserving both feasible orders. Reusing an input buffer requires the runtime's actual safe-in-place and last-reader rules.

For an operation requiring several distinct input buffers and an output, a local minimum working-set bound can reject an implementation before scheduling. Establish which buffers are guaranteed distinct and co-live, and account for views, preallocation, and permitted in-place execution. Summing all reachable tensors would overestimate mandatory memory.

These additions target later search and tighter-memory cases; memory propagation is only about 1.4% of the supplied instrumented time at this point. Improve it for pruning benefit, not because this profile identifies it as the main per-call bottleneck.

### Shared cache constraints

Cache propagation already forces CACHE selections to enable the corresponding cache variable, removes CACHE/SCATTER choices when caching is disabled, and checks fixed cache bytes. Maintain these implications through indexed users rather than all-bucket scans. Intersect cache requirements of every remaining alternative of a required class to discover mandatory cache choices earlier, including dependencies represented inside fused operations where applicable.

The common cache variables couple buckets, so nine independently optimized bucket plans cannot simply be combined. Independent per-bucket relaxations are valid lower bounds when coupling is dropped; independent feasible incumbents require a compatible common cache assignment and a joint memory check.

## 6. What to recover from ExtractorJacksonCarlierRule

### History inspected

- `b7aea21` introduced the rule in `tensor_graphs_cpp/core/plan/extractor.hpp`.
- `264c390` ("faster ExtractorJacksonCarlierRule") replaced repeated work with cached class critical-path minima, flat engine state, reusable Schrage scratch vectors, and push/pop trails.
- `d7da4f1^` retains the rule and `computeSchragePreemptiveBound`; `d7da4f1` deletes the old extractor and its pruning test suite.

The rule's checks were:

1. Candidate critical path plus its forced delivery tail.
2. Critical paths of still-required frontier classes plus their tails.
3. Selected engine workload plus candidate cost.
4. A preemptive single-engine relaxation of selected tasks plus the candidate, represented by `(release, processing_cost, tail)`.

It ran cheap checks first. Before Schrage, `max_release + total_work + max_tail < incumbent` skipped the expensive check because even that upper envelope of the relaxed optimum could not cause a prune. The envelope was a filter for bound evaluation, not a feasible full-plan upper bound. The source's Schrage implementation sorts releases, uses a max-heap of tails, and permits preemption at release events; its sorting/heap work is `O(n log n)` per evaluation. This is a relaxation of elapsed-time scheduling, whose classical foundation is Carlier's [The one-machine sequencing problem](https://www.sciencedirect.com/science/article/pii/S0377221782800076).

### Reuse immediately

Reuse the organization: flat engine indices, incremental workload totals, precomputed immutable metadata, reusable scratch arrays, reversible updates, and cheap tests before expensive ones. Candidate-specific workload filtering in section 4 is the part of its pruning logic directly compatible with today's objective. Setting all releases and tails to zero makes the preemptive bound collapse to engine workload, so that adaptation would provide no additional pruning strength.

The historical `on_pop` rescans remaining tasks to reconstruct `max_r` and `max_q`. Store previous maxima in undo frames if reviving that stack design. The historical rule assumed push/pop traversal; the current priority queue can jump between branches, so all task lists, tails, totals, and caches must follow `SearchState` restoration. Cached quantities depending on a child domain must be refreshed when that domain changes, even if the task itself was selected earlier.

### Conditional port for elapsed makespan

If the intended objective is elapsed execution time, first define and implement a matching leaf evaluator: schedule each operation after its dependencies and all required engines become available, and advance those engines by its duration. Reconcile ties, view operations, copies occupying multiple engines, and runtime synchronization. Changing the objective changes which solution is optimal; it is a separate semantic change, not an optimization preserving the current optimum.

Then use, per bucket:

```text
L[b] = max(workload_bound,
           mandatory_critical_path_bound,
           max over engines preemptive_Schrage_bound[engine])
L    = sum over b weight[b] * L[b]
```

Use only forced tasks, plus the candidate under test. Releases must be lower bounds on when inputs can be available; tails must be lower bounds on unavoidable work after the task. Relaxing cross-engine constraints and permitting preemption expands the feasible set, which is why the bound can be safe. Do not include all alternative implementations as simultaneous jobs. Take maxima of overlapping bounds rather than summing them.

For candidate filtering on a positive-weight bucket, use residual target `(incumbent - other_buckets_lower_bound) / weight[b]`. Apply the historical cheap upper-envelope filter against that target. The historical rule is a reference implementation, not a blanket admissibility proof for a changed graph/cost/cache model.

## 7. Search-space and bookkeeping improvements

- **Incremental branch selection.** `chooseSelection` scans all reachable classes for the smallest domain, and `chooseSchedule` rebuilds topology before selecting each start or offset. Maintain queues of unresolved required classes and ready operations. Preserve the same allowed values; this changes decision order only.
- **Conflict-directed ordering and learned forbidden combinations.** Record the selections/starts/offsets that explain a cycle, engine collision, memory conflict, or budget violation. Reuse those explanations to avoid the same failure under unrelated decisions. Initially use conflict activity only to order branches. Pruning requires a sound explanation; a propagator name or an unsuccessful greedy allocation is insufficient. Cost-based learned constraints must retain the cutoff assumptions under which they were derived.
- **Probe selectively.** Temporarily assign a high-impact implementation/cache choice, propagate, then restore. Remove it only on a proved contradiction or admissible cutoff; a probe stopped for budget reasons is inconclusive. Running full propagation for every candidate would multiply today's main bottleneck.
- **Reduce representation overhead.** `all_nodes` retains every generated node; each node has shared ownership and a vector for often one decision. Compact node/decision arenas and reclaimable closed subtrees can reduce allocation and memory pressure while retaining ancestry required for restoration. Track peak live/open nodes and bytes before redesigning.
- **Avoid enumerating irrelevant assignments.** Unselected alternatives' start variables need no decisions, which the brancher already largely respects. Fixed preallocated offsets and selected views can also be represented through their physical base where semantics permit. Conditional aliasing must remain reversible.
- **Exploit equivalent schedules carefully.** The current objective depends only on selections, so every feasible schedule/offset assignment of the same selections has the same cost. Once a feasible witness exists, remaining completions of that exact selection vector cannot improve it. A certified no-improvement constraint or suitable memo can skip them; do not discard any region that still permits a different selection vector. Gap-free relabeling of starts may remove idle-slot symmetries, but preserving all order, tie, lifetime, and bound semantics needs proof. A greedy schedule that fails does not prove the selection vector infeasible.
- **Reduce late-run logging.** Both search progress conditions contain `iterations >= 1345`, enabling output every iteration thereafter. Make this an explicit debug setting. The supplied excerpt is earlier, so this does not explain its current bottleneck.

Prefer these incremental improvements over introducing a second outer solver or a strict extraction/scheduling/allocation pipeline. They fit the architecture described in `thinking.txt`.

## 8. Conditions required to claim optimality

### Search termination and scope

The current `solve` stops once its time budget has expired and a feasible incumbent exists; a zero budget returns after the first feasible solution. It may exceed the budget while no solution exists. Therefore `--min-compile-time 900000` is a long search budget, not an optimality certificate.

Report the incumbent upper bound, minimum admissible bound across all open regions, gap, and termination reason. Queue priority contains heuristic biases, so its first element need not have the smallest admissible lower bound; maintain a separate bound index if needed. Exact completion requires exhausting all potentially improving regions or proving their lower bounds cannot beat the incumbent. `LB >= incumbent` can discard tied optima while preserving at least one optimal solution; enumerating every optimal plan would require a different policy.

The optimum is relative to the generated/pruned e-graphs, page alignment, memory constraints, and cost model. These recommendations do not prove earlier e-graph pruning sound or make estimated costs equal to measured runtime.

### Correctness questions found during this review

Resolve these before relying on stronger pruning. These are source-level observations, not runtime reproductions:

1. **Failed restoration state.** In `restoreNode`, a failed replay sets `current_node_id` to the parent but returns without explicitly rolling state back to that parent's marker. Verify that subsequent restoration cannot treat contaminated domains as that parent, particularly the early return when target and current IDs match. Incremental caches would amplify any mismatch.
2. **Memory lifetime estimates.** For an unfixed consumer, `MemoryNonOverlapPropagator` extends the producer's lifetime to `max_finish_time` over other fixed operations. That is not generally a lower bound on its last use: the consumer may legally finish before an unrelated late operation. For example, a producer at 0, consumer allowed at 1, and unrelated operation at 10 need not keep the producer live until 11. Use guaranteed-live intervals for pruning and possible-live intervals only for heuristics. Check which current branching/preallocation cases can expose this issue.
3. **Feasible-incumbent validation.** The no-more-branches path evaluates and extracts directly. An independent validator should check selected dependencies, acyclicity, cache requirements, engine exclusions, alias-aware lifetimes, and physical memory caps before an incumbent enables broad pruning. In particular, planner offset initialization uses `max(min_p, max_p)` even if a buffer cannot fit in available capacity; nonempty domains alone do not certify capacity.
4. **Numeric consistency.** Leaf and bound sums use `float` and can traverse operations in different orders. Near-equal totals, incremental add/subtract drift, invalid costs, or rounding can turn a supposed lower bound into an overestimate. Define consistent arithmetic and conservative error bounds; higher precision helps but is not by itself a proof. Do not use an arbitrary positive tolerance to prune improvements while claiming exactness.
5. **Selection-mask boundary.** The planner permits 31 enodes but forms `(1u << (n_enodes + 1)) - 1`; at 31 this shifts a 32-bit value by 32. Cover that boundary explicitly when checking domain completeness.

## 9. Validation and measurement plan

No implementation, rebuild, solver benchmark, or test execution was performed for this document. The speed ranking uses the supplied log and source inspection. Existing `debug_artifacts/search_*` logs include historical tests and older solver architecture; they are not evidence that the current code passes those tests. The current tracked test entry point and test headers do not contain the removed pruning/search suite.

For a future implementation:

1. Capture a deterministic planner/search input after saturation and pruning, including e-graphs, costs, weights, cache candidates, and memory limits. Record the source revision and build flags. Test cold search separately from cache loading.
2. Differentially compare optimized propagation with a from-scratch reference after random assignments, rollback, and priority-queue jumps. For implementation-only speedups, fixed-point domains should match. Stronger propagators may remove more values, but an exhaustive tiny oracle must show that every removal is valid.
3. Exhaustively enumerate small graphs with sharing, alternative cycles, multiple engines, multiple buckets sharing caches, view chains, and tight memory. For every partial assignment, verify `lower_bound <= best feasible completion`; compare the final optimum with pruning disabled. Use an independent oracle so two versions do not share the same bug.
4. Include targeted cases for common-child forcing, unreachable support cycles, Hall intervals, backward precedence, cache budget propagation, unfixed last consumers, late incumbent updates, failed restoration, 31-enode masks, and near-equal floating costs. Test the CPU-to-GPU chain example so elapsed-time bounds cannot accidentally enter the workload objective.
5. Compare stages separately: bucket events; dense metadata; topology reuse; incumbent heuristics; stronger workload bounds; candidate filtering; schedule/memory filtering. Record root bound, time to first feasible solution, best cost versus time, time to proof, remaining gap, popped nodes, domain reductions, contradictions, replay count, per-bucket/propagator time, and peak memory. Iterations per second alone can reward weaker pruning.
6. Run the supplied Gemma configuration with bounded diagnostic budgets first, plus a tight-memory case and a small instance that can finish to proven optimality. Keep the cost model and candidate space fixed when comparing solver improvements. If running Python, use `.venv/Scripts/python.exe`; use `.venvx64/Scripts/python.exe` only for an OR-Tools comparison.

## Source map

Inspected working-tree baseline: `b1acd87` (`change test`). Line numbers below refer to that baseline.

| Area | Source |
| --- | --- |
| Propagator worklist and type-level dependencies | `tensor_graphs_cpp/core/plan/search_engine.hpp:139` |
| Node restoration | `tensor_graphs_cpp/core/plan/search_engine.hpp:76` |
| Workload objective | `tensor_graphs_cpp/core/plan/search_engine.hpp:278` |
| Search termination, incumbent, priorities | `tensor_graphs_cpp/core/plan/search_engine.hpp:436` |
| Selection and root reachability | `tensor_graphs_cpp/core/plan/propagator.hpp:51` |
| Cache propagation | `tensor_graphs_cpp/core/plan/propagator.hpp:173` |
| Cycle checks and precedence | `tensor_graphs_cpp/core/plan/propagator.hpp:300` |
| Engine slot exclusion | `tensor_graphs_cpp/core/plan/propagator.hpp:507` |
| Aliases, lifetimes, memory overlap | `tensor_graphs_cpp/core/plan/propagator.hpp:617` |
| Current cost lower bound | `tensor_graphs_cpp/core/plan/propagator.hpp:966` |
| Branching and repeated topology construction | `tensor_graphs_cpp/core/plan/brancher.hpp:185` |
| Trail and dirty domains | `tensor_graphs_cpp/core/plan/search_state.hpp:153` |
| Variable domains and capacity initialization | `tensor_graphs_cpp/core/plan/planner.hpp:1820` |
| Domain representations | `tensor_graphs_cpp/core/plan/domain.hpp` |
| Architectural intent | `thinking.txt` |

Recommended first implementation slice: bucket-local propagation plus precomputed canonical adjacency and reusable scratch storage, followed by required-child propagation and incremental required-work bounds. Develop verified incumbent completion alongside these. Revisit the historical Schrage bound only after deciding whether the solver should optimize workload or elapsed makespan.
