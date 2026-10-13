# Selection-relaxation bound and root primal heuristic

Code: `tensor_graphs_cpp/core/plan/relaxation_bound.hpp`, `core/plan/propagators/relaxation_bound.hpp`, `core/plan/primal_heuristic.hpp`. Tests: `test.exe --relaxation-bound` (randomized brute force), oracle cases 29–32.

## The relaxation

For one bucket, drop memory, dispatch order and the cache budget, and keep only the selection: the root is active, an active class picks one of its remaining alternatives (the root domains after root propagation), and the chosen alternative's children are active. CACHE/INPUT alternatives are free and always allowed. Each alternative has its simulated duration (`MakespanSimulator`: INPUT/CACHE and views take 0, an infinite cost simulates as 1, an empty engine list runs on CPU engine 0) and occupies its engines.

Every engine executes its operations one after another, so for any plan the bucket makespan is at least `max_e load_e`, where `load_e` is the total duration of selected operations on engine `e`. For any probability vector `λ` over engines, `max_e load_e ≥ Σ_e λ_e load_e = Σ_selected w_λ(a)` with `w_λ(a) = duration(a) · Σ_{e ∈ engines(a)} λ_e`. So any lower bound `B_λ` on the minimum weighted cost of a selection is a lower bound on the bucket makespan, and on the objective after weighting buckets.

## The recursion

Groups: classes linked by COPY_TO alternatives hold the same logical tensor (a CPU and a GPU copy); `group(c)` is a union-find class. `GND(c)` (necessary groups) is the intersection over `c`'s alternatives of the union of each child's group and `GND(child)`; if `c` is active, then for every `G ∈ GND(c)` some member of `G` is active in `c`'s selected sub-DAG (induction on the DAG). Since a COPY_TO only copies within its group, an active group has an active member whose selected alternative is not a COPY_TO; its weight is at least `mp(G)`, the minimum weight of the group's non-COPY_TO alternatives.

For each class `c`, the recursion computes `L(c)` and a support set `S(c)` with the invariant

> for every plan in which `c` is active, `L(c) ≤` the weighted cost of the plan's active classes that lie in `S(c) ∩ T(c)`, where `T(c)` is `c`'s selected sub-DAG.

`L(c) = min over alternatives a of [ w(a) + Σ_{k ∈ A} L(k) + Σ_{G ∈ E} mp(G) ]`, where `A` is a set of positive children with pairwise disjoint supports, and `E` is the set of necessary groups of `a`'s children, other than `group(c)`, whose members avoid every support in `A`. Proof of the invariant: the plan selects some alternative `a`. Its own weight is paid by `c`, which is in `S(c)` and in no child's sub-DAG. Each anchor `k` is active, and by induction contributes at least `L(k)` from `S(k) ∩ T(k) ⊆ S(c) ∩ T(c)`; these sets are disjoint. Each group in `E` has an active producing member in `T(c)`, which is a distinct class for each group (groups are disjoint), lies in `S(c)`, and lies outside the anchors' supports and outside `c`'s own group. `S(c)` is `{c}` plus the union, over all alternatives, of every positive child's support and every charged group's members (a superset is always safe). A class whose `L` is zero gets an empty support. Several anchor orders are tried and the best sum is used.

**Regions.** Shared history (for example the residual stream) appears below every child, so supports overlap and only one child could be an anchor. To avoid that, take the root's necessary groups ordered by the size of their descendant sets and keep a containment chain `G_1, G_2, …, G_k`, where `Desc(G_i)` is the union of the members' descendant sets. Region `R_i = Desc(G_i) \ Desc(G_{i-1})`, and the final region is `Desc(root) \ Desc(G_k)`. In each region the recursion runs with every class below the region given `L = 0` and an empty support. A group may be charged only if all its members lie in the region. Then all supports used in region `i` lie in `R_i`, and the regions are disjoint. Every plan has an active member `m_i` of each `G_i`, so the plan's weighted cost is at least `Σ_i min_{m ∈ G_i} L_i(m) + L_final(root)` (a link whose group has a member below the region contributes 0).

`B_λ` is that sum. The propagator maximizes over `λ` (golden-section search on two engines, a fixed set of weightings otherwise); every evaluated `λ` is valid by itself.

**Floating point.** The simulator accumulates engine finish times in float. Over `n` operations it can undershoot the exact sum by a relative `n·2^-24`, so the stored bound is `B·(1 − (n + 64)·1.2e-7) − 1e-7`, rounded toward zero.

## Where the bound lives

`RelaxationBoundPropagator` runs once, as a GLOBAL batch at the quiescence of the first (root) propagation, and stores `SearchState::bucket_relaxation_lower_bounds`. `updateLowerBoundBucket` takes the maximum with the other bucket bounds. Nodes below the root only have smaller domains, and the relaxation over a superset of plans is still a lower bound, so the stored values stay valid in the whole tree and the certified bound (`docs/core/search_gap.md`) never drops below their weighted sum. Zero-weight buckets are skipped. Cyclic buckets get no bound. A bucket whose root has no feasible alternative in the relaxation has no plan at all, so the propagator reports a conflict.

On Gemma decode the bound is 29.99 per decode bucket (CP-SAT proves the exact relaxation optimum, min-max engine load, at 30.36; the old root bound was 0.53). It takes about 60 ms for all buckets.

## Primal heuristic

After root propagation, `SearchEngine` builds complete plans and evaluates them like any leaf:

1. Per bucket and per engine weighting `λ`, the recursion above is evaluated and each class's region value is used as its marginal cost estimate. Classes are visited parents first. An active class in a weighted bucket picks the alternative minimizing its weight plus the estimates of children that are not yet active.
2. Caches: a CACHE/SCATTER-dependent alternative may only be used for a base whose cache variable still allows 1. The cache set is the set of bases whose dependent alternatives the weighted buckets chose. A cached base must be selected in every bucket and fit the space's cache budget; offending bases are disallowed and the selection is repeated.
3. Zero-weight buckets (for example the "full" bucket) do not count in the objective, but every class whose base is cached must be selected there too. Their selection is a covering extraction: parents first, each active class picks the alternative whose children can still reach the most inactive target classes (ties by estimate).
4. Dispatch order: list scheduling with the simulator's semantics (earliest possible start, ties by the longest remaining path).
5. The cheapest plan by simulated weighted makespan is applied below the root: cache and selection variables are fixed, and start ranks follow the dispatch order inside their windows (pinned starts are kept). Then every propagator runs. The plan becomes an incumbent only if propagation succeeds and every variable is fixed; its cost is computed by the normal leaf evaluation. The state is then restored to the root.
6. Local search: from every distinct initial plan (one per weighting), each weighted bucket is improved in parallel by single-class switches. Newly needed children are completed greedily, unneeded classes are released, cached bases stay selected, and cache-only alternatives stay limited to cached bases. Moves that raise the maximum engine load are skipped; others are kept only if the list-scheduled makespan decreases. A start stops when a full pass finds no improvement (or at the program deadline). Each improved plan is validated as in step 5.

The heuristic never changes a domain or bound that survives the restore, so it does not affect bound semantics.

## Proof phase: largest box first

Once the root heuristic has an incumbent, the remaining search mostly makes proof progress. The queue then pops the open node with the largest box first: its stored log2 volume when volumes are measured, otherwise the shallowest node. Popping a node refreshes its box by propagation (closing it on a conflict) and splits it, so the regions that dominate the frontier shrink first. This only reorders the queue; no node is dropped.

On Gemma decode the accepted implementation (H-G4b) reaches its first incumbent (39.33) about 26 s into the program and its local optimum (37.02) immediately after. The final gap is 0.190 against the 29.99 bound, and the normalized primal-dual integral is 0.225 (baseline 0.992).
