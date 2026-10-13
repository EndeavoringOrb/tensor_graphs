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

## Interval bound

The load bound treats the engines as if all work could overlap. Decode graphs are mostly sequential, so a second bound splits time into windows.

**Chain and windows.** Take the root's necessary groups ordered by descendant-set size and keep a chain `G_1, …, G_k` in which every member of `G_i` necessarily depends on `G_{i-1}` (`G_{i-1} ∈ GND(m)`). Let `t_i` be the earliest finish of an active member of `G_i` (`t_0 = 0`; the last window ends at the root's finish). The earliest member of `G_i` depends on an active member of `G_{i-1}`, so `t_{i-1} ≤ t_i`. A class `x` with `G_{i-1} ∈ GND(x)` starts after some active member of `G_{i-1}` finishes, so at or after `t_{i-1}`. A class in the selected sub-DAG of the earliest member of `G_i` finishes by `t_i`. So the classes counted for window `i` (necessarily after `G_{i-1}`; the recursion only collects them below a member of `G_i`) run inside `[t_{i-1}, t_i]`. Necessary groups are transitive, so any subsequence of the chain is again such a chain. The bound uses every `s`-th link for strides `s = k/4, k/16, …` (down to 16 windows) and keeps the best.

**Window value.** On every engine the window's load is at most its length. Any dependency chain of window classes runs sequentially inside the window. So for `μ ∈ [0, 1]` and any probability vector `λ`:

> `t_i − t_{i-1} ≥ μ · (duration of a chain) + (1 − μ) · Σ_e λ_e load_e(window)`.

`P(c)` extends the recursion of `L(c)`: the load part uses the same anchors, supports and charged groups (scaled by `1 − μ`). The chain part continues either through one anchor (whose `P` replaces its `(1 − μ) L`, so its load is still counted once) or through any child (`μ` times its window critical path, no load). A class's own duration enters the chain part only if it is counted in the window. The window value is `max(critical path, max over λ, μ of min over the members of G_i of P)` (golden-section search on `λ`, `μ ∈ {0, 0.2, …, 0.9}`). `μ = 0` is the load bound, `μ = 1` the critical path. The window values add up to a lower bound on the bucket makespan. The propagator stores `max(load bound, interval bound)` with the same float margin.

On Gemma decode the interval bound is 31.25 per decode bucket (load bound 29.99). Tests: `test.exe --relaxation-bound` checks strides 1, 2 and 3 against the minimum makespan over all selections and dispatch orders of 6000 random graphs. Oracle cases 33 (parallel work that does not depend on the opening link must not be charged to the window) and 34 (a sequential chain with engine choices: load bound 2.5, interval bound 5 = optimum) pin the inequality.

**Exact window value.** When a window's selections (below a member of the closing link) can be enumerated within 20000 search nodes, the window value is the exact minimum over them of max(chain, engine loads), shared classes counted once. Every plan's window is at least its chain and each engine's load.

**Cut DP (threshold).** For larger windows (up to 512 classes), a bisection on `T` proves that no selection has max(chain, loads) ≤ `T`:
1. Alternatives whose shortest chain below (head) plus shortest chain up to a target (tail) exceeds `T` are removed, to a fixpoint. They cannot appear in such a selection.
2. A dynamic program over the window's classes, parents first, enumerates the remaining selections exactly. A state is the set of required, not yet processed classes, so each class is processed once and shared classes count once. Each state keeps a Pareto set of points: engine loads plus, for each required class, the longest chain from the target down to it. A class may take an alternative only if that chain plus the alternative's duration and its children's shortest heads is at most `T`.
3. If the minimum largest load over the surviving selections exceeds `T`, the window is longer than `T`.

Limits keep it valid. Past 5000 states the DP gives up (no proof). Past 16 Pareto points the two closest are merged into their componentwise minimum, which only weakens pruning and lowers the computed minimum. On Gemma this exact DP needs at most a few hundred states per layer-sized window, because the dataflow cut is narrow.

**Window partitions.** Runs at strides 1, 2, 4, … (each also shifted by half a stride) give windows between different links. Any sequence of consecutive windows covering the chain, followed by the final window to the root, is a valid chain of disjoint time windows. An interval DP (`bestWindowPartition`) keeps the best such partition. The strides stop once a stride's own total falls more than 1% below the best.

**Identical buckets.** The propagator serializes each weighted bucket's relaxation (`relaxationSignature`) and computes each distinct relaxation once; decode buckets usually share one. Windows are evaluated in parallel.

On Gemma decode the interval bound is 36.06 per decode bucket (load bound 29.99), in about 1.9 s at the root. Tests: `test.exe --relaxation-bound` checks the exact and cut-DP window values (with enumeration disabled), and the window partition over strides 1–4 with every phase, against the minimum makespan over all selections and dispatch orders of 6000 random graphs.

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

7. Cache-set moves: local search records cached bases whose required selection blocks an otherwise load-reducing move. For those bases (all of them together, then each alone) the base is dropped from the cache set and local search runs again with the base releasable. The result is kept if no weighted bucket still uses the base's cache-only alternatives and the weighted makespan drops. The improved plan is validated as in step 5.

## Proof phase: largest box first

Once the root heuristic has an incumbent, the remaining search mostly makes proof progress. The queue then pops the open node with the largest box first: its stored log2 volume when volumes are measured, otherwise the shallowest node. Popping a node refreshes its box by propagation (closing it on a conflict) and splits it, so the regions that dominate the frontier shrink first. This only reorders the queue; no node is dropped.

On Gemma decode the accepted implementation (H-G4b) reaches its first incumbent (39.33) about 26 s into the program and its local optimum (37.02) immediately after. The final gap is 0.190 against the 29.99 bound, and the normalized primal-dual integral is 0.225 (baseline 0.992).
