# Gemma search research p3: additional bounds, search-structure changes, and bucket-axis scaling

Research only. No code was built or run; every item is a hypothesis to measure. Evidence comes from reading the repo (`search_engine.hpp`, `brancher.hpp`, `types.hpp` `MakespanSimulator`, `kernels.hpp`, `bindings.cpp`, `SEARCH_PERFORMANCE.md`, an old `analysis.txt` log) and the web sources listed at the end.

This note does not repeat items already covered in p0/p1/p2: conditional critical path, mandatory minimum workload, incumbent-driven candidate filtering, incremental weighted bound sums, lexicographic queue keys, generic LBBD, and bucket fingerprinting. p1/p2 correctly note that Hall-interval pruning of START ranks already exists (`start_bounds.hpp`).

## Code observations this note builds on

1. **Makespan depends on much less than the search fixes.** `MakespanSimulator` (`types.hpp:1803`) is append-only per engine: `start = max(children_finish, engine_free)`. Makespan is therefore the longest path in the data-dependency graph plus per-engine order chains. The total rank order that the brancher fixes one START at a time (`chooseSchedule`, `brancher.hpp:954-972`, `setBinaryDecision(start, min)`) is mostly redundant: many rank assignments give identical per-engine sequences. Offsets never enter the cost.
2. **The target hardware has two engines.** `analysis.txt` (older log) shows Snapdragon X Elite: NEON CPU plus Adreno OpenCL, i.e. `CPU` and `QUALCOMM_IGPU` (`kernels.hpp:312-340`). If every selected non-view op in a bucket is on the CPU engine, makespan is just the sum of selected costs, in any order.
3. **Branch order defers the only cross-bucket variables.** `chooseBranch` (`brancher.hpp:1029-1129`) fixes all required selections (bucket 0 first, then bucket 1, ...), then cache bits (`chooseCache`, prefer 0), then cached offsets, then schedule. Cache bits and cached shared offsets are what couple buckets.
4. **Proof cost is bound-driven.** Once the incumbent is optimal, the number of nodes needed to prove it depends on bound strength and variable choice, not value ordering. So a strong bound and an early near-optimal incumbent are the priorities for `--optimal`.
5. **Leaf = everything fixed.** The incumbent is only updated at a leaf where all SELECTED, START and OFFSET variables are fixed (`search_engine.hpp:880-921`).

## A. Better bounds (aimed at `--optimal`)

### A1. Conditional additive-work bound

With one dominant engine the objective is a sum, but critical path only captures the longest chain, and p0 #2 only counts classes that are already mandatory. Define, per bucket and per class,

`W(C) = min over active alternatives a of [ cost(a) + Σ over child classes D of share(D) · W(D) ]`

where `share(D) = 1/k` and `k` is the number of distinct parent classes that could still be active (selection domain not fixed to 0).

- Exact when all of D's parents are alternatives of a single class (only one is chosen).
- Admissible under sharing: if D ends up with fewer active parents than `k`, it is under-charged, never over-charged. Classes that are mandatory on every path (e.g. all alternatives of the root reach them) are charged fully.
- `k` shrinks as parent classes become fixed-unselected, so the bound only tightens down the tree. Recompute or restore on backtrack.
- Use the argmin as the value ordering for `preferredENode` and as a first-incumbent heuristic.
- Root bound is `W(root)`, combined by `max` with the other bounds, never summed with them.

### A2. Static inside/outside reduced-cost fixing

A two-pass sweep over the e-graph (inside values bottom-up, outside values top-down, as in hypergraph shortest paths) gives, for every alternative, the cheapest total extraction that uses it. With U from a cheap feasible incumbent (see F), delete alternatives whose value is at least U (use a conservative float comparison). This is O(E) for all alternatives at once instead of probing each one (p0 #3), and can run before search starts, shrinking domains for every propagator. For multi-engine objectives use the A3 cost, so the value stays a valid lower bound.

I found no published reduced-cost-fixing recipe specific to e-graph extraction; the literature found covers ILP/heuristic extraction (TENSAT, SPORES, e-boost) and treewidth DP. This is an adaptation.

### A3. Two-engine fractional work bound

p0's engine-weight extension reduces to one scalar for CPU/IGPU:

`LB = max over α in [0,1] of Σ over mandatory C of min over a of cost(a) · (α·[CPU ∈ E(a)] + (1−α)·[IGPU ∈ E(a)])`

Concave and piecewise-linear in α, so ternary search or breakpoint enumeration is cheap. The plain per-engine minimum gives 0 for any op that can run on either engine; this one does not. Combine with A1 by using the α-weighted cost inside `W(C)`. Match simulator semantics: views, INPUT, CACHE contribute 0; an op with an empty engine list is CPU; a multi-engine op charges every engine it occupies.

### A4. Subset dominance among alternatives

`isFasterEquivalentENodeDominated` requires identical children. Extend: `a` dominates `b` in the same class if `children(a) ⊆ children(b)` (canonical classes), `cost(a) ≤ cost(b)`, and engine set, memory space, shape/strides, view status, in-place options and cache effects are compatible (same context-preservation argument as the existing rule). Cheap preprocessing; it preserves the optimal value but not necessarily every optimal plan.

## B. Less schedule and memory search

### B1. Branch on engine-sequencing pairs, derive ranks afterwards

Only pairs of ops that share an engine and are not already ordered by dependencies need an order decision. Decide those, then derive START ranks as a canonical linear extension of (dependencies ∪ engine chains). This removes the n!-style symmetry among equivalent total orders. Caveat: memory lifetimes and write-after-read depend on rank, so the memory check must run on the derived order, and the derived order must be one the memory model accepts. If the rank-order matters for feasibility beyond engine chains, pick the canonical extension that is most memory-friendly or fall back to rank branching for that bucket.

### B2. Memory as a feasibility witness, solved after selection and order

Offsets are not in the objective, and at sequence length 16 memory is unlikely to bind for a 270M model. Proposal:

- After selections and order are fixed for a bucket, try a constructive offline allocator first (greedy by size and lifetime, respecting view/in-place/cache rules). TelaMalloc combines heuristics with CP-SAT for the same kind of problem.
- Only if it fails, run the existing exact offset branching. This keeps completeness.
- On failure, record a conflict set (e.g. the mandatory buffers live at the peak) rather than a bare failure.
- Motivation: `SEARCH_PERFORMANCE.md` shows write-after-read and offset propagation dominating the EmbeddingGemma runs. That is a different workload, so check with a profile on Gemma first.

### B3. Symmetry breaking across identical layers

Gemma-3-270m has 18 structurally identical layers (same shapes, same costs at sequence length 16). Cache bits and selections are interchangeable across them, so budget-limited cache choices explore permutations of equivalent assignments.

- Merkle-hash each e-class subgraph including candidate costs and shapes to find interchangeable groups.
- Add lex-order constraints on cache bits within a group only when the memory model is also symmetric for them (equal buffer sizes, no distinguishing fixed offsets).
- Gemma 3 mixes local (sliding-window) and global attention layers; the groups may differ at long sequence lengths, so recompute groups per configuration.

### B4. Reorder branching

Decide cache bits and cached offsets before, or interleaved with, required selections, with probing of both polarities for high-fanout cache bits. Today they come after every required selection in every bucket. Verify with the branch-count line already logged ("branches(cached/selected/start/offset)") before changing.

## C. Scaling toward 2×131072 buckets

### C1. Search over the bucket-index axis

Treat sequence position as a parameter and solve interval endpoints.

- If every kernel cost is monotone non-decreasing in position, the optimum at the lower endpoint bounds the interval from below, and the lower endpoint's plan evaluated at the upper endpoint bounds it from above (if feasible there).
- If both endpoints pick the same selection and each selection's cost is affine in position, that selection is optimal in between: the optimum is a lower envelope (min) of affine functions, which is concave, so it cannot lie below the chord of a selection that touches it at both endpoints.
- Otherwise bisect. Exact solves scale with breakpoints × log B instead of B.
- Must be verified against the fitted cost models. Eligibility boundaries, alignment cliffs and memory effects break monotonicity/affinity; p1 gives GEMM tiling as an example. Memory feasibility and engine contention are also not affine.

### C2. AND/OR search over the shared interface

Once cache bits and shared offsets are fixed, buckets are independent. AND/OR branch-and-bound (Marinescu and Dechter) exploits this with context-based caching. Concretely:

- Memoize each bucket's optimum keyed by the cache bits and shared offsets it actually touches.
- A single-bucket relaxation `opt_b(c_k = v)` is a valid lower bound on `opt_b(g)` for any full assignment `g` containing `c_k = v`, and also for the relaxation with the cache bits left free. A table of these over bucket signatures gives a cheap master bound.
- This is a concrete way to build the LBBD master from p0/p1.

### C3. Hash at sub-DAG level

Most of a decode step's graph is identical across positions; only attention and the cache slice differ, and local-window layers become identical once position exceeds the window. Whole-bucket fingerprinting (p0 #11) will hit rarely; Merkle hashing per e-class subgraph (candidate costs included) should hit often. This also serves B3.

### C4. Parallelism

The target has 12 threads and the search is single-threaded. Independent buckets, once the shared interface is fixed, parallelize directly.

## D. Practical shortcuts for `--optimal`

- **Gap tolerance (needs your decision).** Costs come from fitted interpolation models, so differences around 1e-4 relative are noise. Pruning on `lb ≥ U·(1−ε)` would end the proof much earlier, but gives up strict exactness; make it a flag, not the default.
- **CP-SAT as incumbent and bound provider.** The repo already has `--use-ortools-full` and `--use-ortools-lns` (`ortools_export.hpp`). Feed its incumbent in via `setIncumbent`, and stop as soon as incumbent equals a certified lower bound (root bound from A1/A3 or CP-SAT's best bound). Check that the export's semantics (views, cache addresses, memory reuse, engine sets, weights) match, as p2 warns.

## Suggested order

1. Instrument: bound-gap history, time to first incumbent versus proof time, share of nodes in selection / start / offset phases, and the cache-bit count and symmetry-group sizes for Gemma.
2. A1, A3, then preprocessing A2 and A4, with leaf assertions `LB ≤ simulated makespan` for every bucket (p2 §13).
3. B4 and B3.
4. B2.
5. B1.
6. C2, C3, C1 once bucket counts grow.

## Sources

- [TelaMalloc: Efficient On-Chip Memory Allocation for Production ML Accelerators (ASPLOS '23)](https://dl.acm.org/doi/10.1145/3567955.3567961)
- [Best-First AND/OR Search for Graphical Models (Marinescu, Dechter)](https://cdn.aaai.org/AAAI/2007/AAAI07-186.pdf)
- [Memory Intensive Branch-and-Bound Search for Graphical Models](https://cdn.aaai.org/AAAI/2006/AAAI06-188.pdf)
- [e-boost: Boosted E-Graph Extraction with Adaptive Heuristics and Exact Solving](https://arxiv.org/pdf/2508.13020)
- [Equality Saturation for Tensor Graph Superoptimization (TENSAT)](https://arxiv.org/pdf/2101.01332)
- [Fast and Optimal Extraction for Sparse Equality Graphs](https://doi.org/10.1145/3689801)
