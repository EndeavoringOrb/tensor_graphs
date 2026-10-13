# What the planner search can borrow from OR-Tools CP-SAT and MIP solvers

Date: 2026-10-12. Context: `docs/agents/improve_search.md` (decode, `--optimal --timeout 600`), current retained implementation O8 (eliminated 2,056,153 of 4,851,513 bits; target 2,425,757). Run records live in `docs/agents/SEARCH_IMPROVEMENTS.md` and `logs/search_improvements/`.

## 1. Experiments with CP-SAT on the real Gemma decode bucket

All runs use the native initial-state export `logs/search_improvements/096_native_initial_state.json` (bucket 0: 7,559 classes, 12,950 alternatives, CPU + Qualcomm IGPU engines), OR-Tools 9.15 from `.venvx64`, 8 workers. Scripts: `logs/search_improvements/cpsat_bucket_relaxation.py`, `logs/search_improvements/cpsat_engine_load_bound.py`. Memory (offsets, reuse) is omitted in every model, so each optimum is a *lower bound* on the native bucket makespan: list-scheduling a native dispatch order yields an engine-feasible schedule with the same makespan, and any engine-feasible schedule sorted by start time is a dispatch order whose simulated makespan is no larger.

| Run | Model | Time | Best solution | Proven bound |
|---|---|---:|---:|---:|
| R165 | Time-indexed: optional interval per alternative (presence literal), precedence, `NoOverlap` per engine, cache budget | 300 s | 187.68 | **0.705** |
| R166 | Min over plans of max engine load (selection + cache budget only) | 53 s | 30.361 (optimal) | **30.361** |
| R167 | Same, critical path of the optimal selection | 120 s | load 30.36, critical path **43.07** | 30.35 |
| R168 | Max(engine loads, precedence critical path), no `NoOverlap` | 300 s | 41.71 (CPU 41.68, GPU 4.16, critical path 37.93) | **30.33** |

Native comparison: root lower bound 0.53 (whole objective, weights 1/16), first incumbent 75.01 per bucket (all 16 weighted buckets are equal, so the objective equals the per-bucket makespan).

Findings:
1. **The native root bound is ~57x weaker than an easy relaxation.** A min-max engine-load bound proves 30.36 per bucket. Our `EngineWorkloadPropagator` charges only mandatory work, and after root propagation only the 17 roots are mandatory (D9), so it contributes 0; `CriticalPathPropagator` ignores optional children. CP-SAT's own time-indexed LP (big-M precedences, optional intervals) is just as weak (0.705 after 300 s): precedence relaxations alone do not help, the *load* aggregation does.
2. **The incumbent is probably far from optimal.** The bucket optimum lies in [30.33, ~45]: R168 found a selection with critical path 37.9 and CPU load 41.7 (no memory). Native first incumbent is 75.0. CP-SAT itself is a poor primal solver on the time-indexed model (187.7), so the gain must come from a better selection heuristic, not from handing the problem to CP-SAT.
3. **CP-SAT presolve finds little on this structure.** Probing fixed 538 + 73 + 3 Booleans, 2–4 dual strengthenings per pass, 418 dominated set-packing constraints, symmetry graph built but no useful orbits reported. Nothing comparable to the 1.94M bits from our layout dominance (H6) or the 74k-bit source prefix (H20c). Our encoding-level dominances are already stronger than generic dual reductions here.

## 2. Techniques and how they map onto this planner

| Technique (source) | What it is | Applicability here |
|---|---|---|
| Optional intervals with presence literals, enforcement literals (CP-SAT model; [Didier, EURO MIP 2025](https://www.mixedinteger.org/EUROMIP/2025/slides/frederic-didier.pdf); [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/advanced_modelling.html)) | Start/size of an absent interval are unconstrained and never branched. | This is exactly our "irrelevant coordinate" problem. CP-SAT never pays search effort for absent intervals; our volume metric counts START/OFFSET of every optional class (~2.6M of the remaining 2.8M bits). Under the current contract we emulate it with canonical values (`UnselectedStartOffset`), which only helps once a class is decided. A presence-aware metric would be a contract change. |
| Linear relaxation of `NoOverlap`/cumulative via presence-weighted load (Perron, [CP-SAT for scheduling](https://schedulingseminar.com/presentations/SchedulingSeminar_LaurentPerron.pdf)) | Sum of present durations on a machine <= makespan. | **Highest-value borrow.** R166 shows a 30.36 bound from this alone. Native options: (a) a combinatorial bound (fractional min-cost cover by Lagrangian/subgradient on the selection constraints), (b) shell out to GLOP at the root like `ortools_export.hpp` already does for `--use-ortools-full`. |
| Exact LP propagation + reduced-cost fixing (Didier slides, "Propagating dual-feasible LP") | Dual multipliers give a valid cut `terms <= rhs`; propagating it raises the objective bound and fixes variables whose reduced cost exceeds the incumbent gap. | Sound and directly bit-eliminating (fixing alternatives to 0 makes classes unselectable, which fixes their START/OFFSET). **Limited here by the weighted objective:** with 16 equal-weight buckets, restricting one bucket must raise that bucket's bound by 16 x (incumbent - total bound). Today 16 x (75.0 - 30.4) ~ 714 cost units; even at incumbent 42 / bound 40 it is 32 units, more than any single alternative changes. So reduced-cost fixing only bites near proven optimality. |
| Conflict-driven clause learning / lazy clause generation (Didier slides: "inspired by Chuffed and Stuckey's *Search is dead, long live proof*", PPDP 2013) | Explain propagations, learn nogoods on conflict, backjump. | Our engine has no learning; conflicts (e.g. memory reuse, cycles) are rediscovered. Would speed proofs in deep subtrees; does not shrink near-root boxes. |
| Large Neighborhood Search, feasibility jump (Didier slides) | Fix most of an incumbent, re-solve a neighborhood. | Repo already has `ortools_lns.py`; a native LNS over selections guided by the load LP would attack the 75 -> ~40 incumbent gap (gate metric "best incumbent cost"). |
| Presolve: probing, coefficient strengthening, dominated columns ([Gamrath et al., MPC 2015](https://link.springer.com/article/10.1007/s12532-015-0083-5); [dual reductions survey](https://tidel.mie.utoronto.ca/pubs/dualcumulative.pdf)) | Fix values whose trial propagation conflicts; remove dominated variables. | R165 shows probing finds ~600 fixings in bucket 0. Native root probing on SELECTED values (propagate each alternative, drop failures) is cheap and sound, but small. Dominated-column logic is what H6/H19/H20c/H25 already do in problem-specific form. |
| Symmetry detection and orbital fixing (CP-SAT symmetry graph) | Break automorphisms. | Buckets 0-15 differ in KV positions, so they are not exact automorphisms. Equality-style symmetry breaking does not change the Cartesian box volume anyway (volume counts each variable independently). |
| Logic-based Benders (Hooker), already proposed in `gemma-search-research-2026-10-11-p0.md` | Master over caches, per-bucket subproblems. | Matches the structure (buckets coupled only by the cache set). Would make per-bucket search tractable; the volume metric still counts every bucket's box. |

## 3. What this means for the 50%-bits target

The remaining 2.795M bits are START (~1.56M), OPENCL offsets (~1.06M), SELECTED (~0.18M). Bits leave the frontier only through root propagation/dominance or by closing near-root siblings (stored bound >= incumbent). The experiments quantify the second route for the first time:
- even a perfect per-bucket bound (~30-41) stays far below the current incumbent (75), and
- because a single-bucket change is diluted by weight 1/16, closing a near-root sibling or reduced-cost-fixing an alternative requires essentially proven optimality of the whole 16-bucket objective.

So borrowing CP-SAT's LP bound and LNS is worthwhile for the incumbent-cost and proof metrics, but it does not by itself unlock the bit target. The large reservoirs are "irrelevant coordinates" of optional classes, which CP-SAT avoids by modeling optional intervals with presence literals; the fixed encoding/metric here counts them.

## 4. Recommended next steps (in order of expected value)

1. **Root engine-load bound** (LP or Lagrangian; validated against R166 = 30.36 on bucket 0). Sound, cheap at the root, raises root LB 0.53 -> ~30. Improves pruning in deep subtrees and gives an honest optimality gap.
2. **LP-guided first dive / LNS** for a better incumbent (target < 45 vs 75). Improves "first-incumbent cost" and "best cost" metrics; prerequisite for any bound-based closure.
3. **Root probing of SELECTED values** (small, sound).
4. **Contract question for the user:** whether the volume metric should exclude coordinates of classes whose presence is undecided (CP-SAT-style optional semantics). Without it, the measured experiments and the analysis in `SEARCH_IMPROVEMENTS.md` find no sound route to the 50% target under the fixed encoding.

## Sources

- [MIP aspects of OR-tools CP-SAT solver, F. Didier, EURO MIP 2025 (slides)](https://www.mixedinteger.org/EUROMIP/2025/slides/frederic-didier.pdf)
- [CP-SAT for scheduling, L. Perron (slides)](https://schedulingseminar.com/presentations/SchedulingSeminar_LaurentPerron.pdf)
- [The CP-SAT Primer: advanced modelling](https://d-krupke.github.io/cpsat-primer/advanced_modelling.html), [parameters](https://d-krupke.github.io/cpsat-primer/parameters.html), [understanding the log](https://d-krupke.github.io/cpsat-primer/understanding_the_log.html)
- [Gamrath, Koch, Martin, Miltenberger, Weninger: Progress in presolving for mixed integer programming, Math. Prog. Comp. 7 (2015)](https://link.springer.com/article/10.1007/s12532-015-0083-5)
- [Using dual presolving reductions to reformulate cumulative constraints](https://tidel.mie.utoronto.ca/pubs/dualcumulative.pdf)
- [OR-Tools disjunctive propagators](https://github.com/google/or-tools/blob/stable/ortools/sat/disjunctive.h) (cited in the earlier p0 review)

## 5. Follow-up: the OPENCL capacity blocker was wrong (co-activity bound)

The analysis above assumed OPENCL layouts must reserve room for every possible cache set. A further CP-SAT experiment (R169, `logs/search_improvements/cpsat_max_coactive.py`, reachability included) shows that a single bucket plan can hold at most **467,559 co-active OPENCL pages** (430,590 cacheable), below the 524,288-page capacity, although all OPENCL classes total 1,631,336 pages. Most cacheable classes are alternative weight layouts consumed by different alternatives of one kernel class, so they are mutually exclusive.

Borrowed idea: CP-SAT-style *implied literals* from reachability. Each class gets necessary selection conditions (`x active => class P selects from S`), and disjoint conditions on a common class certify exclusivity. An offline prototype (R170) places every OPENCL class first-fit, sharing pages only between certified-exclusive classes, in 467,784 pages. This is within 0.05% of the co-activity optimum and fits the capacity. This became H27, an exclusivity-shared static layout in `StaticPrivateLayoutPropagator`, with oracle cases 27/28 (see `SEARCH_IMPROVEMENTS.md` for runs).
