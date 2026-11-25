# Reuse feasible search choices across independent buckets

Status: implementation plan, not an implemented change. Prepared 2026-10-15.

## Objective and implementation order

Implementation order: **[families.md](families.md) -> [index_relations.md](index_relations.md) -> [reuse.md](reuse.md)**. Implement this plan third, after family removal and exact index relations. Inspect the resulting checkout rather than restoring interfaces removed by those stages.

Accelerate compilation of ordinary independent buckets by reusing feasible search choices. Reuse can be as simple as the heuristic, brancher, or selector preferring corresponding values in another bucket after finding a valid solution for one bucket. Do not build plan families, interpolation, relocation, representative domains, or a new template compiler.

Each requested bucket retains its exact dependency sets, valid rewrites, and concrete compiled graph. Bucket count remains a user-selected compile/runtime tradeoff: full-only, coarse buckets, or many specialized prefill/decode buckets are all legitimate configurations. Preserve the existing joint optimization of shared caches and memory constraints.

## Starting code and prerequisites

Paths below are relative to `tensor_graphs_cpp/`. The inventory originated at commit `f4093ec0`; the two preceding plans will change some interfaces.

- `core/plan/primal_heuristic.hpp`: feasible plan construction and local improvement.
- `core/plan/brancher.hpp`: variable/value selection and branch ordering.
- `core/plan/search_engine.hpp`: incumbent validation, publication, search deadlines, and heuristic starts.
- `core/plan/search_state.hpp`: bucket-local domains, selection variables, and correspondence metadata.
- `core/plan/planner.hpp`: ordinary bucket saturation, preprocessing, and extraction.
- `core/ops/op_def.hpp` and the index-relation interfaces introduced by the preceding stage: semantic identities and exact recipient arguments.

Before implementation, confirm families and family reweighting are gone, exact index relations are integrated, and eligible partial InfinityDomination works soundly. Use the preceding stages' correctness suites and independent-bucket compile measurements as the baseline. Do not claim an inherited failure is fixed by seeding or silently work around it by enlarging domains.

## Required invariants

1. Preferences affect candidate construction and value ordering only. They do not equate bucket decisions, permanently fix domains, remove alternatives, or alter exact dependency sets.
2. Correspondence is semantic. Bucket-local variable/enode integers are never copied without a verified mapping to a recipient choice.
3. Every transferred choice is checked against current recipient domains and normal propagation, then recosted using recipient shapes, constants, strides, and exact requested sets.
4. Local bucket feasibility does not imply global cache/memory feasibility. Only a normally validated joint candidate can become a global incumbent.
5. Missing, ambiguous, stale, or infeasible preferences fall back to ordinary selection. They do not cause compilation failure.
6. Existing unseeded starts and alternative branches remain available. Search correctness and certified lower bounds must not depend on trusting the donor.
7. Explicit workload weights and the user-requested bucket set remain unchanged.
8. Keep reuse within the current compilation for official inference evaluations. No plans or solver artifacts from previous evaluations; the existing kernel timing calibration exception still applies.

## Implementation sequence

### 1. Add optional cross-bucket value preferences

Primary integration points:

- `core/plan/primal_heuristic.hpp`: feasible candidate construction and local improvement.
- `core/plan/brancher.hpp`: value ordering.
- `core/plan/search_engine.hpp` and `search_state.hpp`: incumbent publication and preference lifetime, if needed.

Use the simplest existing correspondence between buckets that is valid. A selection's integer value currently identifies a bucket-local alternative; copying that integer blindly is incorrect. Map a donor choice by logical/base class and semantic alternative identity, such as kernel implementation and operand roles. Restricted computations can differ in shape, constants, strides, or child structure: resolve the corresponding recipient alternative, not the donor's raw enode number. Missing or ambiguous correspondence means no preference.

When a donor bucket has a feasible assignment, record its mapped choices. For the recipient, try a mapped preferred value first if it remains in the domain and passes normal feasibility checks; otherwise use ordinary selection. Keep the alternative branches available. Never mutate permanent domains to encode the preference.

Prefer kernel/operation selections initially. Do not copy raw start times, memory offsets, cache decisions, or rank variables without their own valid mapping. Let existing recipient propagation/scheduling derive them. If the candidate fails, use ordinary search; do not fail compilation because an optional seed was incompatible.

Keep ordinary heuristic starts alongside seeded starts. Revalidate and recost seeded candidates using the recipient's exact kernel arguments. A better donor candidate can refresh preferences; bound retained donor data and avoid repeated full-search-state copies.

Start with a deterministic donor policy: a previously solved bucket of the same request kind with nearby extents, or simply the first available compatible solution. More elaborate similarity scoring is not a prerequisite. If bucket solving is parallel, publish immutable preference snapshots safely without serializing all compilation to wait for a donor.

### 2. Measure compilation before adding more reuse

Report graph analysis, per-bucket saturation, search preprocessing, first feasible solution, subsequent search, extraction, and serialization time separately. Compare ordinary compilation with and without value preferences on identical bucket sets.

Measure increasing bucket counts, including full-only, a small explicit set, and increasingly specialized prefill/decode sets. Record peak memory, how often preferences map and are accepted, time to incumbent, final modeled cost, and actual inference throughput. Do not promise a speedup from seeding before measuring it.

If another stage dominates, optimize that stage while preserving alternatives. Reusing immutable analyses or kernel pattern construction is acceptable. Reusing unverified proofs, learned conflicts, or partial assignments across different constraints is outside the initial scope.

## Focused validation

- Seeded and unseeded exhaustive small problems have the same feasible set and optimum. Compare certified bounds with the independent planner oracle.
- Exercise different bucket-local IDs, missing/ambiguous alternatives, different shapes and structural constants, nonrectangular exact requests, and donor choices rejected by recipient propagation.
- A globally infeasible cache combination cannot become an incumbent because individual bucket candidates were feasible.
- Donor updates and parallel publication do not create races, invalid lifetimes, or reuse of stale variable identities. Deterministic runs should have reproducible preference ordering where supported.
- Verify seeded candidates are costed with actual recipient arguments, including differing reduction extents and layouts.
- Verify disabled/no-donor/incompatible-donor modes use ordinary search successfully and preserve partial mask/restricted-computation alternatives.
- Check budgets honestly: compare total compilation, not just tree-search time, and record mapping/publication overhead and retained preference memory.

Run planner oracle (including certified-bound validity), search-gap tests, and the applicable native private-layout/write-after-read-cache/incremental-propagators/start-rank-window/relaxation-bound suites after implementation. Use `.venv/Scripts/python.exe` on Windows; `.venvx64/Scripts/python` is for OR-Tools. Follow local AGENTS.md naming, fail-loud, and kernel-variant requirements.

Official performance claims must follow `docs/agents/targets/inference_speed/target.md`: fresh compilation, fixed precision/reference checks, the 600-second compile budget, and repeated-run acceptance gate. Diagnostic runs are not official evaluations. Compare identical bucket sets, source/build conditions, and workload weights. Report large-bucket-set budget failures rather than hiding them with widening or pruning.

## Completion criteria

- Valid donor choices provide optional recipient value preferences through a small, reviewable integration.
- Independent buckets, exact relations, all sound alternatives, shared cache correctness, and certified-bound validity are preserved.
- No family/interpolation machinery or expensive new compilation-sharing framework is introduced.
- Measurements show the benefit and overhead of seeding, including cases where it does not help; retained optimization claims satisfy the existing evaluation contract.
