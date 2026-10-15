# Exact index relations for dependency analysis and restricted computation

Status: implementation plan, not an implemented change. Prepared 2026-10-15.

## Objective and user decisions

Replace bounding-box-based dependency analysis with exact index-set and index-relation representations. No silent conservative widening is permitted. Preserve triangles, bands, strided sets, unions, and data-dependent indexing rather than replacing them with rectangles or full tensors.

Keep domain/shape inference separate from dependency mapping, but share each operation's indexing semantics. A full tensor domain is a normal index set, not a special parallel region system.

Construct fused-operation relations automatically from their registered reference graphs by composing primitive relations. Do not require handwritten fused locality allowlists, operation-name special cases, or separate approximate fused propagation.

Implementation order: **[families.md](families.md) -> [index_relations.md](index_relations.md) -> [reuse.md](reuse.md)**. Implement this plan second, after family removal establishes correct ordinary bucket compilation. This plan does not restore families or interpolation: every requested bucket remains concrete and independently optimizable. Cross-bucket search value preferences belong to the third stage, reuse.md; do not introduce them during this migration.

## Current implementation to inspect

- `tensor_graphs_cpp/core/ops/op_def.hpp`: `OpTraits::inferShape`, `forwardRegion`, and `backwardRegion`.
- `core/ops/common.hpp` and primitive files under `core/ops/`: operation shape and region semantics.
- `core/shape_propagator.hpp`: shape inference and region dispatch.
- `core/plan/bucket_regions.hpp`: dirty/needed/live analysis, mask/support facts, bounding intersections, and cached-value validity analysis.
- `core/rewrite.hpp`: structural/dot/elementwise slice pushdown, `fusedRowLocality`, `SlicePushDownFused`, scatter ownership, and InfinityDomination.
- `core/ops/fused.hpp`: currently rejects direct forward/backward region propagation.
- `core/reference_graph_registry.hpp`, `core/kernel_pattern_cache.hpp`, and kernel reference factories: sources for constructing fused semantics.
- `core/types.hpp`, `core/shapes.hpp`, and region helpers in `core/misc.hpp`: current region representation and coordinate transformations.

Inspect current code before editing. The paths above describe commit `f4093ec0`. Do not edit broad kernel identity inputs such as `core/types.hpp` or `core/kernels.hpp` casually: the inference record warns that hashing their bytes affects kernel IDs/calibration. Prefer new analysis headers and deliberately manage any necessary identity migration.

## Define exactness before implementation

The required result is an exact symbolic relation under the declared operation semantics and available proven facts. It must neither add unrelated indices nor omit dependencies. This is not a requirement to solve arbitrary program equivalence or infer every algebraic cancellation.

Distinguish these objects:

1. **Tensor domain:** all legal logical coordinates and their extent constraints.
2. **Dependency relation:** which operand coordinates each output coordinate depends on, including reduction indices and control/value dependencies.
3. **Value facts:** proven zeros, constants, bounds, or mask properties that can legally simplify dependence/computation.
4. **Physical access description:** layout, strides, and any extra reads/writes a kernel actually performs. A kernel may physically touch values outside its semantic dependency set.
5. **Restricted implementation:** a concrete computation producing a requested set with the original numerical semantics.

For multiplication, indexing relates an output to both operands; eliminating one dependence because the other is proven zero requires an explicitly justified value rule and appropriate floating-point preconditions. Composition describes graph-level dependence; it need not discover that `x - x` is constant, particularly under floating-point NaN/infinity behavior.

Use exact symbolic, unresolved, and unsupported outcomes explicitly. Never return a larger set as an 'exact' answer. A query whose emptiness cannot yet be decided remains undecided; an optimization requiring proof must not proceed. Unknown is not false or empty. Unsupported operation semantics must fail loudly with operation and query context instead of silently returning the full domain.

## Core relation model

Let a primitive operand relation be:

```text
D_operand(output_index, input_index; parameters, runtime_values)
```

For a requested output set `R`, required operand coordinates are:

```text
required_input = { i | exists o: o in R and D_operand(o, i) }
```

For a changed operand set `C`, potentially affected output coordinates under the same operation semantics are:

```text
affected_output = { o | exists i: i in C and D_operand(o, i) }
```

Union contributions from multiple operands; retain operand identity. Provide image/preimage, composition, union, intersection, difference, substitution, membership, and exact equality/subset/disjointness queries where supported. Use three-way proof results for queries that are not decidable by the initial implementation. Difference is needed for cache validity and concat/update partitions, not just positive dependencies.

Start with an immutable, typed expression/predicate DAG supporting:

- Named coordinate and reduction variables, integer constants, and shape/control parameters.
- Integer affine arithmetic, comparisons, conjunction/disjunction, and existential quantification.
- Integer division/remainder for flatten/unflatten and blocked indexing, with validated divisors and sign conventions.
- Piecewise expressions and finite unions.
- Array reads/runtime index expressions, identified by tensor/value version, where static affine relations are insufficient.

Keep affine fragments distinguishable so exact simplification can be efficient. Do not expand a causal triangle into one box per element/row merely to obtain an existing container type. Use hash-consing/memoization and lazy composition to avoid expression explosion. Sharing immutable expressions is acceptable; widening under a resource limit is not. Under a limit, retain the unsimplified exact expression, decline an optimization requiring proof, or report an explicit compilation resource failure.

Before choosing an external symbolic/polyhedral dependency, evaluate its Windows ARM64 support, integration cost, exact query coverage, and licensing. A library is optional; do not make adoption a prerequisite without evidence. The representation must support non-affine runtime index reads even if an affine solver handles only a subset of proofs.

## Primitive operation semantics

Document relation definitions beside each operation and expose them through OpTraits or a corresponding typed registry. Keep domain validation as a distinct entry point.

| Operation | Exact mapping requirement |
|---|---|
| Unary/elementwise | Same logical coordinate; explicit broadcasting maps expanded dimensions to zero |
| Permute | Coordinate permutation and inverse |
| Slice | Normalized start/step mapping, including negative steps when supported |
| Reshape | Flatten/unflatten equality using logical extents; distinguish logical indexing from physical contiguity |
| Repeat/broadcast | Correct modulo or singleton mapping for the actual operation semantics |
| Concat | Piecewise input selection and coordinate offsets |
| Sum/max/argmax | Output coordinates plus the complete legal reduction domain; include axes/control dependencies |
| Dot | Batch/output coordinates and a shared contraction index, with rank/domain compatibility checks |
| Fill/arange | Scalar/control dependencies and generated-coordinate expressions |
| Gather | Runtime index read plus data access at that index; preserve negative-index and bounds semantics |
| Scatter/update | Updated-coordinate mapping and explicit dependence on the base value outside the update |
| Copy/cast/contiguous | Logical identity with separate dtype/layout/device constraints |

Example, matrix multiplication:

```text
C[i,j] = sum_k A[i,k] * B[k,j]
D_A((i,j), (a,b)) := a = i and 0 <= b < K
D_B((i,j), (a,b)) := b = j and 0 <= a < K
```

Retain the shared reduction variable in the computation description; operand dependency sets alone do not encode how to reconstruct the sum.

Example, gather:

```text
Y[i,d] = table[indices[i], d]
```

Represent the dependence on `indices[i]` and on `table[indices[i], d]`. Unknown index values do not justify declaring every table row the exact required set. It is valid to execute a generic full-table-backed gather while retaining the exact symbolic access relation.

For forward dirty propagation through changing runtime indices, include old/new value versions where needed: a changed selector can alter both the selected address and result. Do not specialize a reusable plan on observed runtime indices unless compilation/runtime guards explicitly establish validity. Initial implementations can retain unresolved value expressions without exploiting them.

## Fused operations from reference graphs

Build a relation summary by traversing the registered reference graph from its output to formal inputs, composing primitive relations and unioning paths. Traverse the same summary in reverse for forward affected-set queries; do not implement independent approximations for the two directions.

Instantiate reference graphs with actual input shapes, dtypes, and structural constants, or preserve those values as explicit symbolic parameters. Registry dummy shapes and fabricated axis/slice constants are examples for registration, not valid semantics for every call.

Cache immutable summaries by reference definition and all semantic parameters on which the instantiated relation depends. Kernel names alone are insufficient when the same factory covers varying axes, repeats, or slice controls. Detect reference cycles and missing primitive semantics explicitly.

Validate output domain and formal input mapping against the actual fused operation. Compare composed fused relations with unfused graph relations. Do not infer the fused output shape solely from dependency projections: constants and reductions can have output extents not recoverable that way.

Relation composition makes fused dependency mapping general; it does not prove that a particular fused kernel accepts every restriction. Retain kernel applicability/layout/reduction checks. If a fused implementation cannot handle an exact requested set, consider an unfused restricted computation as another alternative. Do not discard the exact set to fit the fused kernel.

## Masks and InfinityDomination

For a proven causal mask, preserve predicates such as `key <= query`; for a sliding mask include the window predicate. Restricting the query domain then restricts the needed keys naturally, without a family hull or bounding box.

However, an additive finite sentinel is not a structural absence of dependency. `x + (-1e9)` still depends on `x`. Neither a negative threshold nor the name 'mask' proves it may be dropped. Negative infinity also needs non-finite-input semantics considered.

Audit current `kMaskedValue`, `finite`, and `support` rules. Establish the numerical preconditions under which a masked branch can be replaced or eliminated, including max subtraction, exponentiation underflow, all-masked rows, NaNs/infinities, and reduction identities. Record proven facts explicitly and apply them at valid consumers. Do not change the graph's mask semantics or precision to obtain a simpler relation.

InfinityDomination must remain an available eligible optimization, including partial buckets, with explicit background/base dependencies and correct initialization. Resolve the existing partial-update issue rather than retaining blanket disablement. A partially cached masked score tensor and an exactly computed tensor are not interchangeable merely because the eventual softmax usually agrees.

## Restricted computation and kernel lowering

Dependency mapping establishes what values are needed; it does not itself build an equivalent smaller operation. Add operation-level restricted construction using the same indexing expressions, preserving coordinate transforms, reduction domains, output placement, and numerical order requirements from the existing precision contract.

Keep standalone restricted values and updates to cached values as separate alternatives. Model an update's base explicitly; do not use owner strings as the only representation of its untouched values. Cache validity should track exact sets and input/value versions across bucket switches, resets, changed prefixes, and padding changes.

Lower exact sets to existing kernel interfaces when possible. A kernel may compute a larger dense domain as an explicit costed implementation alternative, with valid inputs for every physical read and correctly initialized masked/background cells. This is execution overcomputation, not analysis widening. Preserve the exact semantic request and competing exact implementations. Never feed uninitialized padding because it lies outside the semantic request but inside the selected kernel's physical read footprint.

Adapters to old region-list interfaces must be exact: rectangular sets convert directly; nonrectangular sets require an exact decomposition or remain symbolic. Never silently take a bounding box. Preserve full-bucket execution as a user-selected specialization tradeoff and as a legal concrete candidate, without mislabeling its domain as the exact dependency set.

## Staged implementation

1. Specify the expression grammar, proof-result types, exactness contract, and primitive relation API. Add small finite-domain relation tests before integrating the planner.
2. Implement primitive relations for the full inference operation set and explicit unsupported outcomes for remaining operations. Preserve existing domain/shape validation.
3. Compose fused-reference summaries and test them against primitive graphs, including structural constants and multiple paths to an input.
4. Integrate exact dirty/needed propagation and cache-validity queries. During transition compare old and new results for diagnostics, but do not use old conservative results as new exact answers.
5. Integrate restricted construction and mask/value facts, including sound partial InfinityDomination. Replace manual fused row-locality restrictions where general relations provide the proof.
6. Remove bounding-box fallbacks, duplicated relation logic, and obsolete APIs after all callers migrate. Update serialization/cache invalidation if compiled semantics or keys change.
7. Measure analysis cost, expression size, saturation/search time, compile memory, selected work, and runtime across bucket counts and context lengths. Optimize exact representation/query reuse rather than widening.

## Validation and acceptance

Use finite enumeration as an independent oracle for index sets/relations. For small shapes and valid control values, enumerate primitive access semantics and compare forward/backward images exactly. Perturbing numeric inputs alone is not a sufficient dependency oracle because cancellation can hide accesses.

Required cases include empty sets, scalars, noncontiguous layouts, negative slice bounds/steps, reshape across axes, repeated/broadcast axes, concat boundaries, rank-2/3/4 dot, reductions, changing gather indices, overlapping updates where supported, and explicit unsupported controls. Test relation algebra and undecided proof outcomes.

For fused summaries, compare primitive-graph and summary relations on RMS norm, softmax, attention fragments, view-heavy graphs, and shared-input paths. Include differing axes and structural constants with identical tensor shapes to catch incomplete cache keys.

For masks, test early causal contexts, triangle interiors, sliding-window boundaries, all-masked rows, large values that defeat threshold assumptions, and non-finite inputs according to declared semantics. Verify partial-buffer initialization, cross-bucket cache transitions, prefix edits, and reset.

Run planner oracle (including bound validity), search-gap tests, applicable native planner/cache/layout suites, and affected kernel/rewrite tests. Follow AGENTS.md naming, fail-loud, Python environment, and separate-kernel-variant requirements.

Official inference evaluation remains governed by `docs/agents/improve_inference_speed.md`; do not alter its reference files, precision, timing contract, or acceptance gate. First demonstrate correctness on small independent cases, then evaluate fresh compilation and actual throughput. Count compile time honestly; no offline precomputed relations/plans that violate the evaluation contract.

Completion requires exact primitive and fused mapping, no silent domain widening, sound mask/state handling, retained independent bucket optimization, and recorded compile/runtime results. Explicitly document unresolved symbolic queries and unsupported operations; do not claim universal exact numerical dependence or universal query decidability.
