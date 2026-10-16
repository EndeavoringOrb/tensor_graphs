# Tensor Graphs Compiler: Large-Scale Multi-Bucket Optimization Implementation Plan

## Objective & Non-Negotiable Invariants

This plan scales `tensor_graphs_cpp` compilation to large sequence lengths (e.g., $N = 262,144$, where $2N+1 = 524,289$ buckets are compiled for full + prefill $[0, t) \to [t-1, t)$ + decode $[t-1, t) \to [t-1, t)$) and arbitrary $N$-D dirty regions (e.g., 2D spatial patches in diffusion models).

**Strict Optimality Invariant:** Every optimization in this plan is **100% exact and optimality-preserving**. No shape canonicalization in cost evaluation, no approximate cost models, and no heuristic pruning that can discard a globally optimal execution plan are permitted. Every bucket must be evaluated and validated against its exact concrete shapes, strides, constants, kernel match predicates, and `MakespanSimulator` cost.

The plan is divided into **6 self-contained, sequential sections**. Implement, test, and commit each section before starting the next.

---

## Agent & Developer Execution Playbook (Critical Lessons from Section 1)

Before implementing Sections 2–6, review these project-specific workflow rules and API conventions:

### 1. Build System & Target Scoping
- **Build Command:** Always build via Python:
  ```powershell
  .venv/Scripts/python.exe build.py --targets test
  ```
  (Do not invoke `cmake` or raw compilers directly; `build.py` configures MSVC developer environment, LLVM/clang-cl flags, and object caching).
- **Targeted Searches:** Always scope `rg` to `tensor_graphs_cpp/` or `tests/` (e.g., `rg -n "pattern" tensor_graphs_cpp/`). Never run unconstrained workspace searches, which will scan `.venv`, build caches, and test outputs and stall execution.

### 2. Test Execution & Calibration Caveats
- **Avoid Unfiltered `test.exe`:** Running `.\tensor_graphs_cpp\test.exe` without arguments executes full end-to-end sessions (e.g. `testConstantsThroughViewOps`) that require concrete benchmark entries in `benchmarks/records.bin`. On machines where newly added kernels lack benchmark records, planner cost is `INF` and initial search propagation fails with a contradiction.
- **Section-by-Section Verification Pattern:**
  - Use targeted CLI flags for fast, deterministic unit verification:
    - Pre-existing flags: `--planner-structure`, `--fusion-regression`, `--bucket-independence`, `--relaxation-bound`, `--start-bounds`, etc.
  - **Recommended Practice:** For each section, add a dedicated unit test file in `tensor_graphs_cpp/tests/` and wire a flag (e.g. `--scatter-canonical`, `--triu-mask`) into `tensor_graphs_cpp/test.cpp`.

### 3. Key In-Memory APIs & Data Structures
- **`EGraph` (`core/egraph.hpp`):**
  - `addEClass(shape, strides, dtype, mem_space, [logical_id], [is_clean]) -> EClassId`: Allocates a new e-class.
  - `addIntConst(const std::vector<int32_t> &vals) -> EClassId`: Allocates an `INT32` e-class, creates the staging buffer, and computes content hash automatically.
  - `addENode(EClassId e_class_id, ENode node) -> EClassId`: Adds an e-node to a class.
  - `findConst(EClassId id) -> EClassId`: Returns canonical leader for class equivalence.
- **`ENode` (`core/egraph.hpp`):**
  - Constructor requires 9+ arguments:
    ```cpp
    ENode(KernelId kernelId, OpType opType, std::string opName,
          std::vector<EClassId> children,
          std::vector<uint32_t> shape, std::vector<uint64_t> strides, DType dtype,
          MemSpace mem_space, std::vector<Engine> engines,
          std::string contentHash = "", uint64_t sig = 0, std::string debugOrigin = "")
    ```
  - Standard host defaults: `MemSpace{1, HandleType::CPP}` and `{Engine{0, EngineType::CPU}}`.
- **Header Hierarchy & Circularity Guard:**
  - Order: `core/types.hpp` $\to$ `core/graph.hpp` $\to$ `core/kernels.hpp` $\to$ `core/cost_model.hpp` $\to$ `core/rewrite.hpp` $\to$ `core/plan/planner.hpp`.
  - `cost_model.hpp` is strictly upstream of `rewrite.hpp` and `planner.hpp`. Utilities in `cost_model.hpp` can be included in `rewrite.hpp` without circular dependency concerns.

---

## Section 1: Canonicalizing `SCATTER` Keys Across Cost, Benchmark, and Fusion Memos [COMPLETED]

**Status:** Completed and verified via `tensor_graphs_cpp/tests/scatter_canonical_keys.hpp` (`test.exe --scatter-canonical`).

### Background & Problem

`OpType::SCATTER` takes 5 inputs: `(updates, starts, ends, steps, output_shape)` (`tensor_graphs_cpp/core/ops/scatter.hpp`).

* `ScatterOp::computeWorkload` (`ops/scatter.hpp`) and `PreparedKernel::prepare` (`common/bench_utils.hpp`, line 366) depend **only** on `inShapes[0]` (the `updates` shape) and `outShape` (the target buffer shape). In fact, `PreparedKernel::prepare` explicitly ignores `inputConstants[1..4]` (`synthesize_scatter_control = kernel.opType == OpType::SCATTER && idx >= 1 && idx <= 4`) and synthesizes identity slice controls so benchmarks never write out of bounds.


* However, `CostModel::exactRecordKey`, `CostModel::log_call`, `KernelBench::discoverCalls`, `Planner::memoizedCost`, and `FusionRule::apply` hash the raw byte contents of `inConstants[1..4]` (`starts`, `ends`, `steps`, `shape`).


* Because every bucket $t \in [1, N]$ has different `starts = {0, t-1, 0}` or `ends = {0, t, 0}`, `SCATTER` misses all cost and fusion memos in every bucket and logs $2N$ duplicate benchmark records that execute the **exact same benchmark** in `KernelBench`.



### Files to Modify

1. `tensor_graphs_cpp/core/cost_model.hpp`

2. `tensor_graphs_cpp/core/common/kernel_bench.hpp`

3. `tensor_graphs_cpp/core/plan/planner.hpp`

4. `tensor_graphs_cpp/core/rewrite.hpp`


### Step-by-Step Implementation

1. **Add a helper `isScatterControlInput(KernelId kernelId, size_t inputIdx)` in `core/cost_model.hpp`:**
```cpp
inline bool isScatterControlInput(OpType opType, size_t inputIdx)
{
    return opType == OpType::SCATTER && inputIdx >= 1 && inputIdx <= 4;
}

inline bool isScatterKernelControlInput(KernelId kernelId, size_t inputIdx)
{
    if (inputIdx < 1 || inputIdx > 4)
        return false;
    if (!KernelRegistry::get().hasKernel(kernelId))
        return false;
    return KernelRegistry::get().getKernel(kernelId).opType == OpType::SCATTER;
}

```


2. Update `CostModel::exactRecordKey` (`core/cost_model.hpp`, line 443):


* Inside the `for (size_t i : order)` loop, check `if (i < inConstants.size() && !isScatterKernelControlInput(kernelId, i))` before calling `put(inConstants[i].data(), inConstants[i].size())`. Otherwise call `put(nullptr, 0)`.


* Note: `outShape` and `inShapes[0]` are already hashed into `exactRecordKey` right above this loop, so the update shape and output shape are preserved.




3. Update `CostModel::log_call` (`core/cost_model.hpp`, line 513):


* After `const auto &entry = KernelRegistry::get().getKernel(kernelId);`, if `entry.opType == OpType::SCATTER`, clear `r.inputConstants[i]` for `1 <= i < r.inputConstants.size()` so `calls.bin` and `callObserver` deduplicate `SCATTER` records with identical `(inShapes[0], outShape)`.




4. Update `KernelBench::discoverCalls` (`core/common/kernel_bench.hpp`, line 108):


* In `costs.referenceObserver`, if `KernelRegistry::get().getKernel(record.kernelId).opType == OpType::SCATTER`, clear `record.inputConstants[i]` for `1 <= i < record.inputConstants.size()` before serializing `record` into `calls`.




5. Update `Planner::memoizedCost` (`core/plan/planner.hpp`, line 872):


* In the loop `for (size_t i = 0; i < in_shapes.size(); ++i)`, if `isScatterKernelControlInput(kernel_id, i)`, pass an empty `std::vector<uint8_t>{}` to `putVec` instead of `in_constants[i]`.




6. Update `FusionRule::apply` (`core/rewrite.hpp`, line 592):


* When constructing `signature` for `selectionMemo()`, in the `for (size_t i = 0; i < inputs.size(); ++i)` loop, if `isScatterControlInput(old.getOpType(), i)`, call `appendSignature(signature, nullptr, 0)` instead of hashing `staged->second->data()`.





### Verification

* Run the existing test suite and a multi-bucket compile with `log_cost_calls = true`.


* Verify that:
1. All compiled plans and makespans are identical to baseline.
2. `benchmarks/calls.bin` / `discoverCalls` emits only **one** `SCATTER` record per unique `(update_shape, output_shape)` pair across all decode buckets instead of $N$ records.





---

## Section 2: Analytical `TRIU` + `NEG_INF` Mask Predicate & Skipping Clean Mask Materialization [COMPLETED]

**Status:** Completed and verified via `tensor_graphs_cpp/tests/triu_mask_predicate.hpp` (`test.exe --triu-mask`).

### Background & Problem

Currently, causal attention masks built via `fill(TGConstants::NEG_INF, shape) -> triu(k=1)` (and optional `repeat`/`reshape`/`permute`/`slice` views) trigger three $O(N^2)$ bottlenecks:

1. `Session::foldCleanTensors()` (`core/session.hpp`, line 147) calls `executeReferenceGraph` with `only_clean_nodes = true` (`core/common/execute_ref_graph.hpp`, lines 87–151). Because `FILL` and `TRIU` have constant inputs, `foldCleanTensors()` materializes the $N \times N$ float tensor on the CPU and writes it to `.reftensors` (275 GB at $N = 262,144$).

2. `bucket_regions::unmaskedRegions` (`core/plan/bucket_regions.hpp`, line 192) scans all $N \times N$ floats and builds a `std::vector<Region>` staircase with $N$ individual row boxes. `computeRegionFacts` (line 338) then intersects against this $N$-element vector $2N$ times ($O(N^2)$ per attention layer).

3. `InfinityDomination::apply` (`core/rewrite.hpp`, line 1558) also reads the materialized $N \times N$ float array from `constantStaging` or `repo` and scans all $N \times N$ elements.

### Files to Modify

1. `tensor_graphs_cpp/core/common/execute_ref_graph.hpp`
2. `tensor_graphs_cpp/core/plan/bucket_regions.hpp`
3. `tensor_graphs_cpp/core/rewrite.hpp`

### Step-by-Step Implementation

#### 2.1. Do Not Materialize Large Constant `FILL` / `TRIU` Tensors in `executeReferenceGraph`

In `tensor_graphs_cpp/core/common/execute_ref_graph.hpp` (lines 87–151):

1. Inside `executeReferenceGraph`, when `options.only_clean_nodes` is `true`, add a helper `isSymbolicCleanMaskOrFill(LogicalId id, const Graph &graph)` that returns `true` if:
* `countElements(graph.getNode(id).getShape()) > 4096`, AND
* `id` is `OpType::FILL`, `OpType::TRIU`, or a view op (`REPEAT`, `RESHAPE`, `PERMUTE`, `SLICE`, `CONTIGUOUS`) whose data child (`child_ids[0]`) recursively satisfies `isSymbolicCleanMaskOrFill` or is `OpType::FILL` / `OpType::TRIU`.

2. Mark `is_clean[node_id] = false` for any node where `isSymbolicCleanMaskOrFill(node_id, graph)` is `true` (and consequently any downstream node that consumes it) during the `options.only_clean_nodes` pass so `foldCleanTensors()` never allocates or writes multi-gigabyte `FILL`/`TRIU` tensors to `repo`.

#### 2.2. Analytical `TriuMaskPredicate` in `core/plan/bucket_regions.hpp`

In `tensor_graphs_cpp/core/plan/bucket_regions.hpp`:

1. Define a compact struct `TriuBound` (placed in `bucket_regions.hpp` so both `bucket_regions.hpp` and `rewrite.hpp` can access it):
```cpp
// Represents the unmasked coordinate set:
//   coord[col_axis] < coord[row_axis] + diag_k   (when upper_masked == true, i.e., TRIU(NEG_INF, diag_k))
struct TriuBound
{
    uint32_t row_axis = 0;
    uint32_t col_axis = 0;
    int32_t diag_k = 1;
};
```

2. In `computeRegionFacts(const Graph &graph, const std::vector<LogicalId> &topo, TGStore *repo)` (`bucket_regions.hpp`, line 280+):
* Add a recursive pattern matcher `std::optional<TriuBound> matchTriuMask(LogicalId id, const Graph &graph, TGStore *repo)`:
  * Base case: `node.opType == OpType::TRIU` with `node.child_ids.size() == 2` and `node.getShape().size() >= 2`.
  * Check that `child_ids[1]` has a staged `INT32` constant `k` (`diag_k = k`).
  * Check that `child_ids[0]` is `OpType::FILL` (or a view of `FILL`) whose scalar fill value is `TGConstants::NEG_INF` (`BucketRegions::kMaskedValue`).
  * Return `TriuBound{static_cast<uint32_t>(shape.size() - 2), static_cast<uint32_t>(shape.size() - 1), diag_k}` (since `TRIU` operates on the last two dimensions of the tensor).
* Inductive steps through views:
  * `OpType::CONTIGUOUS`: preserve `TriuBound` unchanged.
  * `OpType::REPEAT`: if `repeats == 1` on `row_axis` and `col_axis` (broadcasting batch/head dims), preserve `TriuBound` unchanged.
  * `OpType::PERMUTE`: map `row_axis` and `col_axis` through the inverse permutation `dims`.
  * `OpType::RESHAPE`: if the last two dimensions `(H, W)` of the input and output shapes are identical (e.g., `[1, 1, N, N] -> [1, N, N]` or `[B*H, N, N]`), adjust `row_axis = outRank - 2`, `col_axis = outRank - 1`.
  * `OpType::SLICE`: if unit steps (`step == 1`) on `row_axis` and `col_axis`, shift `diag_k' = diag_k + start[row_axis] - start[col_axis]`.

* When `matchTriuMask` succeeds for a mask operand `node.child_ids[m]` of an `OpType::ADD` node `id`:
  * Store `masks[id] = {m, makeFull(shape)}` AND `facts.triuMasks[id] = bound` (instead of calling `unmaskedRegions` to build $N$ boxes!).
  * Propagate `facts.triuFinite[id]` and `facts.triuSupport[id]` alongside `finite[id]` and `support[id]` in the forward topological pass (`ADD`, `POWER`, `MUL`, `DIVIDE`, `REPEAT`, `RESHAPE`, `PERMUTE`, `CONTIGUOUS`, `SLICE`).

3. Provide $O(1)$ closed-form helpers when a node has a `TriuBound`:
* **Intersection of a 1-box `Region box` with `TriuBound b` (bounding box):**
  * Let $[r_0, r_1) = \text{box.region}[b.\text{row\_axis}]$ and $[c_0, c_1) = \text{box.region}[b.\text{col\_axis}]$.
  * If $r_0 \ge r_1$ or $c_0 \ge c_1$, the intersection is empty.
  * The maximum valid column in rows $[r_0, r_1)$ is $c_{\text{limit}} = \max<int64\_t>(0, static\_cast<int64\_t>(r_1 - 1) + b.\text{diag\_k})$.
  * The restricted column interval is $[c_0, \min<uint32\_t>(c_1, static\_cast<uint32\_t>(c_{\text{limit}})))$. If $c_0 \ge \text{stop}$, the intersection is empty; otherwise return `box` with `box.region[b.col_axis].stop` updated to that minimum.
* `axisRange` with `TriuBound b` (`bucket_regions.hpp`, line 213+):
  * When `axis == b.col_axis`:
    * Compute $c_{\text{stop}} = \min<int64\_t>(c_1, static\_cast<int64\_t>(r_1 - 1) + b.\text{diag\_k})$.
    * If $c_0 \ge c_{\text{stop}}$, return `std::nullopt`.
    * If `everyRow` is `true`, every row $r \in [r_0, r_1)$ must have at least one valid column in $[c_0, c_1)$, which holds iff the most restrictive row $r = r_0$ satisfies $c_0 < \min<int64\_t>(c_1, static\_cast<int64\_t>(r_0) + b.\text{diag\_k})$. If not, return `std::nullopt`.
    * Return `Dim{c_0, static_cast<uint32_t>(c_{\text{stop}})}`.
* `dotForwardFromB` and `dotNeededB` (`bucket_regions.hpp`, lines 450–480):
  * When `suppA` has a `TriuBound` (with `row_axis = 1`, `col_axis = 2`), evaluate the intersection of each box `b` (or `o`) with the `TriuBound` in $O(1)$ arithmetic instead of looping over $N$ staircase boxes.

#### 2.3. Analytical `TRIU` Matching in `InfinityDomination` (`core/rewrite.hpp`)

In `tensor_graphs_cpp/core/rewrite.hpp` (`InfinityDomination`, line 1480):

1. Add a helper `bool isTriuMaskEClass(EClassId cid, const EGraph &egraph, TriuBound &out_bound)` that walks the e-graph from `cid` through view e-nodes (`REPEAT`, `RESHAPE`, `PERMUTE`, `SLICE`, `CONTIGUOUS`, `COPY_TO`) to an `OpType::TRIU` e-node whose input 0 is `OpType::FILL` with scalar constant `TGConstants::NEG_INF` and input 1 is scalar `INT32` constant `diag_k`.

2. Update `InfinityDomination::match` (line 1530) to match `ADD` nodes where either child satisfies `isConstantFloat(child, ctx)` **or** `isTriuMaskEClass(child, ctx.egraph, dummy_bound)`.

3. In `InfinityDomination::apply` (line 1558):
* If the mask child `constClass` is matched via `isTriuMaskEClass(constClass, egraph, triu_bound)`:
  * In partial buckets where `ctx.regions && outClass.logical_id != LogicalId{} && outClass.base_eclass_id != BaseEClassId{}` (line 1630), `InfinityDomination` **already** uses `const Region box = bucket_regions::boundingBox(live)` and sets `nonInfRegions = {box}`, `cached_update = true`.
  * In the full bucket (when `cached_update == false`), compute the $k \le \text{kMaxRegions}$ (8) covering boxes analytically from `triu_bound` in $O(\text{kMaxRegions})$ arithmetic (dividing the row range $[0, R)$ into `kMaxRegions` equal row stripes $[r_{\text{first}}, r_{\text{last}})$, each with column range $[0, \min(C, r_{\text{last}} - 1 + \text{diag\_k}))$)—producing the **exact same `nonInfRegions`** as `coverWithFewRegions(mergeRegions(runs), kMaxRegions)` in $O(1)$ time without reading any constant buffer.

### Verification Strategy
1. Create a dedicated test file `tensor_graphs_cpp/tests/triu_mask_predicate.hpp` with unit tests for `matchTriuMask`, `TriuBound` intersection arithmetic, and `isTriuMaskEClass`.
2. Add a `--triu-mask` CLI flag to `test.cpp` and invoke `runTriuMaskTests()`.
3. Verify that on an attention causal mask at $N = 16, 64$:
   - `computeRegionFacts` and `computeBucketRegions` yield the exact same boxes with and without the analytical bound.
   - For $N = 262,144$, `foldCleanTensors()` executes in milliseconds without allocating memory.

---

### Key Implementation Guidelines & Gotchas (Learned from Sections 1 & 2)

Before implementing Sections 3–6, review these critical engine semantics discovered during Sections 1 & 2:

1. **`Graph` and View Nodes Have Lazy Shapes**:
   - `Graph::fill`, `Graph::triu`, and view operators allocate `TensorNode`s with empty shapes (`{}`). Shapes are NOT eagerly computed on creation or modification.
   - Whenever inspecting shapes, ranks, or element counts prior to or during rewrite/template steps, always call `ShapePropagator::inferShapeRecursive(nodeId, graph)` if `node.getShape().empty()`.
   - When modifying e-class or node shapes in an `EGraph`, remember to recompute strides via `calcContiguousStrides(shape)` for standard nodes or via the appropriate view op logic.

2. **Dual Constant Storage Hierarchy (`constantStaging` vs `repo`)**:
   - Staged integer and float constants reside in memory in `graph.constantStaging[id]` (a `std::shared_ptr<std::vector<uint8_t>>`).
   - Committed constants reside in `TGStore *repo` accessed via `repo->read(id)`.
   - Always check `constantStaging` first, then fall back to `repo->read(id)`. Never assume a constant is in `repo` during rewrite saturation or template deduction.

3. **E-Graph Class Canonicalization (`egraph.find(cid)`)**:
   - During saturation and clean-class propagation, e-classes are continually merged.
   - Always resolve class IDs through `egraph.find(cid)` before comparing IDs, indexing per-class arrays, or updating parent/child tracking structures.

4. **Namespace Scoping in `core/plan/`**:
   - Headers like `bucket_regions.hpp` and `search_state.hpp` contain disjoint `namespace bucket_regions { ... }` and `namespace plan { ... }` blocks mixed with global definitions.
   - To avoid subtle C++ overload hiding (where an overload in a namespace shadows a global overload of the same function name), place overloads inside the respective namespace and re-export to global scope with `using`.

5. **`ITensorStore` Mock Requirements in Tests**:
   - `ITensorStore`'s `has(LogicalId)`, `read(LogicalId)`, and `write(LogicalId, ...)` are non-virtual wrappers that call the pure virtual `std::string` methods.
   - When writing mock stores for unit tests, you must override:
     ```cpp
     bool has(const std::string &) const override { return false; }
     TensorMetadata getMetadata(const std::string &) const override { return {}; }
     std::vector<uint8_t> read(const std::string &) const override { return {}; }
     bool isWritable() const override { return true; }
     void write(const std::string &, const TensorView &, const void *, uint64_t) override {}
     ```

6. **Build Cadence & Asynchronous Link Timing**:
   - Build using `.venv/Scripts/python.exe build.py --targets test` (never invoke MSVC/clang-cl directly).
   - Because `test.cpp` links in all compiler and search modules (~45 seconds on ARM64), ensure background build tasks are completely finished before executing `test.exe` to prevent running stale binaries.

---

## Section 3: General Partial-Bucket Saturation & Reduction Speedups

### Background & Problem

When saturating a partial bucket (`saturateBucket` in `core/plan/planner.hpp`, line 1747) starting from `startingState = &full_state`:

1. Sweep 1 `FusionRule::sweep_filter` bug (`planner.hpp`, line 440): `full_state` is already saturated, and only a handful of e-classes are modified by `injectInputPartialPaths` / `injectOutputPartialPaths`. Yet `fusion->sweep_filter` is set to `nullptr` before Sweep 1 and only built *after* Sweep 1 finishes (`planner.hpp`, line 554). Consequently, Sweep 1 runs `FusionRule::match` (`matchPatternNode`) on all ~20,000+ e-nodes of `full_state` in every partial bucket.

2. Cloning huge `completed_applications` and `visited` hash sets (`rewrite.hpp`, lines 637, 750): Cloning `fullSaturationRules_` for each partial bucket copies tens of thousands of `FusionRule::ApplicationKey` structs (each containing two heap-allocated `std::vector`s).

3. Slow clean-class fixed-point loop (`planner.hpp`, lines 2050–2100): `while (changed)` scans all `0 .. maxClasses` repeatedly.

4. Late compaction in `reduceBucket` / cycle passes (`planner.hpp`, lines 2850–2900): `bucket_states[b].rebuild(true)` is called *before* `pruneEGraph` and the 5 cycle-removal passes, leaving pruned e-nodes and `hashcons` tables resident in memory across all buckets during `SearchEngine`.

### Files to Modify

1. `tensor_graphs_cpp/core/rewrite.hpp`
2. `tensor_graphs_cpp/core/plan/planner.hpp`

### Step-by-Step Implementation

#### 3.1. Pre-Sweep-1 `sweep_filter` for Partial Buckets

In `tensor_graphs_cpp/core/plan/planner.hpp`:

1. Add a parameter `size_t initial_unchanged_classes = 0` to `Planner::saturate(...)` (line 377).

2. In `Planner::saturateBucket` (line 1747), record `const size_t base_class_count = egraph.getClasses().size();` and a snapshot of `base_class_enode_counts` **before** calling `injectInputPartialPaths` and `injectOutputPartialPaths` (lines 1780–1800).

3. When `seeded` is `true` (`startingState && startingState == fullSaturationState_`), initialize `classSizeAtSweepStart` and `stagedAtSweepStart` from that pre-injection snapshot and call `buildSweepFilter()` **before** the `while (changed)` loop in `Planner::saturate` (line 440).
* Also include any e-classes that can trigger `ScatterPullUp::match` (classes whose children have a partial dirty region) in the initial frontier.
* Result: Sweep 1 of every partial bucket only runs `FusionRule` on newly injected slice/contiguous nodes and their ancestor cone within `max_pattern_depth`, skipping 99% of `full_state`.

#### 3.2. Watermark-Based Rule Cloning Instead of Deep-Copying Hash Sets

In `tensor_graphs_cpp/core/rewrite.hpp`:

1. Add `uint32_t base_enode_watermark = 0;` to `RuleCtx` (`rewrite.hpp`, line 108) and to `FusionRule`, `DotSplitRule`, `RemoveContiguous`, `RemoveCopyChains`, and `RemoveRedundantReshape`.

2. When `full_state` finishes saturating in `Planner::saturateBucket`, set `base_enode_watermark = static_cast<uint32_t>(egraph.getENodes().size())` on each rule in `fullSaturationRules_`, and clear their `visited` / `single_node_visited` / `completed_applications` sets before saving `fullSaturationRules_`.

3. In `DotSplitRule::match`, `RemoveContiguous::match`, `RemoveCopyChains::match`, `RemoveRedundantReshape::match`, and the `singleNodePatternsByOp` check in `FusionRule::match`:
* If `eNodeIdx.value < ctx.base_enode_watermark` and `ctx.sweep_filter` (or child-change tracking) indicates no child class of `eNodeIdx` has changed since `full_state`, immediately return `false` in $O(1)$ without hashing or consulting `visited` sets.
* `rule->clone()` now copies zero hash-set elements, eliminating tens of thousands of heap allocations per partial bucket.

#### 3.3. Single-Pass Bottom-Up Clean-Class Propagation in `saturateBucket`

In `tensor_graphs_cpp/core/plan/planner.hpp` (lines 2050–2100):
Replace the $O(\text{iterations} \times \text{maxClasses})$ quadratic fixed-point loop with a linear bottom-up queue:

1. Let `const uint32_t num_classes = egraph.getClasses().size();` and `const uint32_t num_enodes = egraph.getENodes().size();`.
2. Build an inverted dependency list `std::vector<std::vector<uint32_t>> parent_enodes(num_classes)`:
   - For each valid e-node `e` with owner class `owner = egraph.find(enode.getOwnerClass())`:
     - If `enode.getOpType() == OpType::INPUT`, continue (clean status dictated by graph input classification).
     - Count unique child classes `std::unordered_set<uint32_t> unique_children`: for `cid` in `enode.getChildren()`, insert `egraph.find(cid).value`.
     - Record `dirty_child_count[e] = unique_children.size()`.
     - For each canonical child `c_val` in `unique_children`, append `e` to `parent_enodes[c_val]`.

3. Initialize queue:
   - Create `std::vector<uint32_t> queue;` and `std::vector<bool> in_queue(num_classes, false);`.
   - For all canonical classes where `clean[i] == 1`, set `in_queue[i] = true` and push `i` to `queue`.

4. Process queue:
   - While `!queue.empty()`:
     - Pop canonical clean class `c = queue.back(); queue.pop_back();`.
     - For each parent e-node `e` in `parent_enodes[c]`:
       - If `dirty_child_count[e] == 0` continue;
       - Decrement `dirty_child_count[e]--;`
       - If `dirty_child_count[e] == 0`: all inputs of `e` are now clean.
       - Let `owner_val = egraph.find(egraph.getENode(ENodeId{e}).getOwnerClass()).value;`.
       - If `!clean[owner_val]`:
         - `clean[owner_val] = 1;`
         - If `!in_queue[owner_val]`: `in_queue[owner_val] = true; queue.push_back(owner_val);`

#### 3.4. Compacting & Shrinking `EGraph` After Pruning in `reduceBucket`

In `tensor_graphs_cpp/core/plan/planner.hpp` (lines 2850–2900):

1. After `pruneEGraph(bucket_states[b], all_enode_infos[b])` and the 5 cycle-removal passes (`removeDirectSelfReferenceENodes` through `removeSingleExternalComputationalProducerCycleENodes`) finish:
* If any e-nodes were removed:
  - Re-run `bucket_states[b].rebuild(true)` to renumber canonical e-classes and e-nodes compactly to $0 \dots \vert{}\text{classes}\vert{} - 1$.
  - Clear `bucket_states[b].hashcons` and `bucket_states[b].constantHashIndex` (hashcons is strictly for rewrite deduction; `SearchEngine` only queries canonical e-class lists).
  - Call `shrink_to_fit()` on `enodes` and `classes` vectors.
  - Re-align `all_enode_infos[b]` to the compacted e-nodes so indices match 1-to-1.

### Verification Strategy
1. Run `./tensor_graphs_cpp/test.exe --planner-structure` and `./tensor_graphs_cpp/test.exe --fusion-regression` to verify baseline rule execution and sweep invariants.
2. Run multi-bucket compilation with timing diagnostics and verify that saturation wall-clock time drops significantly while maintaining identical e-graph topologies and planned makespans.

---

## Section 4: General Parametric Template Saturation for Bucket Families (Prefill, Decode, & General $N$-D Regions)

### Background & Problem

For large sequence lengths ($N = 262,144$), there are $N$ Prefill buckets (`[0, t) -> [t-1, t)`) and $N$ Decode buckets (`[t-1, t) -> [t-1, t)`), or in image models, thousands of sliding-window / patch buckets.
Even though each Prefill bucket $t$ has a different dirty input shape `[B, t]` and each Decode bucket $t$ has a different attention KV slice `[B, H, t, D_h]`, **within any structural regime of $t$, the rewrite rules fire identically and produce structurally isomorphic `EGraph`s** whose new e-classes, e-nodes, and staged `INT32` constants are exact affine functions of the bucket's input/output region bounds `(start, stop)`!

### Files to Modify

1. `tensor_graphs_cpp/core/rewrite.hpp`

2. `tensor_graphs_cpp/core/plan/bucket_regions.hpp`

3. `tensor_graphs_cpp/core/plan/planner.hpp`


### Step-by-Step Implementation

#### 4.1. Defer Stage-2 Cost Pruning in `FusionRule::selectFusedCandidates` When Building a Template

In `tensor_graphs_cpp/core/rewrite.hpp` (`FusionRule::selectFusedCandidates`, line 1004):

* Notice that `Planner::computeENodeInfos` (`planner.hpp`, line 1370) **already** calls `applyDominationRules` $\to$ `isFasterEquivalentENodeDominated` (`planner.hpp`, line 77), and `pruneEGraph` deletes dominated e-nodes before search.

* Add a boolean flag `bool template_mode = false` to `RuleCtx` (`rewrite.hpp`, line 108).

* When `ctx.template_mode` is `true`, skip Stage 2 in `FusionRule::selectFusedCandidates` (keep all valid matched candidates in the template `EGraph`) and do not include `ctx.costModel` or `staged->second->data()` in `selectionMemo()`'s key.

* That makes `saturate()` **100% independent of `CostModel`** during template saturation, while `computeENodeInfos` + `pruneEGraph` still performs the **exact** concrete cost domination pruning on each instantiated bucket.

#### 4.2. Define General Bucket Regime Signatures (`BucketRegimeKey`)

In `tensor_graphs_cpp/core/plan/planner.hpp` (or a dedicated header `tensor_graphs_cpp/core/plan/bucket_template.hpp` included by `planner.hpp`):

1. Extract a flat vector of integer parameters `std::vector<int32_t> extractBucketParams(const Bucket &b, const Graph &graph, LogicalId rootId)`:
   - In deterministic sorted `LogicalId` order:
     - For each `(id, regions)` in `b.inputDirtyRegions`: for each `Dim d` in `regions[0].region`: push `d.start`, `d.stop`.
     - For each `Dim d` in `b.outputNeededRegion[0].region`: push `d.start`, `d.stop`.

2. Compute a `BucketRegimeKey` for any partial bucket `b` that classifies each dimension $d$ of each input/output region by the **exact predicates tested by rewrite rules**:
* `is_full_dim`: `start == 0 && stop == full_shape[d]`
* `starts_at_zero`: `start == 0`
* `ends_at_full`: `stop == full_shape[d]`
* `is_unit_extent`: `(stop - start) == 1`
* `dot_split_regime`: for `extent = stop - start`, record:
  * `0` if `extent < 32768` (`DotSplitRule::splitThreshold` in `rewrite.hpp`, line 3164),
  * otherwise `1 + min(ctz(extent), floor(log2(extent / 32768)))` (capturing the exact number of times `DotSplitRule`'s `N >= 32768 && N % 2 == 0` predicate can recursively split `extent`).

3. Group all partial buckets $b \neq \text{full\_idx}$ by `BucketRegimeKey`.
* For example, for $N = 262,144$:
  * Decode $t = 1$ (`[0, 1) -> [0, 1)`) and Prefill $t = 1$ (`[0, 1) -> [0, 1)`) are identical buckets (deduplicated automatically).
  * All Decode buckets $t \in [2, 32767]$ fall into **1 single regime**.
  * All Prefill buckets $t \in [2, 32767]$ fall into **1 single regime**.
  * Buckets with $t \ge 32768$ fall into a small number of regimes indexed by their 2-adic split depth (`dot_split_regime`).

#### 4.3. Two-Witness Affine Template Deduction & Fast Concrete Instantiation

Define an affine slot structure:
```cpp
struct AffineSlot
{
    enum class Target { ECLASS_SHAPE_DIM, ENODE_SHAPE_DIM, STAGED_CONSTANT_INT32 };
    Target target;
    uint32_t entity_idx; // class id, enode id, or LogicalId value
    uint32_t offset;     // dim index for shapes, or int32 index for staged constants
    int32_t base_val;
    int32_t slope_num;
    int32_t slope_den;
};

struct AffineBucketTemplate
{
    EGraph template_egraph;
    std::vector<AffineSlot> affine_slots;
    int32_t witness0_coord = 0;
};
```

For any regime group containing $\ge 3$ buckets (if a group has $\le 2$ buckets, simply run `saturateBucket` directly on them):

1. **Pick two witness buckets $b^{(0)}, b^{(1)}$ in the group** whose parameter vectors $p^{(0)}, p^{(1)}$ differ in varying coordinate $\tau$ (e.g., the first two buckets in the regime, where $\Delta \tau = \tau^{(1)} - \tau^{(0)} \neq 0$).
2. **Run `saturateBucket` on $b^{(0)}$ and $b^{(1)}$ with `template_mode = true`:**
* Because $b^{(0)}$ and $b^{(1)}$ have the exact same `BucketRegimeKey`, every rewrite rule fires in the exact same order, producing two `EGraph`s $G^{(0)}$ and $G^{(1)}$ with the same number of e-classes and e-nodes.
* Verify structural isomorphism: `G0.getClasses().size() == G1.getClasses().size()` and `G0.getENodes().size() == G1.getENodes().size()`. (If any custom rule ever causes a mismatch, fall back to direct `saturateBucket`—guaranteeing 100% soundness).

3. **Build `AffineBucketTemplate` by diffing $G^{(0)}$ and $G^{(1)}$ for indices $\ge \text{full\_state}$:**
* For every e-class $c \ge \vert{}G_{\text{full}}.\text{classes}\vert{}$, compare `G0.classes[c].shape[d]` and `G1.classes[c].shape[d]`. Wherever they differ, record an `AffineSlot` with target `ECLASS_SHAPE_DIM`.
* For every e-node $e \ge \vert{}G_{\text{full}}.\text{enodes}\vert{}$, compare `G0.enodes[e].shape[d]` and `G1.enodes[e].shape[d]`. Wherever they differ, record an `AffineSlot` with target `ENODE_SHAPE_DIM`.
* For staged `INT32` constants in `G0.constantStaging` vs `G1.constantStaging` (using the `constantStaging` lookup learned in Section 2), record an `AffineSlot` with target `STAGED_CONSTANT_INT32` for any differing integer element.

4. **Instantiate Each Remaining Bucket $b$ in the Regime in $O(\vert{}E_{\text{new}}\vert{})$ Time:**
* Copy `template_egraph` into `bucket_states[b]`.
* Let `int32_t delta_tau = tau_b - template.witness0_coord;`.
* For each `slot` in `template.affine_slots`:
  - Calculate `val = slot.base_val + (slot.slope_num * delta_tau) / slot.slope_den;`
  - If `ECLASS_SHAPE_DIM`: update `bucket_states[b].classes[slot.entity_idx].shape[slot.offset] = val;`
  - If `ENODE_SHAPE_DIM`: update `bucket_states[b].enodes[slot.entity_idx].shape[slot.offset] = val;`
  - If `STAGED_CONSTANT_INT32`: update `int32_t` at index `slot.offset` in `bucket_states[b].constantStaging[LogicalId{slot.entity_idx}]`.
* Recompute strides:
  - For non-view e-nodes, set `enode.strides = calcContiguousStrides(enode.shape)`.
  - For view e-nodes, propagate view strides from their parent class view traits.
* Call `bucket_states[b].rebuild(true)`.
* In `computeENodeInfos(bucket_states[b])` (`planner.hpp`, line 1370), for each e-node `i >= full_state_enodes` with `kernelId != 0`, verify `KernelRegistry::get().getKernel(enode.getKernelId()).matches(...)` on the instantiated concrete input/output `TensorNode` shapes and strides; if a kernel's custom `match` function rejects concrete $\tau^{(b)}$ (e.g., due to vector alignment), set `info.cost = TGConstants::INF` so `pruneEGraph` removes it.
* Then run `computeENodeInfos` and `pruneEGraph` as usual.

### Verification Strategy
1. Add a test in `tensor_graphs_cpp/tests/bucket_template_regression.hpp` comparing the pruned `EGraph` (`bucket_states[b]`) and `all_enode_infos[b]` produced by template instantiation against direct `saturateBucket` across all buckets for $N = 16$ and $N = 64$.
2. Verify that the set of surviving e-classes, e-nodes, and costs in every bucket is **100% identical** and produces the exact same compiled plans.

---

## Section 5: `SearchState` Flat Indexing, Cache-First AND/OR Decomposition, Sound Per-Bucket Pruning, and Parallel Subproblem Search

### Background & Problem

1. Hash-Map Overhead in `SearchState` (`core/plan/search_state.hpp`, lines 775–777): `selected_vars[b]`, `start_vars[b]`, and `offset_vars[b]` are `std::unordered_map<EClassId, VarId>`. Because `reduceBucket` compacts `EClassId`s to $0 \dots \vert{}\text{classes}\vert{} - 1$, hash lookups waste memory and cache lines across all 26 propagators.


2. Coupled Multi-Bucket Branching in `HeuristicBrancher` (`core/plan/brancher.hpp`, lines 743–825): Currently, `chooseBranch` branches on `chooseSelection(true)` across all buckets *before* `chooseCache` fixes all `CACHED` variables.


3. **Mathematical Independence Given `CACHED`:**
* Once all `VarType::CACHED` variables are fixed, `CachedOffsetAllocationPropagator` (`core/plan/propagators/cached_offset_allocation.hpp`, line 88) deterministically fixes the `OFFSET` of every cached base across all buckets to `[prealloc_end, cache_end)`.


* `CacheFreshnessPropagator` (`cache_freshness.hpp`) removes `0` from `selected` in every bucket for cached bases ($c_i = 1$), and `CacheExclusionPropagator` (`cache_exclusion.hpp`) removes `CACHE`/`SCATTER` alternatives for uncached bases ($c_i = 0$).


* **Every remaining constraint and objective term is 100% bucket-local**!





### Files to Modify

1. `tensor_graphs_cpp/core/plan/search_state.hpp`

2. `tensor_graphs_cpp/core/plan/brancher.hpp`

3. `tensor_graphs_cpp/core/plan/search_engine.hpp`

4. `tensor_graphs_cpp/core/plan/propagators/critical_path.hpp`, `engine_workload.hpp`, `start_run_makespan.hpp`


### Step-by-Step Implementation

#### 5.1. Add Flat Dense Lookup Tables in `SearchState`

In `tensor_graphs_cpp/core/plan/search_state.hpp` (lines 885–895):

1. Add dense per-bucket vectors alongside the existing maps (keeping the maps or a lightweight wrapper if needed for API compatibility, or providing $O(1)$ inline accessors):
```cpp
std::vector<std::vector<VarId>> dense_selected_vars; // [b][cid.value] -> VarId (or kInvalidVarId)
std::vector<std::vector<VarId>> dense_start_vars;    // [b][cid.value] -> VarId
std::vector<std::vector<VarId>> dense_offset_vars;   // [b][cid.value] -> VarId
// Range of VarIds belonging to bucket b: [bucket_var_begin[b], bucket_var_end[b])
std::vector<VarId> bucket_var_begin;
std::vector<VarId> bucket_var_end;
```

2. Populate `dense_selected_vars[b]`, `dense_start_vars[b]`, `dense_offset_vars[b]`, `bucket_var_begin[b]`, and `bucket_var_end[b]` in `SearchState::addBucketVariables(uint32_t b)` (`search_state.hpp`, line 1046).

3. Add a helper `bool allCachedVarsFixed() const` to `SearchState`:
```cpp
bool allCachedVarsFixed() const
{
    for (const auto &[base_id, vid] : cached_vars)
        if (!domains[vid].isFixed())
            return false;
    return true;
}
```

#### 5.2. Branch on `CACHED` First in `HeuristicBrancher::chooseBranch`

In `tensor_graphs_cpp/core/plan/brancher.hpp` (`HeuristicBrancher::chooseBranch`, line 1045):

1. Move `if (chooseCache(state, out_decision)) return true;` (line 782) **ahead** of `chooseSelection(state, out_decision, true)` (line 830).

2. In `chooseCache` (`brancher.hpp`, line 782), if the root primal heuristic found an incumbent (`!incumbent_cached_nodes.empty()`), prefer the incumbent's cache value (`1` if in `incumbent_cached_nodes`, else `0`) as the left branch instead of always `0`.

#### 5.3. Sound Per-Bucket Upper-Bound Pruning (`OR(LB_b >= UB_b)`)

In `tensor_graphs_cpp/core/plan/search_state.hpp` and `search_engine.hpp`:

1. Add `std::vector<float> bucket_best_costs;` (initialized to `TGConstants::INF` per bucket) and `std::unordered_set<BaseEClassId> best_cost_cache_set;` to `SearchState`.

2. When `SearchEngine::recordIncumbent` (`search_engine.hpp`, line 720) records a new best plan (or when solving for a fixed cache set), record each bucket $b$'s makespan in `bucket_best_costs[b]` for that cache set.

3. Add a method `bool isBucketLowerBoundPruned(uint32_t b) const` on `SearchState`:
* When `allCachedVarsFixed()` is `true` and the current fixed cache set matches `best_cost_cache_set`:
  * Bucket $b$ is completely independent of all other buckets $b' \neq b$.
  * Furthermore, any bucket $b' \neq b$ whose variables in `[bucket_var_begin[b'], bucket_var_end[b'])` have not been modified since the root/cache-fix point already has a known achievable makespan `bucket_best_costs[b']`.
  * If in the current bucket $b$ being branched, `bucket_lower_bounds[b] >= bucket_best_costs[b]` (and all other buckets are at their best or fixed), return `true` (prune immediately).
* Also keep the global check `lower_bound >= best_cost` active at all times.

4. Call `state.isBucketLowerBoundPruned(b)` in `CriticalPathPropagator::propagate`, `EngineWorkloadPropagator::propagate`, `StartRunMakespanPropagator::propagate`, and `SearchEngine::runPropagators`.

#### 5.4. Parallel Per-Bucket Subproblem Solve Given Fixed `CACHED`

In `tensor_graphs_cpp/core/plan/search_engine.hpp`:

1. Add a helper `bool solveBucketSubproblem(uint32_t b, float timeout_seconds)` that solves only bucket $b$'s variables `[bucket_var_begin[b], bucket_var_end[b'])` when `allCachedVarsFixed()` holds:
* Construct a lightweight single-bucket `SearchState` slice (or run `SearchEngine` restricted to bucket `b` by setting `active_bucket_filter = b` on `HeuristicBrancher`), with `state.best_cost = bucket_best_costs[b]` (seeded by the root primal heuristic's validated makespan for bucket $b$).
* Because `bucket_relaxation_lower_bounds[b]` is already computed at the root, if `bucket_relaxation_lower_bounds[b] >= bucket_best_costs[b] - 1e-6f`, bucket $b$ is **already proven optimal at the root in 0 iterations**!
* For any bucket $b$ where `bucket_relaxation_lower_bounds[b] < bucket_best_costs[b]`, `ThreadPool::get().parallel_for` can run `solveBucketSubproblem(b)` in parallel across buckets!

### Verification Strategy
* Run targeted propagator regressions:
  ```powershell
  .\tensor_graphs_cpp\test.exe --relaxation-bound
  .\tensor_graphs_cpp\test.exe --start-bounds
  .\tensor_graphs_cpp\test.exe --start-run-makespan
  ```
* Verify that the optimal makespan and extracted plans match baseline, and check log outputs to confirm that buckets with `LB_b == UB_b` terminate immediately with 0 branch-and-bound nodes explored.

---

## Section 6: Cross-Bucket Plan Seeding (`SELECTED`, `START`, and Dynamic `OFFSET` Reconstruction)

### Background & Problem

In `buildRootPrimalPlans` and `improvePrimalPlan` (`core/plan/primal_heuristic.hpp`, lines 435, 947), every bucket $b \in \{0, \dots, 2N\}$ independently runs:
* `RelaxationBoundComputer::evaluate` + `regionEstimates`
* `extractSelection` / `extractCovering`
* `listSchedule`
* `improveBucketLocalSearch` (line 752, looping over all active classes and testing alternative swaps using `listSchedule`)
* `SearchEngine::tryPlan` (line 760, fixing variables and running `runPropagators` + `preferredOffset`).

Adjacent or structurally isomorphic buckets (e.g. Decode $t-1$ and Decode $t$, or Prefill $t-1$ and Prefill $t$) have identical e-graph topology and nearly identical relative kernel speeds, differing only in tensor sizes (`OFFSET` footprints) and exact durations.

### Files to Modify

1. `tensor_graphs_cpp/core/plan/primal_heuristic.hpp`
2. `tensor_graphs_cpp/core/plan/search_engine.hpp`

### Step-by-Step Implementation

#### 6.1. Structural Lineage Signature for Cross-Bucket `SELECTED` Seeding

In `tensor_graphs_cpp/core/plan/primal_heuristic.hpp`:

1. Add a helper `bool transferBucketSelection(const SearchState &state, uint32_t src_b, const RelaxationBucket &src_rb, const PrimalBucketPlan &src_plan, uint32_t dst_b, const RelaxationBucket &dst_rb, const std::unordered_set<BaseEClassId> &allowed_cache, PrimalBucketPlan &dst_plan)`:
* Fast path (Template / Isomorphic buckets): If `src_rb.classes.size() == dst_rb.classes.size()` and for every $u$, `src_rb.classes[u].cid == dst_rb.classes[u].cid` and the chosen alternative `src_plan.value[u]` exists in `dst_rb.classes[u].alternatives` (with the same `KernelId` and `OpType`) and satisfies `!alt.requires_cache || allowed_cache.count(dst_rb.classes[u].base)`:
  * Copy `dst_plan.value = src_plan.value` directly.
* General path (Different bucket geometries):
  * Build a map from `(BaseEClassId, LogicalId, OpType, KernelId)` chosen in `src_b` to the corresponding alternative in `dst_rb`.
  * Top-down from `dst_rb.root` (as in `extractSelection`), if a class $u$ in `dst_rb` has a matching feasible alternative from `src_b` whose children are all feasible, select it; otherwise fall back to `extractSelection`'s minimum-estimate choice for $u$.

2. Once `dst_plan.value` (`SELECTED`) is populated for bucket `dst_b`, call `listSchedule(dst_rb, dst_plan)` (`primal_heuristic.hpp`, line 79):
* `listSchedule` uses **bucket `dst_b`'s exact concrete durations (`alt.duration`) and engine masks** to compute the makespan-optimal `dst_plan.order` (`START` ranks) and exact simulated `dst_plan.makespan`.

#### 6.2. Seeding `improveBucketLocalSearch` Across Similar Buckets

In `tensor_graphs_cpp/core/plan/primal_heuristic.hpp` (`improvePrimalPlan`, line 947):

1. Partition weighted buckets into structural isomorphism groups (buckets where `rb.classes.size()` and per-class `(OpType, KernelId)` alternative lists match—which groups all Decode buckets in a regime together and all Prefill buckets in a regime together).
2. In `improvePrimalPlan`:
* First run `improveBucketLocalSearch` on **one representative bucket** $b_{\text{rep}}$ per group (or a small logarithmic sample of sequence lengths $t \in \{2, 4, 8, \dots\}$ for prefill).
* Seed the improved `plan.buckets[b_rep].value` into the other buckets $b$ of the group via `transferBucketSelection` + `listSchedule(contexts[b].rb, plan.buckets[b])`.
* Then run `improveBucketLocalSearch` on the remaining buckets $b$: because they start at the already-converged selection from $b_{\text{rep}}$, `improveBucketLocalSearch` either finishes in a single fast verification pass (0 swaps needed) or makes 1–2 boundary swaps where a kernel crossover occurred.

#### 6.3. Fast Concrete `OFFSET` Reconstruction in `SearchEngine::tryPlan`

In `tensor_graphs_cpp/core/plan/search_engine.hpp` (`SearchEngine::tryPlan`, line 760):

1. Optimize `tryPlan` using the bucket decomposition from Section 5:
* Fix all `CACHED` variables first and run `CachedOffsetAllocationPropagator` (which fixes all cached base offsets across all buckets in one step).
* For each bucket $b$ independently:
  * Fix `SELECTED` and `START` for bucket $b$ from `plan.buckets[b]`.
  * Run bucket-local propagation on bucket $b$.
  * For any unfixed `OFFSET` variables in `[state.bucket_var_begin[b], state.bucket_var_end[b])` (when `state.private_layout_spaces` did not already fix them at the root), assign their concrete page offsets using `placement.preferredOffset(state, b, info.eclass_id, d)` (`brancher.hpp`, line 330)—which uses bucket $b$'s exact concrete page sizes (`eclass_page_sizes_[b]`) and exact buffer lifetimes `[start, end)`.
* Because each bucket $b$'s `SELECTED`, `START`, and `OFFSET` validation in `tryPlan` is completely independent once `CACHED` is fixed, this per-bucket validation can also run in parallel across buckets.

### Verification Strategy
* Run multi-bucket tests and verify that:
  1. `tryPlan` accepts seeded plans with 0 propagation conflicts.
  2. The final compiled plans achieve the exact same (or better) makespan as baseline.
  3. Total primal heuristic planning time scales sub-linearly with the number of buckets.

---

## Summary Checklist for the Implementing Agent

| Section | Primary Files | Key Invariant to Verify After Implementation | Status |
| --- | --- | --- | --- |
| **1. `SCATTER` Key Canonicalization** | `cost_model.hpp`, `kernel_bench.hpp`, `planner.hpp`, `rewrite.hpp` | `SCATTER` hits cost/fusion memos across all buckets; `discoverCalls` emits 1 record per `(update_shape, out_shape)`. | **DONE** (`--scatter-canonical`) |
| **2. Analytical `TRIU` + `NEG_INF` Mask** | `execute_ref_graph.hpp`, `bucket_regions.hpp`, `rewrite.hpp` | `foldCleanTensors` skips large `FILL`/`TRIU`; `computeRegionFacts` & `InfinityDomination` run in $O(1)$ without $N^2$ float scans; identical `BucketRegions`. | **NEXT** |
| **3. Partial-Bucket Saturation Speedups** | `planner.hpp`, `rewrite.hpp` | `sweep_filter` active on Sweep 1 of partial buckets; single-pass clean-class queue; `EGraph` compacted after `pruneEGraph`. | PENDING |
| **4. Parametric Template Saturation** | `planner.hpp`, `rewrite.hpp` | Isomorphic regime buckets instantiate concrete shapes/constants in $O(\vert{}E_{\text{new}}\vert{})$ and run exact `computeENodeInfos` + `pruneEGraph`. | PENDING |
| **5. Flat `SearchState` & Cache-First Decomposition** | `search_state.hpp`, `brancher.hpp`, `search_engine.hpp` | Dense vector lookups; `chooseCache` runs first; sound per-bucket `LB_b >= UB_b` pruning when `CACHED` is fixed. | PENDING |
| **6. Cross-Bucket Seeding & `OFFSET` Packing** | `primal_heuristic.hpp`, `search_engine.hpp` | `SELECTED` seeded across regime buckets, `START` recomputed via `listSchedule`, `OFFSET` packed for concrete sizes via `preferredOffset`. | PENDING |