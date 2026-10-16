# Tensor Graphs Compiler: Large-Scale Multi-Bucket Optimization Implementation Plan

## Objective & Non-Negotiable Invariants

This plan scales `tensor_graphs_cpp` compilation to large sequence lengths (e.g., $N = 262,144$, where $2N+1 = 524,289$ buckets are compiled for full + prefill $[0, t) \to [t-1, t)$ + decode $[t-1, t) \to [t-1, t)$) and arbitrary $N$-D dirty regions (e.g., 2D spatial patches in diffusion models).

**Strict Optimality Invariant:** Every optimization in this plan is **100% exact and optimality-preserving**. No shape canonicalization in cost evaluation, no approximate cost models, and no heuristic pruning that can discard a globally optimal execution plan are permitted. Every bucket must be evaluated and validated against its exact concrete shapes, strides, constants, kernel match predicates, and `MakespanSimulator` cost.

The plan is divided into **6 self-contained, sequential sections**. Implement, test, and commit each section before starting the next.

---

## Section 1: Canonicalizing `SCATTER` Keys Across Cost, Benchmark, and Fusion Memos

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

## Section 2: Analytical `TRIU` + `NEG_INF` Mask Predicate & Skipping Clean Mask Materialization

### Background & Problem

Currently, causal attention masks built via `fill(TGConstants::NEG_INF, shape) -> triu(k=1)` (and optional `repeat`/`reshape`/`permute`/`slice` views) trigger three $O(N^2)$ bottlenecks:

1. `Session::foldCleanTensors()` (`core/session.hpp`, line 92) calls `executeReferenceGraph` with `only_clean_nodes = true` (`core/common/execute_ref_graph.hpp`, line 62). Because `FILL` and `TRIU` have constant inputs, `foldCleanTensors()` materializes the $N \times N$ float tensor on the CPU and writes it to `.reftensors` (275 GB at $N = 262,144$).


2. `bucket_regions::unmaskedRegions` (`core/plan/bucket_regions.hpp`, line 153) scans all $N \times N$ floats and builds a `std::vector<Region>` staircase with $N$ individual row boxes. `computeBucketRegions` then intersects against this $N$-element vector $2N$ times ($O(N^2)$ per attention layer).


3. `InfinityDomination::apply` (`core/rewrite.hpp`, line 911) also reads the materialized $N \times N$ float array from `constantStaging` or `repo` and scans all $N \times N$ elements.



### Files to Modify

1. `tensor_graphs_cpp/core/common/execute_ref_graph.hpp`

2. `tensor_graphs_cpp/core/plan/bucket_regions.hpp`

3. `tensor_graphs_cpp/core/rewrite.hpp`


### Step-by-Step Implementation

#### 2.1. Do Not Materialize Large Constant `FILL` / `TRIU` Tensors in `executeReferenceGraph`

In `tensor_graphs_cpp/core/common/execute_ref_graph.hpp`:

1. Inside `executeReferenceGraph`, when `options.only_clean_nodes` is `true`, add a helper `isSymbolicCleanMaskOrFill(LogicalId id, const Graph &graph)` that returns `true` if:
* `countElements(graph.getNode(id).getShape()) > 4096`, AND
* `id` is `OpType::FILL`, `OpType::TRIU`, or a view op (`REPEAT`, `RESHAPE`, `PERMUTE`, `SLICE`, `CONTIGUOUS`) whose data child (`child_ids[0]`) recursively satisfies `isSymbolicCleanMaskOrFill` or is `OpType::FILL` / `OpType::TRIU`.




2. Mark `is_clean[node_id] = false` for any node where `isSymbolicCleanMaskOrFill(node_id, graph)` is `true` (and consequently any downstream node that consumes it) during the `options.only_clean_nodes` pass so `foldCleanTensors()` never allocates or writes multi-gigabyte `FILL`/`TRIU` tensors to `repo`.



#### 2.2. Analytical `TriuMaskPredicate` in `core/plan/bucket_regions.hpp`

In `tensor_graphs_cpp/core/plan/bucket_regions.hpp`:

1. Define a compact struct `RegionSet` (or extend `RegionFacts` with a side-table `std::unordered_map<LogicalId, TriuBound> triuBounds`) representing a half-space constraint on two axes of a tensor:
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


2. In `computeRegionFacts(const Graph &graph, const std::vector<LogicalId> &topo, TGStore *repo)` (`bucket_regions.hpp`, line 250):


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
* The restricted column interval is $[c_0, \min<uint32\_t>(c_1, static\_cast<uint32\_t>(c_{\text{limit}})))$. If $c_0 \ge \text{stop}$, the intersection is empty; otherwise return `box` with `box.region[b.col_axis].stop` updated to that minimum!


* `axisRange` with `TriuBound b` (`bucket_regions.hpp`, line 213):


* When `axis == b.col_axis`:
* Compute $c_{\text{stop}} = \min<int64\_t>(c_1, static\_cast<int64\_t>(r_1 - 1) + b.\text{diag\_k})$.
* If $c_0 \ge c_{\text{stop}}$, return `std::nullopt`.
* If `everyRow` is `true`, every row $r \in [r_0, r_1)$ must have at least one valid column in $[c_0, c_1)$, which holds iff the most restrictive row $r = r_0$ satisfies $c_0 < \min<int64\_t>(c_1, static\_cast<int64\_t>(r_0) + b.\text{diag\_k})$. If not, return `std::nullopt`.
* Return `Dim{c_0, static_cast<uint32_t>(c_{\text{stop}})}`.




* `dotForwardFromB` and `dotNeededB` (`bucket_regions.hpp`, lines 395–426):


* When `suppA` has a `TriuBound` (with `row_axis = 1`, `col_axis = 2`), evaluate the intersection of each box `b` (or `o`) with the `TriuBound` in $O(1)$ arithmetic instead of looping over $N$ staircase boxes!





#### 2.3. Analytical `TRIU` Matching in `InfinityDomination` (`core/rewrite.hpp`)

In `tensor_graphs_cpp/core/rewrite.hpp` (`InfinityDomination`, line 825):

1. Add a helper `bool isTriuMaskEClass(EClassId cid, const EGraph &egraph, TriuBound &out_bound)` that walks the e-graph from `cid` through view e-nodes (`REPEAT`, `RESHAPE`, `PERMUTE`, `SLICE`, `CONTIGUOUS`, `COPY_TO`) to an `OpType::TRIU` e-node whose input 0 is `OpType::FILL` with scalar constant `TGConstants::NEG_INF` and input 1 is scalar `INT32` constant `diag_k`.


2. Update `InfinityDomination::match` (line 883) to match `ADD` nodes where either child satisfies `isConstantFloat(child, ctx)` **or** `isTriuMaskEClass(child, ctx.egraph, dummy_bound)`.


3. In `InfinityDomination::apply` (line 911):


* If the mask child `constClass` is matched via `isTriuMaskEClass(constClass, egraph, triu_bound)`:
* In partial buckets where `ctx.regions && outClass.logical_id != LogicalId{} && outClass.base_eclass_id != BaseEClassId{}` (`rewrite.hpp`, line 990), `InfinityDomination` **already** uses `const Region box = bucket_regions::boundingBox(live)` and sets `nonInfRegions = {box}`, `cached_update = true`!


* In the full bucket (when `cached_update == false`), compute the $k \le \text{kMaxRegions}$ (8) covering boxes analytically from `triu_bound` in $O(\text{kMaxRegions})$ arithmetic (dividing the row range $[0, R)$ into `kMaxRegions` equal row stripes $[r_{\text{first}}, r_{\text{last}})$, each with column range $[0, \min(C, r_{\text{last}} - 1 + \text{diag\_k}))$)—producing the **exact same `nonInfRegions**` as `coverWithFewRegions(mergeRegions(runs), kMaxRegions)` in $O(1)$ time without reading any constant buffer!





### Verification

* Run existing attention / causal-mask tests at small $N$ (e.g., $N = 16, 64$) and verify that `computeRegionFacts`, `computeBucketRegions`, and `InfinityDomination` produce the **exact same `Region`s, `EGraph`s, and compiled plans** whether the mask is materialized or matched via `TriuBound`.


* Test `computeRegionFacts` and `computeBucketRegions` on a graph with $N = 262,144$ and verify that `foldCleanTensors()` and `computeRegionFacts` complete in milliseconds with $O(1)$ memory instead of allocating 275 GB.

---

## Section 3: General Partial-Bucket Saturation & Reduction Speedups

### Background & Problem

When saturating a partial bucket (`saturateBucket` in `core/plan/planner.hpp`, line 1255) starting from `startingState = &full_state`:

1. Sweep 1 `FusionRule::sweep_filter` bug (`planner.hpp`, line 441): `full_state` is already saturated, and only a handful of e-classes are modified by `injectInputPartialPaths` / `injectOutputPartialPaths`. Yet `fusion->sweep_filter` is set to `nullptr` before Sweep 1 and only built *after* Sweep 1 finishes (`planner.hpp`, line 554). Consequently, Sweep 1 runs `FusionRule::match` (`matchPatternNode`) on all ~20,000+ e-nodes of `full_state` in every partial bucket.


2. Cloning huge `completed_applications` and `visited` hash sets (`rewrite.hpp`, lines 318, 439): Cloning `fullSaturationRules_` for each partial bucket copies tens of thousands of `FusionRule::ApplicationKey` structs (each containing two heap-allocated `std::vector`s).


3. Slow clean-class fixed-point loop (`planner.hpp`, lines 1463–1518): `while (changed)` scans all `0 .. maxClasses` repeatedly.


4. Late compaction in `reduceBucket` (`planner.hpp`, lines 1766–1827): `bucket_states[b].rebuild(true)` is called *before* `pruneEGraph` and the 5 cycle-removal passes, leaving pruned e-nodes and `hashcons` tables resident in memory across all buckets during `SearchEngine`.



### Files to Modify

1. `tensor_graphs_cpp/core/rewrite.hpp`

2. `tensor_graphs_cpp/core/plan/planner.hpp`


### Step-by-Step Implementation

#### 3.1. Pre-Sweep-1 `sweep_filter` for Partial Buckets

In `tensor_graphs_cpp/core/plan/planner.hpp`:

1. Add a parameter `size_t initial_unchanged_classes = 0` to `Planner::saturate(...)` (line 307).


2. In `Planner::saturateBucket` (line 1330), record `const size_t base_class_count = egraph.getClasses().size();` and a snapshot of `base_class_enode_counts` **before** calling `injectInputPartialPaths` and `injectOutputPartialPaths`.


3. When `seeded` is `true` (`startingState && startingState == fullSaturationState_`), initialize `classSizeAtSweepStart` and `stagedAtSweepStart` from that pre-injection snapshot and call `buildSweepFilter()` **before** the `while (changed)` loop in `Planner::saturate` (right at line 442)!


* Also include any e-classes that can trigger `ScatterPullUp::match` (classes whose children have a partial dirty region) in the initial frontier.
* Result: Sweep 1 of every partial bucket only runs `FusionRule` on newly injected slice/contiguous nodes and their ancestor cone within `max_pattern_depth`, skipping 99% of `full_state`!



#### 3.2. Watermark-Based Rule Cloning Instead of Deep-Copying Hash Sets

In `tensor_graphs_cpp/core/rewrite.hpp`:

1. Add `uint32_t base_enode_watermark = 0;` to `FusionRule`, `DotSplitRule`, `RemoveContiguous`, `RemoveCopyChains`, and `RemoveRedundantReshape`.


2. When `full_state` finishes saturating in `Planner::saturateBucket`, set `base_enode_watermark = static_cast<uint32_t>(egraph.getENodes().size())` on each rule in `fullSaturationRules_`, and clear their `visited` / `single_node_visited` / `completed_applications` sets before saving `fullSaturationRules_`.


3. In `DotSplitRule::match`, `RemoveContiguous::match`, `RemoveCopyChains::match`, `RemoveRedundantReshape::match`, and the `singleNodePatternsByOp` check in `FusionRule::match`:
* If `eNodeIdx < base_enode_watermark` and `sweep_filter` (or child-change bit) indicates no child of `eNodeIdx` has changed since `full_state`, immediately skip `eNodeIdx` in $O(1)$ without consulting a hash set!


* `rule->clone()` now copies zero hash-set elements!



#### 3.3. Single-Pass Bottom-Up Clean-Class Propagation in `saturateBucket`

In `tensor_graphs_cpp/core/plan/planner.hpp` (lines 1463–1518):
Replace the `while (changed)` loop over `0 .. maxClasses` with a single-pass bottom-up queue:

1. Build `std::vector<std::vector<uint32_t>> parent_enodes(maxClasses)` mapping each canonical child e-class `c` to the e-node indices `e` that consume `c`, and `std::vector<uint32_t> dirty_child_count(numENodes, 0)`.


2. Push all initially clean canonical e-classes (`clean[i] == 1`) into a `std::vector<uint32_t> queue`.


3. For each e-node `e` whose owner class `owner` is not yet clean and whose `opType != OpType::INPUT`: if `dirty_child_count[e] == 0` (e.g., 0-child ops or all children already clean), mark `clean[owner] = 1` and push `owner` to `queue`.


4. Process `queue`: when popping a clean e-class `c`, for each `e` in `parent_enodes[c]`, decrement `dirty_child_count[e]`. If `dirty_child_count[e] == 0` and `egraph.getENode(ENodeId{e}).getOpType() != OpType::INPUT`, mark its canonical owner class `owner` clean (`clean[owner] = 1`) and push `owner` to `queue`.



#### 3.4. Compacting & Shrinking `EGraph` After Pruning in `reduceBucket`

In `tensor_graphs_cpp/core/plan/planner.hpp` (`reduceBucket`, lines 1766–1827):

1. After `pruneEGraph(bucket_states[b], all_enode_infos[b])` and the 5 cycle-removal passes (`removeDirectSelfReferenceENodes` through `removeSingleExternalComputationalProducerCycleENodes`) finish, if any e-nodes were removed:
* Compact `bucket_states[b]` (or clear `bucket_states[b].hashcons` and `bucket_states[b].constantHashIndex` and `shrink_to_fit()` on `enodes` and `classes` vectors) so dead e-nodes and hash tables do not consume RAM during `SearchEngine`.





### Verification

* Run full compilation tests and verify that compiled graphs, selected kernels, and makespan costs are 100% identical to baseline while partial-bucket saturation time and memory footprint drop substantially.

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

In `tensor_graphs_cpp/core/rewrite.hpp` (`FusionRule::selectFusedCandidates`, line 768):

* Notice that `Planner::computeENodeInfos` (`planner.hpp`, line 986) **already** calls `applyDominationRules` $\to$ `isFasterEquivalentENodeDominated` (`planner.hpp`, line 77), and `pruneEGraph` deletes dominated e-nodes before search.


* Add a boolean flag `bool template_mode = false` to `RuleCtx` (`rewrite.hpp`, line 117).


* When `ctx.template_mode` is `true`, skip Stage 2 in `FusionRule::selectFusedCandidates` (keep all valid matched candidates in the template `EGraph`) and do not include `ctx.costModel` or `staged->second->data()` in `selectionMemo()`'s key.


* That makes `saturate()` **100% independent of `CostModel**` during template saturation, while `computeENodeInfos` + `pruneEGraph` still performs the **exact** concrete cost domination pruning on each instantiated bucket!

#### 4.2. Define General Bucket Regime Signatures (`BucketRegimeKey`)

In `tensor_graphs_cpp/core/plan/planner.hpp` (or a new header `tensor_graphs_cpp/core/plan/bucket_template.hpp` included by `planner.hpp`):

1. Extract a flat vector of integer parameters `std::vector<int32_t> extractBucketParams(const Bucket &b, const Graph &graph, LogicalId rootId)` containing, in deterministic sorted `LogicalId` order, the `(start, stop)` coordinates of each `Dim` in `b.inputDirtyRegions` and `b.outputNeededRegion`.


2. Compute a `BucketRegimeKey` for any partial bucket `b` that classifies each dimension $d$ of each input/output region by the **exact predicates tested by rewrite rules**:
* `is_full_dim`: `start == 0 && stop == full_shape[d]`
* `starts_at_zero`: `start == 0`
* `ends_at_full`: `stop == full_shape[d]`
* `is_unit_extent`: `(stop - start) == 1`
* `dot_split_regime`: for `extent = stop - start`, record:
* `0` if `extent < 32768` (`DotSplitRule::splitThreshold` in `rewrite.hpp`, line 1791),


* otherwise `1 + min(ctz(extent), floor(log2(extent / 32768)))` (capturing the exact number of times `DotSplitRule`'s `N >= 32768 && N % 2 == 0` predicate can recursively split `extent`!).




3. Group all partial buckets $b \neq \text{full\_idx}$ by `BucketRegimeKey`.
* For example, for $N = 262,144$:
* Decode $t = 1$ (`[0, 1) -> [0, 1)`) and Prefill $t = 1$ (`[0, 1) -> [0, 1)`) are **identical buckets** (deduplicated automatically!).
* All Decode buckets $t \in [2, 32767]$ fall into **1 single regime**.
* All Prefill buckets $t \in [2, 32767]$ fall into **1 single regime**.
* Buckets with $t \ge 32768$ fall into a small number of regimes indexed by their 2-adic split depth (`dot_split_regime`).





#### 4.3. Two-Witness Affine Template Deduction & Fast Concrete Instantiation

For any regime group containing $\ge 3$ buckets (if a group has $\le 2$ buckets, simply run `saturateBucket` directly on them):

1. **Pick two witness buckets $b^{(0)}, b^{(1)}$ in the group** whose parameter vectors $p^{(0)}, p^{(1)}$ differ in their varying coordinate(s) $\tau$ (e.g., the first two buckets in the regime, where $\Delta \tau = \tau^{(1)} - \tau^{(0)} \neq 0$).
2. **Run `saturateBucket` on $b^{(0)}$ and $b^{(1)}$ with `template_mode = true`:**
* Because $b^{(0)}$ and $b^{(1)}$ have the exact same `BucketRegimeKey`, every rewrite rule fires in the exact same order, producing two `EGraph`s $G^{(0)}$ and $G^{(1)}$ with the same number of e-classes and e-nodes!
* Verify structural isomorphism: `G0.getClasses().size() == G1.getClasses().size()` and `G0.getENodes().size() == G1.getENodes().size()`. (If any custom rule ever causes a mismatch, fall back to direct `saturateBucket`—guaranteeing 100% soundness).


3. **Build `AffineBucketTemplate` by diffing $G^{(0)}$ and $G^{(1)}$ for indices $\ge \text{full\_state}$:**
* For every e-class $c \ge \vert{}G_{\text{full}}.\text{classes}\vert{}$, compare `G0.classes[c].shape[d]` and `G1.classes[c].shape[d]`. Wherever they differ, record an `AffineSlot`:

$$\text{val}(\tau) = \text{val}^{(0)} + \frac{\text{val}^{(1)} - \text{val}^{(0)}}{\tau^{(1)} - \tau^{(0)}} \cdot (\tau - \tau^{(0)})$$


* Record the same `AffineSlot`s for `constantStaging[c]` `int32_t` vectors (such as `starts`, `ends`, `reshape` target shapes, `repeat` counts, and `scatter` shapes) and for `ENode::shape`.




4. **Instantiate Each Remaining Bucket $b$ in the Regime in $O(\vert{}E_{\text{new}}\vert{})$ Time:**
* Copy $G^{(0)}$ (starting from the uncompacted template).
* Evaluate the `AffineSlot`s at bucket $b$'s concrete coordinate $\tau^{(b)}$ to update the varying `shape`s and `constantStaging` `int32_t` buffers.
* Recompute `strides` for updated classes/e-nodes (contiguous strides for non-view e-nodes via `calcContiguousStrides(shape)`, or view strides from the parent class) and call `egraph.rebuild()`.


* In `computeENodeInfos(bucket_states[b])` (`planner.hpp`, line 927), for each e-node `i >= full_state_enodes` with `kernelId != 0`, verify `KernelRegistry::get().getKernel(enode.getKernelId()).matches(...)` on the instantiated concrete input/output `TensorNode` shapes and strides; if a kernel's custom `match` function rejects concrete $\tau^{(b)}$ (e.g., due to vector alignment), set `info.cost = TGConstants::INF` so `pruneEGraph` removes it!


* Then run `computeENodeInfos` and `pruneEGraph` as usual.





### Verification

* Compare the pruned `EGraph` (`bucket_states[b]`) and `all_enode_infos[b]` produced by template instantiation against direct `saturateBucket` across all buckets for $N = 16$ and $N = 64$.
* Verify that the set of surviving e-classes, e-nodes, and costs in every bucket is **100% identical** and produces the exact same compiled plans.

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

In `tensor_graphs_cpp/core/plan/search_state.hpp`:

1. Add dense per-bucket vectors alongside the existing maps (keeping the maps or a lightweight wrapper if needed for API compatibility, or providing $O(1)$ inline accessors):
```cpp
std::vector<std::vector<VarId>> dense_selected_vars; // [b][cid.value] -> VarId (or kInvalidVarId)
std::vector<std::vector<VarId>> dense_start_vars;    // [b][cid.value] -> VarId
std::vector<std::vector<VarId>> dense_offset_vars;   // [b][cid.value] -> VarId
// Range of VarIds belonging to bucket b: [bucket_var_begin[b], bucket_var_end[b])
std::vector<VarId> bucket_var_begin;
std::vector<VarId> bucket_var_end;

```


2. Populate `dense_selected_vars[b]`, `dense_start_vars[b]`, `dense_offset_vars[b]`, `bucket_var_begin[b]`, and `bucket_var_end[b]` in `SearchState::addBucketVariables(uint32_t b)` (`search_state.hpp`, line 881).


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

In `tensor_graphs_cpp/core/plan/brancher.hpp` (`HeuristicBrancher::chooseBranch`, line 743):

1. Move `if (chooseCache(state, out_decision)) return true;` **ahead** of `chooseSelection(state, out_decision, true)`!


2. In `chooseCache` (`brancher.hpp`, line 513), if the root primal heuristic found an incumbent (`!incumbent_cached_nodes.empty()`), prefer the incumbent's cache value (`1` if in `incumbent_cached_nodes`, else `0`) as the left branch instead of always `0`.



#### 5.3. Sound Per-Bucket Upper-Bound Pruning (`OR(LB_b >= UB_b)`)

In `tensor_graphs_cpp/core/plan/search_state.hpp` and `search_engine.hpp`:

1. Add `std::vector<float> bucket_best_costs;` (initialized to `TGConstants::INF` per bucket) and `std::unordered_set<BaseEClassId> best_cost_cache_set;` to `SearchState`.


2. When `SearchEngine::recordIncumbent` (`search_engine.hpp`, line 640) records a new best plan (or when solving for a fixed cache set), record each bucket $b$'s makespan in `bucket_best_costs[b]` for that cache set.


3. Add a method `bool isBucketLowerBoundPruned(uint32_t b) const` on `SearchState`:
* When `allCachedVarsFixed()` is `true` and the current fixed cache set matches `best_cost_cache_set`:
* Bucket $b$ is completely independent of all other buckets $b' \neq b$.
* Furthermore, any bucket $b' \neq b$ whose variables in `[bucket_var_begin[b'], bucket_var_end[b'])` have not been modified since the root/cache-fix point already has a known achievable makespan `bucket_best_costs[b']`.
* If in the current bucket $b$ being branched, `bucket_lower_bounds[b] >= bucket_best_costs[b]` (and all other buckets are at their best or fixed), return `true` (prune immediately!).




* Also keep the global check `lower_bound >= best_cost` active at all times.




4. Call `state.isBucketLowerBoundPruned(b)` in `CriticalPathPropagator::propagate`, `EngineWorkloadPropagator::propagate`, `StartRunMakespanPropagator::propagate`, and `SearchEngine::runPropagators`.



#### 5.4. Parallel Per-Bucket Subproblem Solve Given Fixed `CACHED`

In `tensor_graphs_cpp/core/plan/search_engine.hpp`:

1. Add a helper `bool solveBucketSubproblem(uint32_t b, float timeout_seconds)` that solves only bucket $b$'s variables `[bucket_var_begin[b], bucket_var_end[b'])` when `allCachedVarsFixed()` holds:
* Construct a lightweight single-bucket `SearchState` slice (or run `SearchEngine` restricted to bucket `b` by setting `active_bucket_filter = b` on `HeuristicBrancher`), with `state.best_cost = bucket_best_costs[b]` (seeded by the root primal heuristic's validated makespan for bucket $b$).
* Because `bucket_relaxation_lower_bounds[b]` is already computed at the root, if `bucket_relaxation_lower_bounds[b] >= bucket_best_costs[b] - 1e-6f`, bucket $b$ is **already proven optimal at the root in 0 iterations**!
* For any bucket $b$ where `bucket_relaxation_lower_bounds[b] < bucket_best_costs[b]`, `ThreadPool::get().parallel_for` can run `solveBucketSubproblem(b)` in parallel across buckets!



### Verification

* Run all unit and integration tests for `SearchEngine` and `Planner::planAll`.
* Verify that the optimal makespan and extracted plans match baseline, and check the logs to confirm that buckets with `LB_b == UB_b` terminate immediately with 0 branch-and-bound nodes explored.

---

## Section 6: Cross-Bucket Plan Seeding (`SELECTED`, `START`, and Dynamic `OFFSET` Reconstruction)

### Background & Problem

In `buildRootPrimalPlans` and `improvePrimalPlan` (`core/plan/primal_heuristic.hpp`, lines 313, 680), every bucket $b \in \{0, \dots, 2N\}$ independently runs:

* `RelaxationBoundComputer::evaluate` + `regionEstimates`

* `extractSelection` / `extractCovering`

* `listSchedule`

* `improveBucketLocalSearch` (which loops over all active classes and tests alternative swaps using `listSchedule`)


* `SearchEngine::tryPlan` (which fixes variables and runs `runPropagators` + `preferredOffset`).



Adjacent or structurally isomorphic buckets (e.g. Decode $t-1$ and Decode $t$, or Prefill $t-1$ and Prefill $t$) have identical e-graph topology and nearly identical relative kernel speeds, differing only in tensor sizes (`OFFSET` footprints) and exact durations.

### Files to Modify

1. `tensor_graphs_cpp/core/plan/primal_heuristic.hpp`

2. `tensor_graphs_cpp/core/plan/search_engine.hpp`


### Step-by-Step Implementation

#### 6.1. Structural Lineage Signature for Cross-Bucket `SELECTED` Seeding

In `tensor_graphs_cpp/core/plan/primal_heuristic.hpp`:

1. Add a helper `bool transferBucketSelection(const SearchState &state, uint32_t src_b, const RelaxationBucket &src_rb, const PrimalBucketPlan &src_plan, uint32_t dst_b, const RelaxationBucket &dst_rb, const std::unordered_set<BaseEClassId> &allowed_cache, PrimalBucketPlan &dst_plan)`:
* Fast path (Template / Isomorphic buckets): If `src_rb.classes.size() == dst_rb.classes.size()` and for every $u$, `src_rb.classes[u].cid == dst_rb.classes[u].cid` and the chosen alternative `src_plan.value[u]` exists in `dst_rb.classes[u].alternatives` (with the same `KernelId` and `OpType`) and satisfies `!alt.requires_cache || allowed_cache.count(dst_rb.classes[u].base)`:
* Copy `dst_plan.value = src_plan.value` directly!


* General path (Different bucket geometries):
* Build a map from `(BaseEClassId, LogicalId, OpType, KernelId)` chosen in `src_b` to the corresponding alternative in `dst_rb`.
* Top-down from `dst_rb.root` (as in `extractSelection`), if a class $u$ in `dst_rb` has a matching feasible alternative from `src_b` whose children are all feasible, select it; otherwise fall back to `extractSelection`'s minimum-estimate choice for $u$.






2. Once `dst_plan.value` (`SELECTED`) is populated for bucket `dst_b`, call `listSchedule(dst_rb, dst_plan)` (`primal_heuristic.hpp`, line 79):


* `listSchedule` uses **bucket `dst_b`'s exact concrete durations (`alt.duration`) and engine masks** to compute the makespan-optimal `dst_plan.order` (`START` ranks) and exact simulated `dst_plan.makespan`!





#### 6.2. Seeding `improveBucketLocalSearch` Across Similar Buckets

In `tensor_graphs_cpp/core/plan/primal_heuristic.hpp` (`improvePrimalPlan`, line 680):

1. Partition weighted buckets into structural isomorphism groups (buckets where `rb.classes.size()` and per-class `(OpType, KernelId)` alternative lists match—which groups all Decode buckets in a regime together and all Prefill buckets in a regime together).
2. In `improvePrimalPlan`:
* First run `improveBucketLocalSearch` on **one representative bucket** $b_{\text{rep}}$ per group (or a small logarithmic sample of sequence lengths $t \in \{2, 4, 8, \dots\}$ for prefill).
* Seed the improved `plan.buckets[b_rep].value` into the other buckets $b$ of the group via `transferBucketSelection` + `listSchedule(contexts[b].rb, plan.buckets[b])`.


* Then run `improveBucketLocalSearch` on the remaining buckets $b$: because they start at the already-converged selection from $b_{\text{rep}}$, `improveBucketLocalSearch` either finishes in a single fast verification pass (0 swaps needed) or makes 1–2 boundary swaps where a kernel crossover occurred!



#### 6.3. Fast Concrete `OFFSET` Reconstruction in `SearchEngine::tryPlan`

In `tensor_graphs_cpp/core/plan/search_engine.hpp` (`SearchEngine::tryPlan`, line 663):

1. Notice how `tryPlan` currently validates a `PrimalPlan`:
* It fixes all `CACHED` vars, then for all buckets $b$ fixes `SELECTED` and `START`, then calls `runPropagators(kInvalidVarId)` once, and then loops over all unfixed `VarId v = 0 .. state.numVars()` calling `fixDomain(state, v, value)` and `runPropagators(v)`.




2. Optimize `tryPlan` using the bucket decomposition from Section 5:
* Fix all `CACHED` variables first and run `CachedOffsetAllocationPropagator` (which fixes all cached base offsets across all buckets in one step).


* For each bucket $b$ independently:
* Fix `SELECTED` and `START` for bucket $b$ from `plan.buckets[b]`.


* Run bucket-local propagation on bucket $b$.
* For any unfixed `OFFSET` variables in `[state.bucket_var_begin[b], state.bucket_var_end[b])` (when `state.private_layout_spaces` did not already fix them at the root), assign their concrete page offsets using `placement.preferredOffset(state, b, info.eclass_id, d)` (`brancher.hpp`, line 259)—which uses bucket $b$'s exact concrete page sizes (`eclass_page_sizes_[b]`) and exact buffer lifetimes `[start, end)`!




* Because each bucket $b$'s `SELECTED`, `START`, and `OFFSET` validation in `tryPlan` is completely independent once `CACHED` is fixed, this per-bucket validation can also run in parallel across buckets!



### Verification

* Run full compilation benchmarks at $N = 16$, $N = 64$, and larger sequence lengths.
* Verify that:
1. `tryPlan` accepts the seeded plans with 0 propagation conflicts.
2. The final compiled plans achieve the exact same (or better, within a fixed time budget) makespan as baseline.
3. Total compilation time scales sub-linearly with the number of buckets $2N+1$.



---

## Summary Checklist for the Implementing Agent

| Section | Primary Files | Key Invariant to Verify After Implementation |
| --- | --- | --- |
| **1. `SCATTER` Key Canonicalization** | `cost_model.hpp`, `kernel_bench.hpp`, `planner.hpp`, `rewrite.hpp` | `SCATTER` hits cost/fusion memos across all buckets; `discoverCalls` emits 1 record per `(update_shape, out_shape)`. |
| **2. Analytical `TRIU` + `NEG_INF` Mask** | `execute_ref_graph.hpp`, `bucket_regions.hpp`, `rewrite.hpp` | `foldCleanTensors` skips large `FILL`/`TRIU`; `computeRegionFacts` & `InfinityDomination` run in $O(1)$ without $N^2$ float scans; identical `BucketRegions`. |
| **3. Partial-Bucket Saturation Speedups** | `planner.hpp`, `rewrite.hpp` | `sweep_filter` active on Sweep 1 of partial buckets; single-pass clean-class queue; `EGraph` compacted after `pruneEGraph`. |
| **4. Parametric Template Saturation** | `planner.hpp`, `rewrite.hpp` | Isomorphic regime buckets instantiate concrete shapes/constants in $O(\vert{}E_{\text{new}}\vert{})$ and run exact `computeENodeInfos` + `pruneEGraph`. |
| **5. Flat `SearchState` & Cache-First Decomposition** | `search_state.hpp`, `brancher.hpp`, `search_engine.hpp` | Dense vector lookups; `chooseCache` runs first; sound per-bucket `LB_b >= UB_b` pruning when `CACHED` is fixed. |
| **6. Cross-Bucket Seeding & `OFFSET` Packing** | `primal_heuristic.hpp`, `search_engine.hpp` | `SELECTED` seeded across regime buckets, `START` recomputed via `listSchedule`, `OFFSET` packed for concrete sizes via `preferredOffset`. |