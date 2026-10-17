# Causal Mask, Region Analysis & E-Graph Cleanup Plan

Origin: `tests.test_gemma_output --config prefill+decode` fails with

```
EClass 3803 has 70 enodes, exceeding domain bitmask capacity (max 63).
```

This document is the implementation plan agreed on while analysing that failure. It
covers the root cause, the cleanups found along the way, and the redesign of how the
causal mask's sparsity is exploited. Each phase lists files, concrete steps, and how to
verify it. Phases are ordered so each one is independently verifiable; do not merge
phases into one change.

Out of scope here (tracked separately, revisit after Phase 5):
- `Expected constant for shape inference but not found in staging` errors.
- Cycle reduction leaving 60-120 cyclic components per bucket (needs the debug dump after
  passes 4 & 5 with `TG_CYCLE_DIAGNOSTICS`). Phase 4 is expected to reduce these, because
  the `replaceRegion` chains create `class -> view -> per-layer concat -> class` loops.

---

## 0. Background: why EClass 3803 has 70 enodes

Class 3803 (bucket 0) is a clean `[4,2,16]` F32 contiguous tensor: two rows of the
head-expanded causal mask `mask_expanded = repeat(mask, 4, 0)`
(`kernels/fused/gemma_attention/ref.hpp:19`). Its enodes are 69 `FastContiguous_ND`
enodes, each with a *different* child class, plus one `CACHE`.

Grouping by `DebugOrigin` gives 18 groups (one per Gemma layer) of
`rewrite.hpp:1769` (replaceRegion slice) + `rewrite.hpp:3319` (SlicePushDownStructural
`sliceOf`) entries, plus a tail of 3319/3689/1461 entries.

Mechanism (hypothesis, confirmed by Phase 2 logging):

1. The global and local masks are constants with equal value for `seq_len <= 512`, so
   constant hashing puts them in one class; every layer's `add(scores_i, mask_expanded)`
   shares it.
2. `InfinityDomination` (full saturation, no regions) computes the mask's non-`-inf`
   boxes with `coverWithFewRegions(..., kMaxRegions = 8)` and applies
   `replaceRegion` **sequentially** (`rewrite.hpp:2405`):
   ```
   currentTarget = M
   for reg in nonInfRegions:
       currentTarget = replaceRegion(currentTarget, add(slice(scores_i), slice(M)), reg)
   ```
   From the second region on, the base is the previous per-layer result `T_k_i`, which
   already contains `scores_i` updates.
3. Slices of `T_k_i` that only cover untouched mask cells are proven equal to the mask
   slice by `SlicePushDownStructural` (through the CONCAT), and merged into the mask-slice
   class. Each merge brings `CONTIGUOUS(SLICE(T_k_i, box))`, whose child is a per-layer
   **view** class. View classes are never merged, so congruence never collapses the
   parents.
4. Bucket egraphs are copy-on-write children of the saturated full state, so these chains
   exist in every bucket, including bucket 0.

The merges are sound; the problem is redundant derivations. The fix is to stop creating
them (Phase 4) and to let congruence collapse equal views (Phase 3).

Agreed principles that constrain every phase:
- **Fail loud** (AGENTS.md).
- **Never prune sound alternatives** unless provably dominated (memory:
  full-saturation-reference). Alternatives that compute a superset of the needed cells
  (dense, bounding box) are kept; they are choices, not analysis errors.
- **Region facts are never over-approximated.** dirty / needed / live / finite / support
  are exact; when a box cannot represent a set exactly, use a box list.
- **No symbolic shapes / parametric regions.** Too invasive.
- **No new atomic ops** (no `DOT_MASKED`). Specialised kernels are allowed only as fused
  kernels matched through a ref factory.
- Accepted assumption: model values are never `NaN` or `+inf`. This does **not** extend to
  uninitialised / stale buffer contents.

---

## Phase 1: Mechanical cleanups (no behaviour change intended)

### 1.1 Remove OR-Tools completely

Remove every OR-Tools mention except in `docs/reviews/`.

C++:
- Delete `tensor_graphs_cpp/core/plan/ortools_export.hpp`,
  `tensor_graphs_cpp/test_ortools_full.cpp`.
- `core/settings.hpp`: remove `use_ortools`, `use_ortools_full`, `use_ortools_lns`, their
  `--use-ortools*` flags and parsing.
- `core/session.hpp`: remove the OR-Tools state builder (the block that moves
  `bucket_states` into `bucket_egraphs`, adds CACHE enodes, prunes, and returns early
  when neither OR-Tools flag is set, around `session.hpp:735-775`) and everything only it
  uses (state struct fields, includes). This also removes the second, duplicate CACHE
  insertion path; `planner.hpp` `addCacheENodes` remains the only one.
- `bindings.cpp` (about 26 references), `main.cpp:102-110`, `bench_model.cpp:403`: remove
  OR-Tools / LNS options and cache-file prefixes.
- Build scripts (`build.py`, CMake if present): drop any OR-Tools target/sources.

Python and repo:
- Delete `ortools_full.py`, `ortools_lns.py`, `ortools_hints.py`, `ortools_cp_model.py`,
  `test_ortools_full.py`, `test_ortools_lns.py`, `lns_selectors/gnn/` (whole directory,
  and `lns_selectors/` if it becomes empty), `test_gnn_generate_samples.py`.
- `main.py` (`--use-ortools*` args, lines ~122-226) and `txt2img.py` (~124-216): remove
  args and plumbing.
- `pyproject.toml`: remove the `ortools` extra; regenerate `uv.lock`.
- `AGENTS.md`: remove the `.venvx64` / OR-Tools sentence.
- `docs/agents/targets/gap_integral/target.md`, `docs/agents/targets/search_volume/progress.md`: remove or
  rewrite the OR-Tools passages.
- `thinking.txt:4` mentions the OR-Tools CP model as inspiration; confirm with the user
  whether to edit it (it is a notes file).

Verify: `rg -i ortools -g '!docs/reviews/**'` returns nothing; project builds; `test.cpp`
suite passes.

### 1.2 `mergeRestricted` -> `egraph.merge`

`EGraph::merge` (`egraph.hpp:707`) already throws `[ILLEGAL EClass MERGE]` on
shape/dtype/strides/mem-space mismatch, so `mergeRestricted` (`rewrite.hpp:261`) only adds
a *silent skip* on stride/mem-space mismatch.

- Replace the 4 call sites (`rewrite.hpp:2849, 3158, 3692, 4064`) with `egraph.merge`.
- Delete `mergeRestricted`.
- If a call site now throws, that is a real bug the skip was hiding: investigate, do not
  reintroduce a skip.

### 1.3 Always throw on merging different constants

`EGraph::merge` prints `[CRITICAL CONSTANT MERGE]` and only throws under
`TG_THROW_ON_BAD_MERGE`. Remove the macro; always `Error::throw_err` with both class ids
and their shapes.

### 1.4 `pruneEGraph` fail loud and honest logging

`planner.hpp` `pruneEGraph`:
- `enodeId.value >= enodeInfos.size()` must throw instead of silently dropping the enode.
- The pass only removes infinite-cost enodes plus the death cascade; change the log from
  "dominated enodes" to "infinite-cost enodes (+cascade)".

### 1.5 Drop the redundant CACHE mem-space check

`planner.hpp` `addCacheENodes` (around line 3031): a class has a single mem space, so
`getMemSpace() == cls.mem_space` is always true. Check only `OpType::CACHE`.

CACHE policy stays as is: every bucket with weight > 0 except the full bucket gets CACHE
enodes (prefill and cached prefill benefit, e.g. weight loads); the full bucket
initialises everything.

### 1.6 `ScatterPullUp::isUpdated` depth limit

`rewrite.hpp:2470-2495` stops after 4 view ops (`depth`). Either walk the full view chain
(views are acyclic within one class path; guard with a visited set) or throw when the
limit is hit. A silent miss loses scatter forms.

Verify Phase 1: build, `test.cpp` suite, `tests/test_gemma_output.py --config decode`
(should behave exactly as before), `tests/verify_saturation.py` against existing gold
(should still match exactly: none of these changes alter saturation).

---

## Phase 2: Diagnostics (confirm the mechanism before changing saturation)

All behind environment variables and off by default.

1. **Class derivation dump** `TG_DUMP_CLASS=<bucket>:<class id>` (or a base id): after
   pruning, print each enode of the class and recurse 3 levels into children: op, shape,
   strides, `DebugOrigin`, `logical_id`, `base_eclass_id`. Expected for 3803:
   `CONTIGUOUS -> SLICE(view) -> CONCAT(..., update of layer i)`.
2. **Merge trace** `TG_TRACE_BASE_ID=<base id>`: in `EGraph::merge`, when either side's
   base id (or `logical_id`) matches, log the caller `SourceLocation`, the active rule
   name, and the other class's first enode. Check first that the chosen id is stable
   across runs (class 3803 is a post-compaction id).
3. **InfinityDomination trace** `TG_TRACE_INF_DOM`: per application, the add's
   `logical_id`, `nonInfRegions` count and boxes, and the number of slices
   `replaceRegion` created.
4. **Class-size histogram** `TG_CLASS_SIZE_HISTOGRAM`: per bucket, after saturation and
   after pruning: counts of classes with >16, >32, >63 enodes, and the top 20 classes with
   an (op, `DebugOrigin`) breakdown. Also used to verify Phases 3-4.

Verify: run `prefill+decode` with 1, 2, 4 enabled; record findings in this doc (append a
"Findings" section). If the mechanism differs from Section 0, stop and revise Phases 3-4.

---

## Phase 3: View-level merging (injectivity of CONTIGUOUS / COPY_TO)

Problem: class M contains `CONTIGUOUS(V_1) ... CONTIGUOUS(V_n)` where every `V_i` is a view
class with identical shape, strides, dtype and mem space. `CONTIGUOUS` is the identity on
values, so `CONTIGUOUS(V_i) == CONTIGUOUS(V_j)` implies `value(V_i) == value(V_j)`; with
identical layouts, `V_i` and `V_j` can share one class. After merging them, the parents are
identical and `rebuild` deduplicates them.

```
before                                      after
M ├─ CONTIGUOUS(V_M)   V_M = SLICE(M0, b)   M ├─ CONTIGUOUS(V)
  ├─ CONTIGUOUS(V_1)   V_1 = SLICE(T_1, b)    └─ CACHE
  ├─ CONTIGUOUS(V_2)   V_2 = SLICE(T_2, b)  V ├─ SLICE(M0, b)
  └─ ...                                      ├─ SLICE(T_1, b)
                                              └─ SLICE(T_2, b) ...
```

The view class V still holds n enodes, so this alone does not bound class size; it
removes duplication in every *consumer* of M. Phase 4 removes the per-layer views
themselves.

Implementation:
- New rule `ViewInjectivity` in `rewrite.hpp` (registered in the planner rule list next to
  `RemoveContiguous`). `match`: a class with at least two `CONTIGUOUS` enodes. `apply`:
  group those enodes by the child class's `(shape, strides, dtype, mem_space)`; within a
  group, `egraph.merge` all child classes.
- Same for `COPY_TO` (injective for a fixed source layout and mem space): group by the
  child layout and source mem space.
- Base ids: these merges are legal (equal values, identical layouts) and use the
  existing mechanism, with no new code paths:
  - Base ids are assigned only after full saturation (`populateBaseEClassIds`,
    `planner.hpp:2458`), so merges during full saturation never involve two base ids.
  - Bucket saturation runs with `allow_base_equalities = true` (`planner.hpp:2002`).
    Merging two base classes records an alias in `base_equalities`, and the class keeps
    the lower id.
  - `reconcileBaseEqualities` (`planner.hpp:2893`, `plan/base_equalities.hpp`) runs
    before cache decisions. It applies every alias to the full state and **all** buckets,
    rewrites `scatter-into:b<id>` tags, and throws on incompatible layouts.
    `findEClassByBaseId` resolves aliases.

  What to watch:
  - Two persistent values become one cache candidate / buffer. That is the intended
    effect.
  - An equality found in one bucket is merged into every other bucket. Rebuild can
    cascade there and grow classes **after** each bucket's own saturation, so the
    class-size histogram must also run after reconciliation.
  - Reconciliation repeats a full `rebuild` per graph until no new equality appears. More
    equalities mean more rounds × buckets, so log the round count and time.
  - Cycles can appear in buckets that never derived the equality themselves. Compare
    cycle-reduction stats before and after.
- Watermark / sweep-filter integration like the other rules (only look at classes that
  gained a `CONTIGUOUS`/`COPY_TO` enode since the base watermark).

Verify: unit test in `tensor_graphs_cpp/tests/` that builds two different views with equal
values and layouts under one contiguous class and checks that, after saturation, the views
share a class and the contiguous class has one CONTIGUOUS enode. Class-size histogram
before and after on `prefill+decode`.

---

## Phase 4: Replace InfinityDomination with exact region analysis

### 4.1 What already exists

No new rewrite rule is needed. Two existing pieces already do the "absorbing element"
work:

- **Analysis: `computeBucketRegions`** (`plan/bucket_regions.hpp:922`) runs on the logical
  `Graph` once per bucket before saturation (`planner.hpp:2163`). It computes
  `dirty` (forward), `needed` (backward from `bucket.outputNeededRegion`, plus dirty cells
  that must stay valid for caching) and `live = dirty & needed`, using per-op
  `forwardRegion` / `backwardRegion` from `core/ops/*.hpp` `OpTraits` through
  `ShapePropagator`. `RegionFacts` (graph-wide) supplies `masks`, `finite`, `support` and
  analytic `triuMasks` / `triuFinite` / `triuSupport`, which the backward pass already
  applies:
  - `add(x, mask)`: masked cells of `x` are not needed.
  - `SUM` needs its input only on its support; `MAX` only on its finite range.
  - `DOT` with A's support known: only B rows for A's nonzero columns
    (`dotNeededB` / `dotNeededBTriu`); A's zero columns are not needed.
- **Consumption in saturation**: `ScatterPullUp` exposes
  `CONTIGUOUS(SLICE(class, box))` for classes with a partially updated child, the
  `SlicePushDown*` rules restrict ops to that box and add SCATTER forms when
  `scatterFormValid` holds.

For decode bucket `i` this already gives single exact boxes:

```
probs . v     triuSupport on probs  -> v rows 0:i+1, probs [:, i, 0:i+1]
sum / max     triuSupport/Finite    -> input [:, i, 0:i+1]
add(s, M)     mask                  -> scores [:, i, 0:i+1]
dot(q, k^T)   per-op backward       -> q row i, k^T cols 0:i+1 -> k rows 0:i+1
cache         dirty k/v = row i     -> live k/v = row i
```

So `InfinityDomination`'s bucket path (cached-update SCATTER of the masked add) duplicates
`ScatterPullUp + SlicePushDownElementwise`, and its full-saturation path (`replaceRegion`)
only benefits the full bucket, which has weight 0 and should just do the full computation.

### 4.2 Verify redundancy before deleting

Add a temporary setting/env (`TG_DISABLE_INF_DOM`) that skips registering
`InfinityDomination` (`planner.hpp:471`). Run `decode` and `prefill+decode` with and
without it, with `TG_DEBUG_MISSING=1 TG_DEBUG_REGIONS_ALL=1`:
- live boxes of `scores` / `k` / `v` / `probs` match the table above;
- `[debug missing scatter]` counts do not get worse;
- final planned cost per bucket does not get worse;
- generated tokens match.

If something regresses, find the specific pattern InfinityDomination handled and extend
the generic analysis or push-down for it. Do not keep InfinityDomination.

### 4.3 Make region facts exact

1. `boundedIntersect` (`bucket_regions.hpp:401-427`, both overloads) returns the bounding
   box of each box's intersection with a staircase ("A superset of the exact
   intersection"). Replace it with exact intersection as a box list (for `TriuBound`,
   one box per row segment; implement it analytically, without scanning rows of dense
   data). The analysis cost is O(rows) arithmetic on the logical graph, negligible next to
   saturation. Audit every caller, including the cache-validity `keep` computation.
2. Reshape `backwardRegion` fallback (`ops/reshape.hpp:181-189`) uses
   `unravelFlatBounds(flat_start, flat_stop)`, a bounding box. Replace it with an exact
   decomposition: split the output box into maximal runs that are contiguous in flat
   index space, and unravel each flat range into the minimal exact box list (a
   contiguous flat range unravels into at most `2*rank-1` boxes). Do the same audit for
   reshape `forwardRegion` and the other ops' region functions (`rg "unravelFlatBounds|
   boundingBox" tensor_graphs_cpp/core`), and for the RESHAPE case in
   `SlicePushDownStructural::buildRestrictedImpl` (it currently returns no restriction
   when axes do not line up; that is safe, but an exact box list would allow more).
3. `MAX` with a fully masked row keeps the whole box: this is required (softmax over a
   fully masked row depends on all its masked values). Keep it, and add a comment saying
   why.
4. The mask "sentinel approximation" (`bucket_regions.hpp:37-40`, NEG_INF is a finite
   `-1e9`-style value) is covered by the accepted no-NaN/+inf assumption; update the
   comment to say so.

### 4.4 Materialisation choice in saturation

`ScatterPullUp::regionFor` (`rewrite.hpp:2497-2507`) and `scatterFormValid` take
`boundingBox(liveOf(id))`. With exact facts:
- decode: live is one box, so nothing changes;
- prefill / cached prefill: live `scores` is a staircase (P boxes). Keep materialising
  the **bounding box** as the (single) slice alternative. It is a non-dominated
  alternative (it computes extra cells), not an analysis error. Exact per-row slices
  would cost O(P) e-graph growth; prefill sparsity is served by the fused causal kernel
  (Phase 5) instead.

Initialised-data argument for the bounding-box choice (no runtime check needed): the
extra cells only read inputs through slices that the push-down rules derive from the
materialised box itself, so every input cell they read is either
- computed in this run (the push-down restricts the producer to the box it is asked
  for), or
- a persistent cached buffer. `Session` runs the full bucket before any partial bucket
  that reads cached nodes (`session.hpp:584`), so these are initialised. Cells that are
  stale (for example padding regions not kept valid) still hold values an earlier run
  computed, which are finite under the accepted assumption.

For the causal staircase the bounding box only reaches key columns up to the last live
query row, which are live or cached anyway. Cheap insurance for verification: a debug
switch that fills fresh (never written) executor allocations with NaN; run the Gemma
tests with it on.

### 4.5 Delete InfinityDomination

Delete from `rewrite.hpp`: `InfinityDomination` (from line 1786), `replaceRegion` (1747),
`coverWithFewRegions`, `kMaxRegions`, `unmasked_memo`, `mapTriuBoundForwardENode` and the
other helpers only it uses. Remove the registration at `planner.hpp:471` and the
`TG_DISABLE_INF_DOM` switch.

Side effect to handle: `InfinityDomination::isConstantFloat` (`rewrite.hpp:2121`) loads
constant data from the repo into `egraph.constantStaging` during matching.
`tests/constant_view_regression.hpp:208` reproduces that state. Find everything that
relied on this staging (`rg constantStaging tensor_graphs_cpp/core`), and if it is still
needed, load it explicitly in one place (base egraph initialisation) instead of as a rule
side effect. Update or delete:
- `tests/triu_mask_predicate.hpp:300-360` (InfinityDomination box tests): delete the
  InfinityDomination part, keep the `CoordinatePredicate` / `matchTriuMask` parts;
- `tests/saturation.hpp:107` and `tests/constant_view_regression.hpp:208` comments/setup;
- `core/misc.hpp:374` TODO that mentions InfinityDomination.

Contingency: if 4.2 shows InfinityDomination is needed and that cannot be replaced
generically, fix `replaceRegion` instead of deleting it:
- build the result from the **original** base in a single pass over all regions (no
  slicing of a previous per-layer intermediate);
- skip full-extent slices (as `SlicePushDownStructural::sliceOf` does);
- recurse into slices of the base, not of `CONTIGUOUS(SLICE(base))`;
- no `kMaxRegions` cap: regions are exact (one per row segment).

### 4.6 Verify Phase 4

- `prefill+decode` no longer throws the 63-enode error; class-size histogram shows no
  reachable class above 63 (if one remains, dump it with Phase 2 tooling before doing
  anything else);
- `decode` and `prefill+decode` generate the expected text;
- planned cost per bucket is not worse than before Phase 4 (record numbers here);
- saturation time recorded (expected to drop);
- `tests/verify_saturation.py`: saturation changes on purpose. Regenerate gold
  (`--generate-gold`) after the checks above pass (approved), and note it in
  `docs/agents/targets/saturation_seq_len/progress.md`. The same applies after Phase 3;
- cycle-reduction stats before/after (expected fewer remaining cyclic components).

---

## Phase 5: Causal attention fused kernel (prefill sparsity)

Goal: prefill / cached prefill skip the masked upper triangle without O(P) e-graph growth,
symbolic regions, or predicate-aware atomics. Backends without the kernel fall back to the
dense form (accepted tradeoff; decode remains general through Phase 4).

### 5.1 Preconditions to verify first

1. The mask's construction enodes (`triu(fill(1), k)`, the `mul` by the NEG_INF fill, the
   reshape / repeat to heads) survive in the mask class after it gets constant-staged.
   Staging adds data to the class but does not remove enodes; confirm on the real graph
   with `TG_DUMP_CLASS`.
2. How `FusionRule::matchPatternClass` treats a class that has staged constant data
   (`isConstant(...)` / `childIgnoreConst` path, `rewrite.hpp:~1760`): it must still be
   able to match through the construction enodes, not only as an opaque constant leaf. If
   it cannot, extend matching so pattern nodes that are ops can match a staged class's
   op enodes.
3. The global and local masks share one class (equal values for seq <= 512). The ref
   factory targets the plain causal construction; the class contains it.

### 5.2 Ref factory and kernels

- New fused kernel directory, for example `kernels/fused/causal_attention/` with
  `ref.hpp` exposing `refFactoryCausalAttention(q, k, v, diag_k)`: it builds the mask
  internally exactly as `compute_attention_mask(0)` does (`fill`, `triu` with constant
  `diag_k`, NEG_INF fill, `mul`, reshape, `repeat` over heads) followed by the
  `refFactoryGemmaAttentionEx` body. `diag_k` is a constant input, so cached prefill
  (shifted diagonal) is covered by the same kernel.
- Implementations: at least a CPU one (Neon / AVX2 per AGENTS.md: capability guards wrap
  the whole file, threaded and unthreaded are separate kernels, no branching inside run
  functions), and CUDA / OpenCL as available. The kernel skips column blocks above the
  diagonal (FlashAttention-style causal tiling).
- Optionally a smaller `refFactoryCausalDot` (masked scores only) if it is useful for
  models whose attention does not match the full factory.
- Calibrate cost records (`benchmarks/records.bin`) for the new kernels; missing records
  give infinite planner cost.

### 5.3 Verify Phase 5

- Unit test: fused kernel output equals the reference factory for several `(P, diag_k)`,
  including `diag_k > 0` (cached prefill).
- `prefill+decode`: the planner selects the causal kernel in prefill buckets on the
  backends that have it; generated text matches; planned and measured prefill cost
  improves.

---

## Phase 6: Follow-ups (not part of this plan's acceptance)

- Investigate `Expected constant for shape inference but not found in staging`
  (`ops/concat.hpp:14`, `sum.hpp:14`, `max.hpp:14`, `scatter.hpp:14`). Suspect: constant
  staging not moved or rebuilt correctly after merges (`rebuildConstantHashIndex` skips
  non-canonical entries "data was moved during merge").
- Cycle reduction: enable `TG_CYCLE_DIAGNOSTICS`, add dumps after passes 4 and 5, rerun
  `prefill+decode`.
- Future direction: a single generic slice push-down / pull-up rule driven by the per-op
  `forwardRegion` / `backwardRegion` plus a per-op `restrict` constructor (move
  `SlicePushDownStructural::buildRestrictedImpl`'s per-op switch into `core/ops`), with
  fused ops composing the relation through their pattern graph (`FusedOp::backwardRegion`
  currently throws).
- `CoordinatePredicate` remains the representation for value facts; it is not used for
  materialisation.

---

## Decision log

| Decision | Reason |
| :--- | :--- |
| No parametric / symbolic regions | Would touch shapes, kernel matching, costs, memory planning everywhere. |
| No `DOT_MASKED` atomic; causal sparsity in prefill via ref-factory fused kernel | Kernels stay generic to the framework; matching stays pattern based. |
| Decode sparsity via exact region analysis, not a fused kernel | Works on every backend; the main decode win (reading only `k`/`v` rows `0:i+1`) applies to decomposed ops. |
| Facts exact; superset computations kept as alternatives | Never over-approximate analysis; never prune non-dominated alternatives. |
| No region-count caps (`kMaxRegions`) | Caps force unnecessary computation. |
| Full bucket does dense computation | Weight 0; it only initialises state. |
| CACHE on every weighted bucket except full | Prefill and cached prefill can reuse cached values such as weight loads. |
| No NaN / +inf in model values | Accepted assumption; does not cover uninitialised buffers. |
| Remove OR-Tools | Unused; a future version would export the initialised `SearchState` instead. |
| ViewInjectivity rejected (Phase 3) | Unsound under the bucket validity model; see Implementation status. |

---

## Implementation status (2026-10-17)

### Phase 1: done
- OR-Tools removed from C++ (`ortools_export.hpp`, session OR-Tools state and the duplicate
  CACHE insertion path, settings flags, bindings arguments, `main.cpp`, `bench_model.cpp`),
  Python (`ortools_*.py`, `test_ortools_*.py`, `lns_selectors/`, `test_gnn_generate_samples.py`,
  `main.py` / `txt2img.py` arguments), `pyproject.toml` / `uv.lock`, `AGENTS.md`, and
  `tensor_graphs_cpp/todo.txt`. `max_time_seconds` (OR-Tools only) was removed; `cpu_only`
  is a generic setting and was kept. Left in place: historical experiment logs in
  `docs/agents/*` and `docs/core/relaxation_bound.md` that cite offline CP-SAT results, and
  `thinking.txt` (notes).
- `mergeRestricted` replaced by `egraph.merge`; the different-constant merge always throws;
  `pruneEGraph` throws on enodes without an `ENodeInfo` and logs "infinite-cost enodes";
  redundant CACHE mem-space check removed; `ScatterPullUp::isUpdated` walks the whole view
  chain (visited set) instead of stopping after 4 views.
- `verify_saturation` (seq 9) was still EQUIVALENT after Phase 1.

### Phase 2: done
Environment switches: `TG_CLASS_SIZE_HISTOGRAM` (per bucket after saturation, after base
reconciliation, after pruning), `TG_DUMP_CLASS=<bucket>:<class>|<bucket>:b<base>` (derivation
dump, ids as SearchState reports them), `TG_TRACE_MERGE_SHAPE=4,2,16` / `TG_TRACE_BASE_ID=<id>`
(merge trace with the applying rule), cycle listings after passes 4 and 5
(`egraph_cycles4.txt`, `egraph_cycles5.txt` with `TG_CYCLE_DIAGNOSTICS`), and a log line with
the alias count, rounds and time of `reconcileBaseEqualities`. (`TG_TRACE_INF_DOM` existed
until InfinityDomination was deleted.) The rule-disable switch for 4.2 already existed:
`TG_DISABLE_RULES=InfinityDomination`.

Findings (decode config, before Phase 4), confirming Section 0:
- InfinityDomination applied 18 times in full saturation (one per layer), each with 8
  two-row boxes `(0-4, 2r:2r+2, 0:2r+2)` (cover cells that are masked too), adding ~120-180
  enodes each.
- Class 3802 (bucket 0) = `CONTIGUOUS(SLICE(T_i))` alternatives, `T_i` a per-layer
  `Smart_Concat` produced by `replaceRegion` that already holds the layer's update.
- After pruning, bucket 0 still had 23 classes above 63 enodes (largest 142), all mask pieces.

### Phase 3: implemented, found unsound, removed
`ViewInjectivity` (merge equal-layout children of CONTIGUOUS / COPY_TO enodes of one class)
broke `--bucket-independence` (wrong masked output), also when restricted to full
saturation. Bisection pointed at a merge that put `SLICE(x_j, box)` alternatives into a class
whose consumers use it in partial buckets. The equality is true for complete values, but the
bucket planner only keeps each logical node valid on its own needed region: a child with a
SCATTER form is stale outside it. Existing rewrites only create alternatives that read child
cells their logical consumers need; injectivity creates alternatives reading cells nobody
asked for, so a bucket can read stale data. Making it sound would need per-alternative
validity reasoning in buckets. Phase 4 removes the motivating duplication at its source, so
the rule and its test were removed.

### Phase 4: done
- `InfinityDomination`, `replaceRegion`, `coverWithFewRegions`, `kMaxRegions` deleted, with
  their tests (`triu_mask_predicate` sections 3-4) and comments.
- Exact region facts: `intersectBoxWithTriuBound` / `intersectTriu` return one box per row
  segment cut by the diagonal plus one box for whole rows; support/finite list intersections
  use exact `intersect`; reshape forward/backward regions fall back to `reshapeRegionExact`
  (flat-contiguous runs unravelled exactly) instead of a bounding box. The materialised
  slice stays the bounding box of `live` (4.4).
- Decode (InfinityDomination disabled, before exact facts): no class above 16 enodes after
  pruning, ~42k classes per decode bucket (was ~196k), saturation done at 17.9 s (was ~79 s),
  generated text correct, plan cost 111.31 (no earlier baseline: it used to fail).
- `verify_saturation` seq 9: family vertices 2028 -> 680, 181 ms -> 76 ms; gold regenerated.
- Fixture `--bucket-independence`: it only "exercised persistent updates" through cached
  InfinityDomination pieces. Two pre-existing problems hid this:
  `Session(Settings)` ignored `records_path` (always loaded `benchmarks/records.bin`), and
  `plan()` reloads records, which dropped dummy records added to the session's cost model,
  so every SCATTER cost infinity. Fixed: the session honours `records_path`; the fixture
  writes dummy records (at each kernel's registered dummy shapes) to a file, uses 16x16
  inputs, and searches to the end (min_compile_seconds = -1, ~20 s per partial case). It
  now selects scatter-based partial plans (8.31 vs 9.01 modeled cost) with correct outputs.

### Phase 5: both options implemented (the planner chooses)
The design in Section 5 (mask built inside a constant-`diag_k` factory) could not be
restricted to bucket rows: `SlicePushDownFused` only restricted kernels whose row structure
it derives from the pattern (`fusedRowLocality`), and a baked-in diagonal is wrong after
slicing. Both alternatives discussed afterwards are implemented and coexist:

- `Masked_Attention_F32(_Threaded)` (`kernels/fused/masked_attention/`): additive mask
  input. `refFactoryMaskedAttention` generalises `refFactoryGemmaAttentionEx` to
  `Sq != Sk`. Keys whose mask value is `NEG_INF` are skipped (never read); a row with no
  unmasked key follows the reference densely.
- `Causal_Attention_F32(_Threaded)` (`kernels/fused/causal_attention/`): `diag_k` input;
  the factory builds the model's causal mask from it, written as `triu(CONTIGUOUS(FILL))`
  because the planner makes TRIU's zero-stride input contiguous (different classes, so
  `triu(FILL)` never matched). The kernel computes the visible key range and never reads
  or stores the mask.
- Explicit row restrictions (`core/fused_row_restriction.hpp`, registered per kernel): output
  axis, which inputs are sliced along it, and an INT32 offset input that gets the row
  start added. `SlicePushDownFused` uses them before falling back to derived row locality.
- `SlicePushDownFused` no longer stops after the first fused candidate (the Step 12 "early
  restriction break"): every fused alternative gets its restricted form, otherwise only one
  kernel per class could be chosen in partial buckets. verify_saturation time was unchanged.
- The registry lives outside `core/kernels.hpp` on purpose: build.py seeds every kernel uid
  from `core/kernels.hpp` and `core/types.hpp`, so editing them invalidates all cost records
  (seen once: only the 72 new records loaded and search failed).
- Tests: `test.exe --attention-kernels` (dense scalar reference with causal masks for
  several `diag_k`, fully masked rows, `Sq < Sk`, the row-restriction identity, an arbitrary
  mask, and that saturating the model's attention discovers both kernels), plus the generic
  fused kernel tests.
- Calibration: run the model with `--saturate-only` (logs cost calls), then
  `bench.exe _Attention_F32`. Limitation: the masked kernel is benchmarked on random mask
  data (the mask class is not a staged constant), so its records are dense costs and the
  cost model cannot see its skipping; the causal kernel's records use the real `diag_k`.
