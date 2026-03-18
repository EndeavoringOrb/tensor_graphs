# E-Graph Planner Migration Plan

Scope: tensor_graphs_cpp only. Replace planner + kernel selection with custom e-graph implementation.
Do not commit anything, only edit files. Keep editing until the solution is fully fleshed out according to this plan. Do not ask the user for input, do not run special commands outside of (1) reading/writing/searching files (2) python build.py, python build.py --debug, python build.py --cuda --debug (3) tensor_graphs_cpp/main (4) tensor_graphs_cpp/bench (5) gdb tensor_graphs_cpp/main (6) gdb tensor_graphs_cpp/bench, the user will only be back to supervise in the morning, you have the whole night to work. Just keep going until it is finished, don't stop for anything.
Do NOT edit this PLAN.md file.
No placeholder code. No simplified code.

## Goals
- Use e-graph saturation to explore rewrites, fusions, adapters, and kernel variants.
- Model kernels as enodes (explicit kernel choice).
- Keep dirty-bucket behavior via graph structure: (SLICE, CONTIGUOUS) before, (CONCAT) after.
- Produce per-node (full kernel, cached kernels list) in a single CompiledGraph.

## Plan (Locked)

### 1) Add a Custom E-Graph Core
- New header: `tensor_graphs_cpp/core/egraph.hpp` (don't separate into .hpp and .cpp, put implementation in .hpp).
- Data structures:
  - `ENodeKey`: kernel UID + child eclass IDs + any required attributes for shape/contiguity checks.
  - `EClass`: list of enodes, analysis data (shape, dtype, contiguity flags, backend availability, refcount).
  - Union-find for eclass merging; hash-consing map for enode canonicalization.
- Uniform `Rule` interface (pattern + predicate + apply). No typed rule APIs.

### 2) Kernel Metadata and Matching
- Expose/organize kernel metadata from `KernelRegistry` for rule predicates.
- Keep kernel match as the single legality gate, including inplace legality (refcount-aware), contiguity requirements, and backend constraints.
- Kernel match functions used during saturation to allow/deny rule applications.

### 3) Initialization (Reference Kernels Only)
- Input Graph contains atomic nodes only (no OpType::FUSED).
- For each node, initialize its eclass with exactly one REGISTER_REF_KERNEL that matches its backend and requires no rewrites.
- All other kernels (alternate backends, fused kernels, etc.) must be reached via rewrites.

### 4) Dirty-Bucket Prepass (Graph Rewriting Before E-Graph)
- Replace partial-kernel slicing in executor with explicit graph structure:
  - Insert `(SLICE, CONTIGUOUS)` on each input for dirty regions.
  - Use `CONCAT` to merge cached slices + computed dirty slices.
- Dirty cached graph shape:
  - `full_op(a,b)` becomes `concat(slice(cached_full, non_dirty_left), partial_op(contiguous(slice(a, dirty)), contiguous(slice(b, dirty))), slice(cached_full, non_dirty_right))`.
- Cached path produces multiple regions (list) per node; plan for `cached_kernels` list in compiled output.

### 5) Rewrite Rules (Uniform Interface)
- Algebraic rewrites from `core/rewrite.hpp` applied only to atomic op kernels (not OpType::FUSED).
- Fusion rewrites built from `ReferenceGraphRegistry` patterns; pattern matching must be shape-aware (API adjusted as needed).
- Adapter/backend rewrites:
  - Copy-sandwich insertion: `cpu_k -> copy_to_gpu, gpu_k, copy_to_cpu` when a GPU kernel exists.
  - Copy pair elimination: `copy_to_cpu(copy_to_gpu(x)) -> x` and vice-versa to keep chains on a backend.
- Contiguity rewrite: if analysis marks eclass as non-contiguous and no CONTIGUOUS successor exists, insert CONTIGUOUS alternative.
- Rule predicates gate CUDA availability and kernel matching.

### 6) Saturation
- Worklist-based saturation; apply rules until fixpoint or resource cap.
- Use eclass analysis to maintain shape/contiguity/backend flags.
- Shape-aware matching for fusion rewrites; contiguity tracked to enable CONTIGUOUS insertion.

### 7) Extraction (Single Pass)
- Bottom-up DP to select minimal-cost enode per eclass using `CostModel.estimateCost`.
- Legality enforced via kernel match predicates during extraction (invalid enodes skipped).
- Produce per-node:
  - `full_kernel`: single kernel ID for full compute.
  - `cached_kernels`: list aligned to dirty regions (multiple kernels).

### 8) Compilation and Execution
- Update `CompiledGraph` / `OpInstruction` in `core/types.hpp` to store full + cached kernels.
- Update `Executor` in `core/executor.hpp` to:
  - Use cached kernels when outputs are cached and dirty regions exist.
  - Use full kernel otherwise.
- Remove old partial-kernel handling and adapter insertion paths from planner.

### 9) Cleanup and Integration
- Remove/retire: BeamStrategy, AdapterChain, edgeCost, augmented topo sort, and adapter insertion in planner.
- Update `Session` (`core/session.hpp`) to call the new planner and prepass, keeping cache mechanics intact.

### 10) Verification
- Build, Run, if error then run with gdb -> find error -> fix -> repeat
- Check output tokens are `236888, 564, 236789, 236757, 9775, 531` for tensor_graphs_cpp/main with input tokens `2, 9259`. Make sure this works for `python build.py` and `python build.py --cuda`. For both cpu and gpu builds, make sure it works with nothing benchmarked, and all kernels benchmarked.
- Nothing benchmarked:
  1. `rm benchmarks/*`
  2. `tensor_graphs_cpp/main`
- Everything benchmarked:
  1. `tensor_graphs_cpp/main`
  2. `tensor_graphs_cpp/bench`
  3. Repeat running main then bench until bench doesn't bench any new kernels and says there are no kernels left to bench.
