# Current Progress & Remaining Tasks: Static Memory & Compiled Execution

This document tracks the remaining work required to fulfill the objectives in `TODO.md` based on the current state of the `feature/static-execution` branch.

## 1. Optimize `StaticExecutor` (Phase 5 Refinement)
The current `StaticExecutor` implementation performs dictionary lookups and creates tensor views inside the `run` loop, which introduces Python overhead.
- [x] **Pre-calculate Views:** Move view creation (`_get_view`) for all instructions into the `__init__` or a preparation phase.
- [x] **Flatten Instruction Loop:** Ensure the `run` loop only iterates over pre-prepared tuples of `(kernel, input_views, output_view, attrs)`.

## 2. Evolve Kernel API (Zero-Copy Goal)
Currently, kernels return new arrays/tensors, forcing `StaticExecutor` to perform a copy (`np.copyto` or `out.copy_`) to the pre-allocated buffer.
- [x] **Update `KernelRegistry`:** Make kernels that accept an `out` parameter.
- [x] **Refactor ALL Kernels:** Update kernels to write directly into the provided `out` buffer.
- [ ] **Conditional Execution:** Update `StaticExecutor` to pass the `out_view` to kernels.

## 3. Improve Constant & Weight Handling
- [ ] **Automate Weight Loading:** Refactor `Compiler` or `StaticExecutor` to automatically identify all `PERSISTENT` nodes and facilitate their initialization from a state dict.
- [ ] **Constant Embedding:** Ensure `OpType.CONSTANT` values are correctly placed into the persistent memory pool during compilation so they don't need to be passed in the `feed_dict` every time.

## 4. Robustness & Correctness
- [ ] **Alignment Verification:** Ensure the 256-byte alignment in `MemoryPlanner` is respected across all backends (especially for SIMD/CUDA requirements).
- [ ] **Gemma 3 Validation:** Run `examples/gemma-3-270m.py` to end-to-end verify the system on a real transformer model.

## 5. Documentation & Cleanup
- [ ] **Remove Redundancy:** Remove `Executor` in favor of `StaticExecutor`.
- [ ] **Update `CLAUDE.md`:** Document the new compilation workflow.