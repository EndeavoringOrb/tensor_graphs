# Current Progress & Remaining Tasks: Static Memory & Compiled Execution

This document tracks the remaining work required to fulfill the objectives in `TODO.md` based on the current state of the `feature/static-execution` branch.

## 1. Optimize `StaticExecutor` (Phase 5 Refinement)
The current `StaticExecutor` implementation performs dictionary lookups and creates tensor views inside the `run` loop, which introduces Python overhead.
- [x] **Pre-calculate Views:** Move view creation (`_get_view`) for all instructions into the `__init__` or a preparation phase.
- [x] **Flatten Instruction Loop:** Ensure the `run` loop only iterates over pre-prepared tuples of `(kernel, input_views, output_view, attrs)`.

## 2. Evolve Kernel API (Zero-Copy Goal)
Currently, kernels return new arrays/tensors, forcing `StaticExecutor` to perform a copy (`np.copyto` or `out.copy_`) to the pre-allocated buffer.
- [ ] **Update `KernelRegistry`:** Make kernels that accept an `out` parameter.
- [ ] **Refactor ALL Kernels:** Update kernels to write directly into the provided `out` buffer.
    - [ ] `reference/add.py`
    - [ ] `reference/arange.py`
    - [ ] `reference/cast.py`
    - [ ] `reference/concat.py`
    - [ ] `reference/copy_to.py`
    - [ ] `reference/cos.py`
    - [ ] `reference/divide.py`
    - [ ] `reference/dot.py`
    - [ ] `reference/exp.py`
    - [ ] `reference/fill.py`
    - [ ] `reference/gather.py`
    - [ ] `reference/max.py`
    - [ ] `reference/mul.py`
    - [ ] `reference/negate.py`
    - [ ] `reference/permute.py`
    - [ ] `reference/power.py`
    - [ ] `reference/repeat.py`
    - [ ] `reference/reshape.py`
    - [ ] `reference/sin.py`
    - [ ] `reference/slice.py`
    - [ ] `reference/sqrt.py`
    - [ ] `reference/sum.py`
    - [ ] `reference/triu.py`
    - [ ] `reference/where.py`
    - [ ] `cpu_numpy/fma.py`
    - [ ] `cpu_numpy/gelu.py`
    - [ ] `cpu_numpy/rms_norm.py`
    - [ ] `cpu_numpy/rope.py`
    - [ ] `cpu_numpy/softmax.py`
    - [ ] `gpu_torch/rms_norm.py`
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