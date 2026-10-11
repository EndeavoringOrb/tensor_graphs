# Memory Alignment Architecture

## Implemented: Option A (Per-MemSpace Page Alignment)

Buffer and offset allocation operates on discrete pages defined per `MemSpace`:

- **`CPP` (`HandleType::CPP`)**: **64 bytes**
  - Matches CPU cache lines and 512-bit AVX-512 vector operations.
  - Enables streaming non-temporal stores (`_mm256_stream_ps`, `_mm512_stream_ps`) and aligned vector loads (`_mm256_load_ps`) without requiring full 4096-byte pages.
  - Reduces internal padding fragmentation for CPU activations and temporary buffers.
- **`CUDA` (`HandleType::CUDA`)**: **256 bytes**
  - Aligns with CUDA memory transaction coalescing and warp-level access granularity.
- **`OPENCL` (`HandleType::OPENCL`)**: **4096 bytes**
  - Satisfies OpenCL platform arena and virtual memory buffer requirements.

Configured via `SearchState::getDefaultPageAlignment(const MemSpace &ms)` and queried with `state.getPageAlignment(ms)`.

---

## Future Work / TODO: Option B (Flexible Per-Kernel / Per-Tensor Alignment in Solver)

Option B treats alignment as a per-kernel requirement or constraint variable within the search solver rather than a fixed global page size per memory space:

- **Concept**:
  - Kernels declare their minimum required buffer/pointer alignment (e.g. 16B for SSE, 32B for AVX2, 64B for AVX-512/streaming stores, 4B/unaligned for scalar fallbacks).
  - Tensors/eclasses receive alignment attributes based on candidate kernel matches.
  - The solver satisfies alignment requirements dynamically when packing offsets:
    $$\text{offset} \equiv 0 \pmod{\text{align}(\text{candidate})}$$
- **Advantages over Option A**:
  - Small activations and scalar/view intermediates (e.g. 4-byte scalars, 8-byte shapes) avoid 64-byte padding overhead.
  - Prevents cache line / L1/L2 cache capacity waste for compact operations.
  - Retains high-performance aligned/streaming instructions for large tensors without paying alignment fragmentation everywhere.
