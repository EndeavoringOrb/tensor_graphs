# Roadmap: Multi-Path Execution & Optimization Framework

## Phase 1: Backend Abstraction & Device Management
The current system implicitly assumes NumPy/CPU. We need to formalize "Backends" to support the "Many Paths" vision (e.g., swapping between CUDA and NumPy).

- [x] **Define Backend Spec:** Create a `Backend` enum/class (e.g., `CPU_NUMPY`, `CPU_TORCH`, `GPU_TORCH`).
  - .cu cuda kernels will be loaded using torch.utils.cpp_extension so both cuda kernels and pytorch gpu kernels will use GPU_TORCH backend.
- [x] **Update Registry:** Refactor `KernelRegistry` to store implementations keyed by `(OpType, Backend, Signature)`.
- [x] **Update TensorNode:** Make `TensorNode` store backend.
- [x] **Data Transfer Ops:** Implement atomic `CopyTo` operations between backends.
- [x] **Kernel Interface Standardization:** Ensure all kernels (CUDA, NumPy, etc.) accept the same input structure so they are interchangeable by the dispatcher.

## Phase 2: Graph Canonicalization & Identity
To search a database for "this graph," we need a way to mathematically identify a graph regardless of variable naming or memory address.

- [ ] **Structural Hashing:** Implement a hashing algorithm (Merkle-tree style) that generates a unique ID based on:
    - OpType
    - Connected topology
    - Constant values (statistically or exactly)
    - *Ignore:* Variable names (e.g., `node_a` vs `node_b`).
- [ ] **Graph Normalization:** Create a pass to "clean" graphs before hashing (e.g., consistent ordering of commutative inputs like Add/Mul).
- [ ] **Equivalence Checker:** Build a tool to verify if `Graph A` is mathematically equivalent to `Graph B`. i.e. `(a + b) * c` == `ac + bc`.
- [ ] **Subgraph Isomorphism Search:** Implement a mechanism to find if a specific subgraph (e.g., a decomposed Tanh) exists within a larger graph.

## Phase 3: The Benchmark Database (SQL)
Implementing the schema to store performance records.

- [ ] **Schema Definition:** Write DDL (SQL) for the proposed schema:
    - `canonical_graphs` (The Math)
    - `implementations` (The Code/Recipe)
    - `environments` (The Hardware)
    - `workloads` (The Input Shapes)
    - `benchmark_traces` (The Results)
- [ ] **Database Connector:** Create a lightweight ORM or SQL wrapper to interact with the DB (SQLite for local dev, Postgres for production).
- [ ] **Environment Sniffer:** Write a script to auto-detect hardware info (GPU name, VRAM, Driver version) to populate the `environments` table.

## Phase 4: The Path Explorer (The "Planner")
Instead of a single `resolve_dispatch`, we need an iterator that generates valid execution graphs.

- [ ] **Decomposition Registry:** Ensure all `CompositeOp`s expose their decomposition logic clearly.
- [ ] **Path Generator:** Create a generator that yields execution strategies:
    - *Path A:* Use monolithic CUDA kernel.
    - *Path B:* Decompose to atomic ops -> run on CUDA.
    - *Path C:* Transfer to CPU -> NumPy Ops -> Transfer to GPU.
- [ ] **Cycle Prevention:** Implement logic to detect and prune inefficient paths (e.g., redundant `Device->Host->Device` loops).
- [ ] **Recipe Serializer:** Define a JSON format to save a specific "Execution Path" (a sequence of Kernels + Transfers) so it can be stored in the `implementations` table.

## Phase 5: Profiling & Autotuning Engine
The system that fills the database.

- [ ] **Micro-Benchmarker:** A harness that takes a `TensorNode` (root), generates dummy data matching the signature, and runs `timeit` on specific backends.
- [ ] **Accuracy Verifier:** A harness that compares the output of a candidate path against a "Golden Reference" (likely High-Precision NumPy) to ensure tolerance compliance.
- [ ] **Offline Profiler:** A script that iterates through unprofiled `canonical_graphs` in the DB and runs benchmarks overnight.
- [ ] **Research:** Ask GPU/HPC communities (e.g., GPUMode) about best practices for consistent micro-benchmarking (warmups, L2 cache flushing, etc.).

## Phase 6: The "Smart" Executor
Tying it all together.

- [ ] **Execution Policy:** Create a config to select mode: `FASTEST` (DB lookup), `EXPLORE` (run benchmarks), or `HEURISTIC` (fallback).
- [ ] **Heuristic Fallback:** If no DB record exists, implement a static cost model (e.g., "Fused is better than Atomic," "Keep data on GPU").
- [ ] **JIT Compilation (Future):** Investigate compiling the chosen path into a static execution plan (reducing Python overhead).