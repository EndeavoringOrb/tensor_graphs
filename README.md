# tensor_graphs

**A modular Intermediate Representation (IR) for decomposing, analyzing, and optimizing mathematical models.**

`tensor_graphs` is a lightweight framework designed to represent complex neural networks (LLMs, Diffusion Models) as a directed acyclic graph (DAG) of atomic operations. By decomposing models into primitives (Add, Mul, Dot Product), we enable powerful graph rewrites, symbolic analysis, and automatic kernel fusion.

---

## What does this do?

1.  **Decomposition**: Break models down into atomic mathematical units.
2.  **Explicit Typing**: Strictly typed IR (`FP32`, `FP8E4M3`) to handle quantization and mixed-precision explicitly.
3.  **Kernel Dispatch**: A registry-based backend that matches specific hardware implementations (kernels) to operations.
4.  **Optimization**: Automatically fuse operations (e.g., `Mul` + `Add` → `FusedMulAdd`).