# tensor_graphs

**A modular Intermediate Representation (IR) for decomposing, analyzing, and optimizing mathematical models.**

`tensor_graphs` is a lightweight framework designed to represent complex neural networks (LLMs, Diffusion Models) as a directed acyclic graph (DAG) of atomic operations. By decomposing models into primitives (Add, Mul, Dot Product), we enable powerful graph rewrites, symbolic analysis, and automatic kernel fusion.

---

## 🚀 The Vision

Current deep learning frameworks are often monolithic. `tensor_graphs` aims to be the "glue" between high-level model definitions and low-level hardware kernels.

1.  **Decomposition**: Break models down into atomic mathematical units.
2.  **Explicit Typing**: Strictly typed IR (`FP32`, `FP8E4M3`) to handle quantization and mixed-precision explicitly.
3.  **Kernel Dispatch**: A registry-based backend that matches specific hardware implementations (kernels) to operations.
4.  **Optimization**: Automatically fuse operations (e.g., `Mul` + `Add` → `FusedMulAdd`).

---

## ⚡ Quick Start

Here is how to define a simple computational graph ($y = x \cdot w + b$), optimize it, and execute it.

### 1. Build the Graph
```python
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph

# Define Nodes with Shape and Type (Strict)
x = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "input_x")
w = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "weight_w")
b = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "bias_b")

# Build DAG: y = (x * w) + b
mul_node = TensorNode(OpType.MUL, (32,), DType.FP32, [x, w], "mul_op")
output_node = TensorNode(OpType.ADD, (32,), DType.FP32, [mul_node, b], "add_op")

# Data inputs
inputs = {
    "input_x": np.random.rand(32).astype(np.float32),
    "weight_w": np.random.rand(32).astype(np.float32),
    "bias_b": np.zeros(32, dtype=np.float32)
}

# Execute (Reference Backend uses KernelRegistry)
result = evaluate_graph(output_node, inputs)
print("Reference Result:", result[:4])
```

### 2. Optimize (Fuse) the Graph
The optimizer detects patterns (like Multiply followed by Add) and fuses them into a single node.

```python
from tensor_graphs.optim.fusion import fuse_graph

# Run the optimizer pass
optimized_graph = fuse_graph(output_node)

print(f"Original Op:  {output_node.op_type}")      # 'Add'
print(f"Optimized Op: {optimized_graph.op_type}") # 'FusedMulAdd'
```

---

## 📂 Architecture

```text
tensor_graphs/
├── ir/          # The Graph (Nodes, Types, DAG)
├── ops/         # The Vocabulary
├── optim/       # The Compiler (Fusion, Dispatcher)
└── backend/     # The Execution
    ├── registry.py      # Kernel Database
    └── ops/implementations.py # Explicit Kernels (Scalar, Vector, Matrix)
```