# tensor_graphs

**A modular Intermediate Representation (IR) for decomposing, analyzing, and optimizing mathematical models.**

`tensor_graphs` is a lightweight framework designed to represent complex neural networks (LLMs, Diffusion Models) as a directed acyclic graph (DAG) of atomic operations. By decomposing models into primitives (Add, Mul, Dot Product), we enable powerful graph rewrites, symbolic analysis, and automatic kernel fusion.

---

## 🚀 The Vision

Current deep learning frameworks are often monolithic. `tensor_graphs` aims to be the "glue" between high-level model definitions and low-level hardware kernels.

1.  **Decomposition**: Break models down into atomic mathematical units.
2.  **Symbolic Analysis**: Use tools like SymPy to rearrange graphs for maximum parallelism.
3.  **Optimization**: Automatically fuse operations (e.g., `Mul` + `Add` → `FusedMulAdd`).
4.  **Code Generation**: (Future) Use LLMs to write optimized CUDA/Triton kernels for valid subgraphs found in the IR.

---

## 📂 Architecture

The project is structured to separate the graph definition from the execution logic.

```text
tensor_graphs/
├── ir/          # The Graph (Nodes, DAG traversal)
├── ops/         # The Vocabulary (Atomic operations: Add, Mul, Dot)
├── optim/       # The Compiler (Fusion, Pattern Matching, SymPy analysis)
└── backend/     # The Execution (Reference NumPy impl, future CUDA)
```

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tensor_graphs.git
   cd tensor_graphs
   ```

2. Install dependencies (NumPy and SymPy):
   ```bash
   pip install -r requirements
   ```

3. Run the test suite to ensure everything is working:
   ```bash
   python3 -m unittest discover tensor_graphs/tests
   ```

---

## ⚡ Quick Start

Here is how to define a simple computational graph ($y = x \cdot w + b$), optimize it, and execute it.

### 1. Build the Graph
```python
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph

# Define Nodes
x = TensorNode(OpType.INPUT, (32,), [], "input_x")
w = TensorNode(OpType.INPUT, (32,), [], "weight_w")
b = TensorNode(OpType.INPUT, (32,), [], "bias_b")

# Build DAG: y = (x * w) + b
mul_node = TensorNode(OpType.MUL, (32,), [x, w], "mul_op")
output_node = TensorNode(OpType.ADD, (32,), [mul_node, b], "add_op")

# Data inputs
inputs = {
    "input_x": np.random.rand(32),
    "weight_w": np.random.rand(32),
    "bias_b": np.zeros(32)
}

# Execute (Reference Backend)
result = evaluate_graph(output_node, inputs)
print("Reference Result:", result[:4])
```

### 2. Optimize (Fuse) the Graph
The optimizer detects patterns (like Multiply followed by Add) and fuses them into a single node. This reduces memory bandwidth overhead in hardware implementations.

```python
from tensor_graphs.optim.fusion import fuse_graph

# Run the optimizer pass
optimized_graph = fuse_graph(output_node)

print(f"Original Op:  {output_node.op_type}")      # 'Add'
print(f"Optimized Op: {optimized_graph.op_type}") # 'FusedMulAdd'

# Verify the result is numerically identical
opt_result = evaluate_graph(optimized_graph, inputs)
assert np.allclose(result, opt_result)
print("Optimization successful!")
```

---

## 🧠 Optimization Passes

### Kernel Fusion
The `optim/fusion.py` module traverses the graph post-order. It looks for specific subgraphs that can be replaced by more efficient "super-nodes."

*   **Input**: `Add(Mul(A, B), C)`
    *   *Depth*: 2 operations
    *   *Memory*: Writes intermediate `Mul` result to VRAM.
*   **Output**: `FusedMulAdd(A, B, C)`
    *   *Depth*: 1 operation
    *   *Memory*: Computes in registers; zero intermediate VRAM writes.

### Symbolic Analysis
The `optim/symbolic.py` module converts the IR into **SymPy** expressions. This allows us to:
1.  Verify if two complex graphs are mathematically equivalent.
2.  Calculate derivatives automatically.
3.  Rearrange terms to minimize dependency depth.

```python
from tensor_graphs.optim.symbolic import to_sympy

expr = to_sympy(output_node)
print(f"Symbolic Form: {expr}")
# Output: bias_b + input_x * weight_w
```

---

## 🗺 Roadmap

- [x] **Core IR**: DAG definition and topological sorting.
- [x] **Reference Backend**: NumPy execution for ground truth verification.
- [x] **Basic Optimizer**: Pattern matching for `Mul+Add` fusion.
- [ ] **Symbolic Solver**: Use SymPy to automatically propose better graph layouts.
- [ ] **LLM Integration**: Feed optimized subgraphs to an LLM to generate custom CUDA kernels.
- [ ] **GPU Backend**: A runner that compiles and executes the generated CUDA code.

---

## 🤝 Contributing

Contributions are welcome! If you want to add new atomic operations (like `Softmax` or `RMSNorm`) or improve the fusion logic:

1.  Add the Op name to `tensor_graphs/ops/atomic.py`.
2.  Implement the logic in `tensor_graphs/backend/reference.py`.
3.  Add a test case in `tests/test_atomic_ops.py`.