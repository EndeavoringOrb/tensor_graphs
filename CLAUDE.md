# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests with coverage
pytest --cov=tensor_graphs tensor_graphs/tests

# Run a single test file
pytest tests/ops/test_add.py

# Run a specific test
pytest tests/ops/test_add.py::test_add_vec32_optimized

# Run error check & tests & black formatter
./check.sh
```

## Architecture

Tensor Graphs is a graph-based tensor computation library for LLM/diffusion model optimization. The architecture is layered:

### IR Layer (`tensor_graphs/ir/`)
- **TensorNode**: Core graph node containing `op_type`, `shape`, `dtype`, `parents`, `attrs`
- **DType/TensorSignature**: Type system for kernel matching; shapes support `None` as wildcards
- **graph.py**: Utilities for topological sort and input extraction

### Ops Layer (`tensor_graphs/ops/`)
- **OpType**: Enum of all atomic operations (ADD, MUL, DOT, etc.)
- **CompositeOp (interface.py)**: Base class for composite ops that decompose into atomic graphs
- **Fused ops** (`fused/`): Pre-fused patterns like RMSNorm, GELU registered via `@register_composite`

### Backend Layer (`tensor_graphs/backend/`)
- **KernelRegistry** (`registry.py`): Kernel selection by signature scoring
  - Exact dimension match: +10 pts, wildcard (None): +1 pt, mismatch: disqualified
  - Kernels registered via `@KernelRegistry.register(op_type, [input_sigs])`
- **Kernels** (`kernels/atomic/`): Reference implementations per op type with varying specificity
- **reference.py**: `evaluate_graph()` traverses nodes, selects kernels, evaluates to numpy arrays

### Compiler Layer (`tensor_graphs/compiler/`)
- **dispatch.py**: `resolve_dispatch()` rewrites graphs:
  1. Skip Input/Constant nodes
  2. Try direct kernel match
  3. Try decomposition via CompositeOp
  4. Auto-insert Cast nodes to FP32 if no kernel found

### Optimization Layer (`tensor_graphs/optim/`)
- **fusion.py**: Pattern-matching graph rewriting (RMSNorm, GELU, MulAdd)
- `fuse_graph()` recursively traverses and replaces matched subgraphs with fused ops

## Key Patterns

**Adding a new atomic kernel:**
1. Create `tensor_graphs/backend/kernels/atomic/<op>.py`
2. Register kernels with `@KernelRegistry.register(OpType.<OP>, [TensorSignature(...)])`

**Adding a new fused op:**
1. Create `tensor_graphs/ops/fused/<name>.py`
2. Define class inheriting from `CompositeOp` with `decompose()` method
3. Register with `@register_composite`
4. Add pattern matcher in `tensor_graphs/optim/fusion.py`
5. Add kernel in `tensor_graphs/backend/kernels/fused/`