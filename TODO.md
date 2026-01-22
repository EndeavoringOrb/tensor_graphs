This `TODO.md` is organized by functional area and prioritized for a coding agent to work through. Each task includes the relevant file paths and a description of the technical debt or missing feature.

# TODO: Tensor Graphs Project

## 1. Kernel & Backend Generalization
Remove "demo-only" hardcoded logic from the reference kernels.
- [ ] **Improve Kernel Scoring:** Enhance `KernelRegistry._score_candidate` to handle broadcasting rules more explicitly (e.g., matching a dimension of `1` against a dimension of `N`).
    - Files: `tensor_graphs/backend/registry.py`

## 2. Compiler & Symbolic Improvements
- [ ] **In-place Graph Mutation:** Update `resolve_dispatch` to perform graph rewriting/substitution on the existing graph rather than just returning a new root.
    - Files: `tensor_graphs/compiler/dispatch.py`
- [ ] **SymPy Matrix Support:** Update `to_sympy` to use `sympy.MatrixSymbol` for `DOT` operations to allow for better symbolic algebraic simplification of transformer blocks.
    - Files: `tensor_graphs/optim/symbolic.py`
- [ ] **Automated Fusion Pass:** Expand `fusion.py` to recognize `RMSNorm` or `GELU` patterns from atomic sequences and "lift" them back into Fused Ops.
    - Files: `tensor_graphs/optim/fusion.py`

## 3. Example Model Cleanup (Gemma-3)
- [ ] **Optimize Generation Loop:** Refactor `examples/gemma-3-270m.py` to build the model graph **once** and reuse it across tokens. Currently, it rebuilds the entire graph for every new token.
- [ ] **Graph-Based Masking:** Move the causal mask generation entirely into the graph builder so it doesn't require a separate `evaluate_graph` call for every sequence length change.
- [ ] **KV Caching:** Add support for KV cache nodes in the IR to avoid recomputing the entire prefix for every token.
    - Files: `examples/gemma-3-270m.py`