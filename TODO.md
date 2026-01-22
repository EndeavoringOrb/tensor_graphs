This `TODO.md` is organized by functional area and prioritized for a coding agent to work through. Each task includes the relevant file paths and a description of the technical debt or missing feature.

# TODO: Tensor Graphs Project

## 1. Missing Atomic Operations & Primitives
These are required to complete decompositions for high-level LLM operations.
- [x] **Implement `Gather` Atomic Op:** Needed for `Embedding` decomposition.
    - Files: `tensor_graphs/ops/atomic.py`, `tensor_graphs/backend/kernels/atomic/gather.py`
- [x] **Implement `Fill/Full` Atomic Op:** Replace the current workaround of passing "ones" as a manual input.
    - Files: `tensor_graphs/ops/atomic.py`, `tensor_graphs/backend/kernels/atomic/fill.py`
- [x] **Add `Where/Select` Op:** Needed for more robust masking logic.
    - Files: `tensor_graphs/ops/atomic.py`

## 2. Infrastructure & IR Enhancements
Improve the robustness of the Graph Representation.
- [x] **Add Attribute Support to `TensorNode`:** Ops like `Sum` and `Repeat` shouldn't hardcode axes. Add an `attrs: Dict[str, Any]` field to `TensorNode`.
    - Files: `tensor_graphs/ir/node.py`
- [x] **Formalize Constants:** Create a proper `ConstantNode` or a dedicated `OpType.CONSTANT` that stores values internally rather than relying on the `inputs` dictionary for fixed scalars.
    - Files: `tensor_graphs/ir/node.py`, `tensor_graphs/backend/reference.py`
- [ ] **Proper Slice Representation:** Refactor `Slice` to handle standard Python slicing logic (start, stop, step) more gracefully within the IR.
    - Files: `tensor_graphs/ops/atomic.py`, `tensor_graphs/backend/kernels/atomic/slice.py`

## 3. Fused Op Decompositions
Complete the mathematical "lowering" of high-level ops to atomic ops.
- [ ] **Complete `RoPE.decompose`:** Replace the placeholder in `llm.py` with actual `Slice`, `Negate`, `Concat`, `Mul`, and `Add` nodes.
    - Files: `tensor_graphs/ops/fused/llm.py`
- [ ] **Implement `Embedding.decompose`:** Once `Gather` is implemented, lower `Embedding` to a `Gather` operation.
    - Files: `tensor_graphs/ops/fused/llm.py`
- [x] **Make `Softmax` Axis-Aware:** Use the new `attrs` field (from Step 2) to allow Softmax on any dimension, not just `-1`.
    - Files: `tensor_graphs/ops/fused/activation.py`

## 4. Kernel & Backend Generalization
Remove "demo-only" hardcoded logic from the reference kernels.
- [x] **Generalize `repeat.py`:** Use an axis attribute instead of hardcoding `axis=1`.
    - Files: `tensor_graphs/backend/kernels/atomic/repeat.py`
- [x] **Generalize `sum.py`:** Support `keepdims=False` via node attributes.
    - Files: `tensor_graphs/backend/kernels/atomic/sum.py`
- [ ] **Improve Kernel Scoring:** Enhance `KernelRegistry._score_candidate` to handle broadcasting rules more explicitly (e.g., matching a dimension of `1` against a dimension of `N`).
    - Files: `tensor_graphs/backend/registry.py`

## 5. Compiler & Symbolic Improvements
- [ ] **In-place Graph Mutation:** Update `resolve_dispatch` to perform graph rewriting/substitution on the existing graph rather than just returning a new root.
    - Files: `tensor_graphs/compiler/dispatch.py`
- [ ] **SymPy Matrix Support:** Update `to_sympy` to use `sympy.MatrixSymbol` for `DOT` operations to allow for better symbolic algebraic simplification of transformer blocks.
    - Files: `tensor_graphs/optim/symbolic.py`
- [ ] **Automated Fusion Pass:** Expand `fusion.py` to recognize `RMSNorm` or `GELU` patterns from atomic sequences and "lift" them back into Fused Ops.
    - Files: `tensor_graphs/optim/fusion.py`

## 6. Example Model Cleanup (Gemma-3)
- [ ] **Optimize Generation Loop:** Refactor `examples/gemma-3-270m.py` to build the model graph **once** and reuse it across tokens. Currently, it rebuilds the entire graph for every new token.
- [ ] **Graph-Based Masking:** Move the causal mask generation entirely into the graph builder so it doesn't require a separate `evaluate_graph` call for every sequence length change.
- [ ] **KV Caching:** Add support for KV cache nodes in the IR to avoid recomputing the entire prefix for every token.
    - Files: `examples/gemma-3-270m.py`