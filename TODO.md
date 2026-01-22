# TODO: Tensor Graphs Project

## 1. Compiler & Symbolic Improvements
- [ ] **In-place Graph Mutation:** Update `resolve_dispatch` to perform graph rewriting/substitution on the existing graph rather than just returning a new root.
    - Files: `tensor_graphs/compiler/dispatch.py`

## 2. Example Model Cleanup (Gemma-3)
- [ ] **Optimize Generation Loop:** Refactor `examples/gemma-3-270m.py` to build the model graph **once** and reuse it across tokens. Currently, it rebuilds the entire graph for every new token.
- [ ] **KV Caching:** Add support for KV cache nodes in the IR to avoid recomputing the entire prefix for every token.
    - Files: `examples/gemma-3-270m.py`