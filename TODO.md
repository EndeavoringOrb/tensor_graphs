# TODO: Tensor Graphs Project

## 1. Example Model Cleanup (Gemma-3)
- [ ] **Optimize Generation Loop:** Refactor `examples/gemma-3-270m.py` to build the model graph **once** and reuse it across tokens. Currently, it rebuilds the entire graph for every new token.
- [ ] **KV Caching:** Add support for KV cache nodes in the IR to avoid recomputing the entire prefix for every token.
    - Files: `examples/gemma-3-270m.py`