Remove sympy in favor of nodes. Instead of sympy expr seq_len, just have an input node with seq_len as value and connect that to wherever it is needed.

I want to implement incremental execution with selective materialization. Dynamic programming with bounded memoization. Example:

inputs: a,b,c
OP #1. v1 = mul(b, c)
OP #2. v2 = square(v1)
OP #3. out = add(a, v2)

If a is updated, we only need to do OP #3 using the cached output of OP #2. But caching the output of all nodes is too expensive. I want to add a cache policy to all nodes (NEVER, ALWAYS, AUTO) that can be set by the user when constructing a graph but defaults to AUTO. The issue is how to decide whether to cache an output. I'm thinking the user defines an amount of memory that is allocated to cache, and then each time we compute a new output the process to decide whether to cache is:
1. If there is free space, cache it.
2. If there is not free space, compare the kernel cost saved by caching this value against the combined kernel cost of the cached values you must evict to store this value.

Additionally, input nodes specifically will have value-based hashing, all non-input nodes only need to store cached_output and bool:dirty (dirty can be propogated from dirty inputs). Are there other criteria I should take into account like number of times this has been invalidated?
score=kernel_cost*((num_times_graph_ran-num_times_dirty)/num_times_graph_ran).
-increase with kernel_cost (cost of recomputing this node and its entire upstream subgraph, making sure to account for cached upstream nodes) 
-decrease with num_times_dirty
The ultimate goal is speed of inference serving (so compilation and warmup times are not a concern).

For non-input nodes if any parent is dirty, this node is dirty.
For input nodes, hash value & shape & backend and whatever. Mark dirty if hash changes.

Do not cache view-only nodes, caching is useless; recompute cost ≈ 0. Automatically downgrade these to CachePolicy.NEVER.
-reshape
-slice
-permute

Eviction policy is knapsack-style eviction.
-Maintain cache entries in a min-heap by score
-Evict lowest score first until space is sufficient
-Do we need to worry about having sequential available space in memory or will virtual memory take care of this for us?