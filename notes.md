# tensor_graphs

I want to make a language that can be used to describe mathematic functions (like LLMs or diffusion models). The idea being that if I can decompose LLMs and diffusion models down into basic operations like addition, multiplication, dot products, then I can make a system for automatic optimization. Like I can write an optimized addition+multiplication CUDA kernel and then that can be applied to every LLM and diffusion model that uses that combination. I also want to be able to symbolically rearrange stuff so I can test out different dependancy graphs to see if I can increase parallelism and in turn increase speed. Then this can also be combined with LLMs to automatically generate new fused kernels because we have reference implementations for the atomic operations (add, mul, dot, etc).

My vision is the user says "run this graph" and the framework uses the best kernels. But how do we define "best"? Best is fastest at a given accuracy/tolerance. So lets say the graph calls for tanh on (tensor, cuda, shape(4,16), fp32). We could use the dedicated cuda kernel, or we could replace it with cuda2numpy+numpytanh+numpy2cuda or we could decompose it into lower level cuda kernels. My point is there are many "paths" we can use to execute a graph. I want to integrate a database of records "I ran this graph on this hardware this many times and here are all the timings" so that when deciding which path to take, you can use actual profiling to decide and if there isn't anything in the database then you profile and add to the database. I realize that there are going to be many many paths for valid execution, but I want to have the framework be able to generate a list (or an iterator) over all paths so that eventually we can profile them all. If someone just wants the fastest then it will use the database (+best guess where no data). But there should also be a program that iterates over possible paths and profiles them. Do not write any code, just brainstorm.

---

Right now planner and db put the whole graph into the db. I want it to look for all "paths" and the db should only benchmark specific kernels. So for example, lets say I execute a TensorNode representing tanh. It should iterate over all kernels and check if the kernel's reference graph is equal to the input graph. Now lets say we have a GPU_TORCH tanh kernel, and a CPU_NUMPY tanh kernel, and tanh can be decomposed down further. This means we have 3 options (although there could be many more as we can have multiple tanh kernels for a given backend). So for each of the 3 options, we need to see if we can fit it into the graph, like are we able to convert our inputs to the kernel input signature and the output back (this might be a no-op). Then for each of the remaining options (the ones we can convert to/from) if they are not a kernel we need to decompose and call plan on the decomposed children. Like tanh -> (DIVIDE, (e^x - e^(-x)), (e^x + e^(-x))). Can you help me concretely plan this out? Then for kernels we need to get the runtime (call db wrapper that handles interpolation). the runtime estimation needs to take into account convert time which I think can be we could just always compute a kernel's cost as (transfer + execution) and for sequential operations on the same backend the transfer cost will be near 0 in the db (same as an op, just query db for CopyTo, no need for separate transfer cost estimation).

I think this means we need to update the DB. The db shouldn't store canonical_graphs, just the kernel and its benchmark time. This way we can fetch the reference graph in python and we don't need to worry about hashing graphs because we only store kernels. This will be better because there doesn't have to be distinction between atomic and fused, it is just all kernels. tensor_graphs/benchmark/db.py should handle initializing the db and interpolating results. Like when the planner queries the DB for MatMul(100, 100) x (100, 100) and finding no exact match, it should look for the nearest shapes (e.g., 128x128) and scale the latency based on other data points. Like fit linear regression mapping input parameters -> runtime and then use that linear regression to predict runtime. This can also be cached in the python wrapper.

For checking graph equality we need to normalize graphs. Always rewriting B+A to A+B (based on hash). Always rewriting Add(Add(A, B), C) to a canonical associative form (e.g., left-associative tree). Like if I have `a+(b*c)` that can be `(a*b)+(a*c)`. Also you can match `a+(b*c)` to a fused multiply-add operation. I feel this can be really powerful for decreasing runtime (maybe you can rewrite something to be more parallelizable). Hashing should have commutative sort.

To avoid infinite recursion during decomposition, an operations decomposition should not be allowed to contain itself.

Something else is calling the DB and running the planner for every add and mul will be very slow. Can we implement some caching mechanism so we only need to do "TransformerBlock" once? But I don't want to hardcode any blocks, it should be automatically discovered using the graph equality checker.

---

## High level TODO

### 1. Memory Management Architecture (The "Buffer" System)
Instead of the `Executor` creating new arrays/tensors on the fly, it should work within a pre-calculated memory map.

*   **[ ] Buffer Metadata & Physical Devices:** 
    *   Define a `Device` abstraction (e.g., `cuda:0`, `cpu`) distinct from `Backend`. Multiple backends (Torch, Triton, Custom CUDA) can share the same `Device` memory.
    *   Track `Device` capacity and current allocation.
*   **[ ] Static Memory Planner (The "Dry Run"):**
    *   Implement **Liveness Analysis**: Iterate through the `ExecutionRecipe` and track the "first use" and "last use" of every node.
    *   **Reference Counting:** During planning, calculate exactly how many times each tensor is consumed.
*   **[ ] Offset-Based Allocation:**
    *   Instead of `malloc`, use a "Greedy Block" algorithm. Assign each transient tensor a `(device, offset, size)` within a single large `WorkspaceBuffer`.
    *   **Buffer Reuse:** If Tensor A is no longer needed after Op 5, Tensor B (Op 6) should occupy the same memory offset if it fits.
*   **[ ] Pre-Execution Memory Estimation:**
    *   Add a method: `recipe.estimate_memory_requirements() -> Dict[Device, int]`. This allows the user to fail early with `OutOfMemory` before a single kernel is launched.

### 2. Persistent Tensors & State (KV Caching)
Weights and KV Caches should not be managed like transient activations.

*   **[ ] Persistent vs. Transient Flag:**
    *   Update `TensorNode` or the `ExecutionRecipe` to distinguish between "Weights" (static), "State" (KV Cache - persistent but mutable), and "Activations" (transient).
*   **[ ] The `StateNode` IR:**
    *   Introduce a node that represents a variable. It needs a `read()` and `update()` semantic.
    *   **KV Cache Optimization:** Create a specific logic for appending to a pre-allocated buffer (e.g., a 4096-length buffer where we only "activate" the first `N` tokens).
*   **[ ] Weight Binding:**
    *   Modify the `Executor` to allow "Pre-binding." Instead of passing weights in the `feed_dict` every call, they are loaded onto the device once and the `ExecutionRecipe` stores pointers/references to those device addresses.

### 3. "Compiled" Execution Model
The Python overhead of traversing the graph is currently your biggest bottleneck for small models like 270M.

*   **[ ] The `CompiledGraph` Object:**
    *   The `Planner` should return a `CompiledGraph`.
    *   This object contains a **Flat Instruction List** (e.g., `[(kernel_fn, input_offsets, output_offset, attrs), ...]`).
    *   Execution then becomes a simple `for` loop over a list of function pointers—no recursion, no dict lookups.
*   **[ ] Symbolic Shape Support:**
    *   Gemma's sequence length changes. Instead of re-planning when `seq_len` goes from 10 to 11, the `Planner` should handle `sympy` expressions for shapes.
    *   Memory offsets would be calculated as functions of `N` (sequence length).

### 4. Graph-Level Optimization Passes (The "Middle-End")
Before the Planner picks kernels, the graph itself should be cleaned.

*   **[ ] Operator Overloading & Fluent API:**
    *   Implement `__add__`, `__mul__`, `__matmul__`, `__getitem__` in `TensorNode` to allow `x = A @ B + C` syntax.
*   **[ ] Algebraic Simplification Pass:**
    *   **Reshape Folding:** `reshape(reshape(x, shape1), shape2) -> reshape(x, shape2)`.
    *   **Identity Elimination:** Remove copies or reshapes that do nothing.
    *   **Constant Folding:** If an operation has only `OpType.CONSTANT` parents, compute the result at compile time and replace the node with a new `CONSTANT`.
*   **[ ] DType/Backend Coalescing:**
    *   If the Planner sees `Copy(CPU->GPU) -> Op -> Copy(GPU->CPU)`, it should check if a CPU version of `Op` exists that is faster than the two transfers combined.

### 5. Data Ingestion & Tooling
*   **[ ] Direct Safetensors -> Device Loading:**
    *   Write a utility to map a `.safetensors` file directly into `GPU_TORCH` or a raw CUDA pointer, bypassing the NumPy/CPU stage entirely.
*   **[ ] Accuracy vs. Speed Profiling:**
    *   Update the `BenchmarkDB` to store `max_relative_error`.
    *   Allow the Planner to take a "Tolerance" flag: "Give me the fastest path that stays within $10^{-5}$ precision."

### 6. Summary of Updated Execution Workflow
1.  **Define:** User defines graph using overloaded operators.
2.  **Bind:** User binds huge weights to specific `Persistent` nodes.
3.  **Compile:**
    *   Simplifier pass (folding).
    *   Planner pass (kernel selection & fusion).
    *   **Memory Planner pass** (liveness analysis & offset assignment).
4.  **Run:** `CompiledGraph.run(inputs)` executes the flat instruction list using pre-allocated offsets.