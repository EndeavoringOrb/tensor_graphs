# tensor_graphs

I want to make a language that can be used to describe mathematic functions (like LLMs or diffusion models). The idea being that if I can decompose LLMs and diffusion models down into basic operations like addition, multiplication, dot products, then I can make a system for automatic optimization. Like I can write an optimized addition+multiplication CUDA kernel and then that can be applied to every LLM and diffusion model that uses that combination. I also want to be able to symbolically rearrange stuff so I can test out different dependancy graphs to see if I can increase parallelism and in turn increase speed. Then this can also be combined with LLMs to automatically generate new fused kernels because we have reference implementations for the atomic operations (add, mul, dot, etc).

Please help me figure out the best way to integrate the tensor_graphs framework with a robust way to benchmark kernels. My vision is the user says "run this graph" and the framework uses the best kernels. But how do we define "best"? Best is fastest at a given accuracy/tolerance. So lets say the graph calls for tanh on (tensor, cuda, shape(4,16), fp32). We could use the dedicated cuda kernel, or we could replace it with cuda2numpy+numpytanh+numpy2cuda or we could decompose it into lower level cuda kernels. My point is there are many "paths" we can use to execute a graph. I want to integrate a database of records "I ran this graph on this hardware this many times and here are all the timings" so that when deciding which path to take, you can use actual profiling to decide and if there isn't anything in the database then you profile and add to the database. I realize that there are going to be many many paths for valid execution, but I want to have the framework be able to generate a list (or an iterator) over all paths so that eventually we can profile them all. If someone just wants the fastest then it will use the database (+best guess where no data). But there should also be a program that iterates over possible paths and profiles them. Do not write any code, just brainstorm.

## General TODO:
- Need to update registry to store backend as well (numpy/cuda, prefer backend as opposed to device because there is no guarantee that two cpu backends will use the same format)
- Need to build graph comparison tool for checking if two graphs are equal
- Need to build graph profiler
- Need to figure out exact SQL schema for benchmarkDB

## Example Graph Execution Workflow
1. User builds gemma-3-270m graph (example)
2. Execute graph
    -are there any kernels that encompass this graph (check if that kernel's graph's atomic representation can be reorganized as this graph's atomic representation)? if so then add them as an option
    -can this graph be decomposed any further? (i.e. cuda->cuda2numpy+numpyOp+numpy2cuda, or tanh->exp,div,add,neg) if so then add all possible decomposed graphs as an option for execution. Note that backend transfer should be its own set of ops
3. Choose from options. Search benchmarkDB then fall back to heuristic. flag the unprofiled resolved graphs for offline profiling.
4. Execute chosen option (repeat from step 2)

## benchmark DB schema
### TODO:
- split up hardware envs into devices, drivers, misc_attrs and have junction tables
- somewhere need to record tolerance, like using fp8 kernel for fp32 operation
- figure out how to choose from list of resolved graphs based on benchmark DB (including interpolation for unprofiled cases)

Table: hardware_environments (must work for various gpus+cpus nvidia, amd, intel, mac cpus, snapdragon)
- id: UUID
- device_name: VARCHAR (e.g., "NVIDIA GeForce RTX 4090")
- memory_capacity: # bytes memory
- attributes: (json, driver versions, cuda version)

Table: atomic node axes
- id: UUID
- atomic_node_id: FK -> atomic nodes
- name: "batch_size" or "hidden_size"
- value: INT

Table: hashes
- id: UUID
- hash
- hash_algo

Table: atomic nodes (atomic graph that was requested, ex: gemma-3-270m or sdxl image gen. want this table to be a bunch of nodes + a parent-to-child junction table so all the columns are constant size)
- id: UUID
- inputs list
  - name
  - dtype (fp32, bfloat16, int4, bool, etc.)
  - shape: list of atomic node axes
- outputs list
  - name
  - dtype
  - shape: list of atomic node axes
- name (unique, will be used to find reference implementation file)
- reference implementation source file hashes.id

Table: resolved nodes (the resolved graph that was used for computation. want this table to be a bunch of nodes + a parent-to-child junction table so all the columns are constant size.)
- id
- root atomic node (this table doesn't need input/output because it can be retrieved by following atomic nodes)
- name (unique, will be used to find kernel implementation file)
- kernel id

Table: benchmark_runs
- id: UUID
- resolved root node id: UUID (FK -> resolved nodes)
- peak memory bytes

Table: run_samples
- benchmark id (FK -> benchmark_runs)
- iteration_index: (was this the 1st,3rd,7th,etc run of the benchmark)
- latency_ns

---

Do not write any code, just help me brainstorm, poke holes in my plans, and point out existing programs/frameworks.
