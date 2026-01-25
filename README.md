# tensor_graphs

I want to make a language that can be used to describe mathematic functions (like LLMs or diffusion models). The idea being that if I can decompose LLMs and diffusion models down into basic operations like addition, multiplication, dot products, then I can make a system for automatic optimization. Like I can write an optimized addition+multiplication CUDA kernel and then that can be applied to every LLM and diffusion model that uses that combination. I also want to be able to symbolically rearrange stuff so I can test out different dependancy graphs to see if I can increase parallelism and in turn increase speed. Then this can also be combined with LLMs to automatically generate new fused kernels because we have reference implementations for the atomic operations (add, mul, dot, etc).

My vision is the user says "run this graph" and the framework uses the best kernels. But how do we define "best"? Best is fastest at a given accuracy/tolerance. So lets say the graph calls for tanh on (tensor, cuda, shape(4,16), fp32). We could use the dedicated cuda kernel, or we could replace it with cuda2numpy+numpytanh+numpy2cuda or we could decompose it into lower level cuda kernels. My point is there are many "paths" we can use to execute a graph. I want to integrate a database of records "I ran this graph on this hardware this many times and here are all the timings" so that when deciding which path to take, you can use actual profiling to decide and if there isn't anything in the database then you profile and add to the database. I realize that there are going to be many many paths for valid execution, but I want to have the framework be able to generate a list (or an iterator) over all paths so that eventually we can profile them all. If someone just wants the fastest then it will use the database (+best guess where no data). But there should also be a program that iterates over possible paths and profiles them. Do not write any code, just brainstorm.

## General TODO:
- make interface.py sample_inputs automatically generate input requirements based on nodes
- Need to update registry to store backend as well (numpy/cuda, prefer backend as opposed to device because there is no guarantee that two cpu backends will use the same format)
- Need to build graph comparison tool for checking if two graphs are equal
- Need to build graph profiler
- Need to figure out exact SQL schema for benchmarkDB
- ask at gpu mode for tips on profiling
- remove infinite cpu->cuda->cpu loops and stuff like that

## Example Graph Execution Workflow
1. User builds gemma-3-270m graph (example)
2. Execute graph
    -are there any kernels that encompass this graph (check if that kernel's graph's atomic representation can be reorganized as this graph's atomic representation)? if so then add them as an option
    -can this graph be decomposed any further? (i.e. cuda->cuda2numpy+numpyOp+numpy2cuda, or tanh->exp,div,add,neg) if so then add all possible decomposed graphs as an option for execution. Note that backend transfer should be its own set of ops
3. Choose from options. Search benchmarkDB then fall back to heuristic. flag the unprofiled resolved graphs for offline profiling.
4. Execute chosen option (repeat from step 2)

## benchmark DB schema
This benchmark DB stores benchmarks for kernels and corresponding atomic graphs so that you can answer the question "Do I use the tanh cuda kernel or do I use the decomposed tanh atomic graph"

## 1. Database Schema

### A. The Math (Definitions)
This table represents the mathematical intent (e.g., "Gemma Attention Block" or "Tanh"). It is agnostic to how it is calculated.

**Table: `canonical_graphs`**
*   `id`: UUID (Primary Key)
*   `human_name`: VARCHAR (e.g., "gemma_2_attention_layer", "rmsnorm")
*   `structural_hash`: HASH (The hash of the atomic graph topology + op types). **This is the main search key.**
*   `atomic_graph_json`: JSONB (The serialized graph structure for reference).

### B. The Code (Solutions)
This table unifies **Kernels** and **Graph Decompositions**. Both are valid ways to execute a canonical graph.

**Table: `implementations`**
*   `id`: UUID (Primary Key)
*   `canonical_graph_id`: FK $\rightarrow$ `canonical_graphs.id` (Claiming "I implement this math")
*   `type`: ENUM ("KERNEL", "GRAPH_RECIPE")
*   `name`: VARCHAR (e.g., "FlashInfer_v2_fwd", "Numpy_Tanh_Decomp")
*   `backend`: ENUM ("CUDA", "HIP", "NumPy", "Metal", "Torch_Compile")
*   `source_hash`: VARCHAR (Hash of the .cu file or the graph decomposition logic).
*   `requirements`: JSONB (e.g., `{"cuda_min": "12.0", "arch": ["sm_80", "sm_90"]}`)

### C. The Context (Environment)
Normalizing hardware and software to allow querying like "Find best kernel for RTX 4090".

**Table: `environments`**
*   `id`: UUID (Primary Key)
*   `hardware_name`: VARCHAR (e.g., "NVIDIA_H100_80GB_HBM3")
*   `memory_bytes`: BIGINT
*   `platform_info`: JSONB (OS info, Driver versions).
*   `libs_info`: JSONB (Snapshot of library versions, matching Trace `environment.libs`).

### D. The Input (Workload)
Stores specific input configurations so we can distinguish between `batch_size=1` and `batch_size=32`.

**Table: `workloads`**
*   `id`: UUID (Primary Key)
*   `canonical_graph_id`: FK $\rightarrow$ `canonical_graphs.id`
*   `axes_hash`: HASH (Hash of the `axes` dictionary for fast lookups).
*   `axes_json`: JSONB (e.g., `{"batch_size": 32, "seq_len": 2048, "hidden_dim": 4096}`).
*   `input_descriptors`: JSONB (Stores the `inputs` object from the Trace, defining where test data comes from, e.g., safetensors paths).

### E. The Stats (Trace)
This maps the JSON Trace directly to SQL for analytics.

**Table: `benchmark_traces`**
*   `id`: UUID (Primary Key)
*   `implementation_id`: FK $\rightarrow$ `implementations.id` (The Solution)
*   `workload_id`: FK $\rightarrow$ `workloads.id` (The Inputs)
*   `environment_id`: FK $\rightarrow$ `environments.id` (The Hardware)
*   `status`: ENUM ("PASSED", "INCORRECT_NUMERICAL", "RUNTIME_ERROR", etc.)
*   `latency_ms`: FLOAT
*   `speedup_factor`: FLOAT
*   `max_relative_error`: FLOAT (For accuracy filtering)
*   `timestamp`: TIMESTAMP
*   `full_log`: TEXT (Compressed stdout/stderr).