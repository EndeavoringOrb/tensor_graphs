# Detailed implementation plan for the Static Memory & Compiled Execution System.

### Phase 1: Abstractions & Metadata
Currently, `TensorNode` is used for both symbolic IR and execution. We need to decouple the *Logical Tensor* (the graph node) from the *Physical Buffer* (the memory address).

**1.1 Define Buffer Metadata**
Create a `BufferDescriptor` to track where a tensor lives.
```python
class StorageType(Enum):
    TRANSIENT = "transient"   # Activations (recyclable)
    PERSISTENT = "persistent" # Weights (static)
    STATE = "state"           # KV Cache (persistent but mutable)

@dataclass
class BufferAllocation:
    node_id: str
    device: str          # e.g., "cuda:0" or "cpu"
    storage_type: StorageType
    size_bytes: int
    offset: int = 0      # Assigned later by the Allocator
```

**1.2 Update the ExecutionRecipe**
The `ExecutionRecipe` should transition from a mapping of `Node -> Backend` to a **Linear Program**.
*   **Instruction List:** A flat list of `(kernel_fn, input_buffer_ids, output_buffer_id, attrs)`.
*   **Memory Map:** A dictionary of `buffer_id -> BufferAllocation`.

---

### Phase 2: The Middle-End (Liveness Analysis)
To reuse memory, we must know exactly when a buffer is no longer needed.

**2.1 Reference Counting & Intervals**
1.  Perform a `topological_sort` on the planned graph.
2.  Assign an integer `time_step` to each node based on its position in the sort.
3.  **Birth:** The node's `time_step`.
4.  **Death:** The `time_step` of the **last** child node to consume it as an input. 
    *   *Note:* Graph "Output" nodes have a death of `Infinity`.

```python
def compute_liveness(topo_nodes):
    intervals = {} # node -> [start, end]
    for i, node in enumerate(topo_nodes):
        # Initial birth
        intervals[node] = [i, i]
        # Update death of parents
        for p in node.parents:
            intervals[p][1] = max(intervals[p][1], i)
    return intervals
```

---

### Phase 3: The Greedy Static Allocator
We will use a "First-Fit" algorithm on a 1D memory space. Imagine the memory as the Y-axis and Time as the X-axis. We are packing rectangles (size x duration) without overlaps.

**3.1 Offset Assignment Algorithm**
1.  Separate nodes into `PERSISTENT` and `TRANSIENT`.
2.  **Persistent:** Assign fixed offsets in a `WeightPool` (no reuse).
3.  **Transient:** 
    *   Sort nodes by `start_time`.
    *   Maintain a list of "currently active" allocations.
    *   When a node "dies," free its offset back to the pool.
    *   When a new node is born, find the smallest available offset that doesn't collide with existing active offsets.

---

### Phase 4: The Lowering (Compiler)
Lower the `ExecutionRecipe` into a `CompiledGraph`. This is where you remove the Python overhead.

**4.1 The Virtual Machine Instruction**
Create a simplified class that stores only the pointers and logic needed for execution.

```python
@dataclass
class OpInstruction:
    kernel: Callable
    input_offsets: List[int]
    output_offset: int
    attrs: Dict
```

**4.2 Implementing the Physical Buffer**
*   **GPU (Torch Backend):** Allocate one large `torch.ByteTensor` on the device. Every instruction will use `buffer[offset : offset + size].view(shape, dtype)`.
*   **CPU (Numpy Backend):** Use a `bytearray` or a pre-allocated `np.empty(total_size, dtype=np.uint8)`. Use `np.frombuffer` with offsets and strides to create views.

---

### Phase 5: The Static Executor
The new `run` loop should look like this (extremely fast):

```python
class StaticExecutor:
    def __init__(self, compiled_graph):
        self.raw_buffer = preallocate(compiled_graph.total_size)
        self.instructions = compiled_graph.instructions

    def run(self, inputs):
        # 1. Copy inputs into pre-assigned offsets in raw_buffer
        for name, data in inputs.items():
            offset = self.input_map[name]
            copy_to_buffer(self.raw_buffer, offset, data)

        # 2. Hot Loop (Zero Dict Lookups)
        for inst in self.instructions:
            # Prepare views (this is still slightly slow in Python, 
            # ideally kernels take raw pointers/offsets)
            args = [view(self.raw_buffer, off) for off in inst.input_offsets]
            out = view(self.raw_buffer, inst.output_offset)
            
            inst.kernel(*args, out, inst.attrs)

        return view(self.raw_buffer, self.output_offset)
```

---

### Immediate Next Steps (The Action Plan):
1.  **Modify `TensorNode`:** Add a `storage_type` attribute.
2.  **Update `examples/gemma-3-270m.py`:** Use the `param()` and `constant()` helpers to mark weights as `PERSISTENT`.
3.  **Build `LivenessAnalyzer`:** Test it on a simple `Add(Mul(x, y), z)` graph to ensure death times are correct.
4.  **Build `MemoryPlanner`:** Implement the first-fit offset assignment.
5.  **Refactor `Executor`:** Create the `StaticExecutor` that takes a `bytearray` and runs the flat loop.