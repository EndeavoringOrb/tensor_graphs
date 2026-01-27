from typing import List, Dict, Any, Callable
from ..ir.node import TensorNode
from ..ir.graph import topological_sort
from ..ir.dtypes import TensorSignature
from ..backend.registry import KernelRegistry
from .planner import ExecutionRecipe
from .liveness import LivenessAnalyzer
from .memory_planner import MemoryPlanner
from .compiled_graph import CompiledGraph, OpInstruction, TensorMetadata


class Compiler:
    def __init__(self, memory_planner: MemoryPlanner = None):
        self.memory_planner = memory_planner or MemoryPlanner()

    def compile(self, recipe: ExecutionRecipe) -> CompiledGraph:
        # 1. Topo Sort
        nodes = topological_sort(recipe.root)

        # 2. Liveness
        liveness = LivenessAnalyzer.analyze(nodes)

        # 3. Memory Planning
        allocations = self.memory_planner.plan(nodes, liveness)

        # 4. Instruction Generation
        instructions = []
        node_metadata = {}

        # Helper to get offset
        def get_offset(node: TensorNode) -> int:
            if node not in allocations:
                # This might happen if the node is zero-sized or special?
                # Or if MemoryPlanner missed it.
                raise RuntimeError(f"No allocation for node {node.name}")
            return allocations[node].offset

        # Helper to get metadata
        def get_metadata(node: TensorNode) -> TensorMetadata:
            return TensorMetadata(shape=node.shape, dtype=node.dtype)

        # Calculate total memory needed
        max_offset = 0
        for alloc in allocations.values():
            max_offset = max(max_offset, alloc.offset + alloc.size_bytes)

        # We also need to identify inputs/outputs
        input_offsets = {}
        output_offsets = {}

        # Note: nodes is sorted.
        for node in nodes:
            # Store metadata
            node_metadata[node.name] = get_metadata(node)

            # Inputs
            if node.op_type == "Input":
                input_offsets[node.name] = get_offset(node)
                continue  # No instruction for Input (data is copied in)

            if node.op_type == "Constant":
                # Constants are pre-loaded. We might need an init step?
                # Or we assume they are Inputs in the static execution model?
                # The prompt says: "Use the param() and constant() helpers to mark weights as PERSISTENT."
                # Persistent nodes are allocated in the weight pool.
                # We need to populate them!
                # For now, we'll treat them as inputs that need to be filled,
                # OR we handle them in the Executor initialization.
                pass

            # Find Kernel
            # Backend?
            backend = recipe.assignments.get(node, node.backend)
            input_sigs = [p.signature for p in node.parents]

            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel:
                if node.op_type == "Constant":
                    # Constants might not have kernels if they are handled specially
                    continue
                raise RuntimeError(f"Kernel not found for {node.op_type} on {backend}")

            # Create Instruction
            input_offs = [get_offset(p) for p in node.parents]
            input_names = [p.name for p in node.parents]
            output_off = get_offset(node)

            instr = OpInstruction(
                node_name=node.name,
                kernel=kernel,
                input_offsets=input_offs,
                input_node_names=input_names,
                output_offset=output_off,
                attrs=node.attrs,
            )
            instructions.append(instr)

        # The root is the output (usually)
        # If there are multiple outputs, we need to know them.
        # For now, assume root is the single output.
        output_offsets[recipe.root.name] = get_offset(recipe.root)

        return CompiledGraph(
            instructions=instructions,
            buffer_allocations={n.name: a for n, a in allocations.items()},
            node_metadata=node_metadata,
            total_memory_bytes=max_offset,
            input_offsets=input_offsets,
            output_offsets=output_offsets,
        )
