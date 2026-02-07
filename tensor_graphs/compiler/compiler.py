from typing import Optional, Dict, Any
from ..ir.node import TensorNode
from ..ir.graph import topological_sort
from ..backend.registry import KernelRegistry
from .planner import ExecutionRecipe
from .liveness import LivenessAnalyzer
from .memory_planner import MemoryPlanner
from .compiled_graph import CompiledGraph, OpInstruction, TensorMetadata
from .shape_inference import ShapeInference
from .constant_folding import ConstantFolding


class Compiler:
    def __init__(self, memory_planner: Optional[MemoryPlanner] = None):
        self.memory_planner = memory_planner or MemoryPlanner()

    def compile(
        self, recipe: ExecutionRecipe, known_values: Optional[Dict[str, Any]] = None
    ) -> CompiledGraph:
        # 1. Constant Folding (Rebuilds graph structure, so must happen first)
        if known_values:
            recipe.root = ConstantFolding.fold(recipe.root, known_values)

        # 2. Topo Sort (After potential structural changes from folding)
        nodes = topological_sort(recipe.root)

        # 3. Shape Inference (if values are provided)
        if known_values:
            ShapeInference.infer(nodes, known_values)

        # 4. Liveness
        liveness = LivenessAnalyzer.analyze(nodes)

        # 5. Memory Planning
        allocations = self.memory_planner.plan(nodes, liveness)

        # 6. Instruction Generation
        instructions = []
        node_metadata = {}

        # Helper to get offset
        def get_offset(node: TensorNode) -> int:
            if node not in allocations:
                raise RuntimeError(f"No allocation for node {node.name}")
            return allocations[node].offset

        # Helper to get metadata
        def get_metadata(node: TensorNode) -> TensorMetadata:
            if node.shape is None:
                raise ValueError(
                    f"Node {node.name} has undefined shape during compilation."
                )

            if any(s is None for s in node.shape):
                raise ValueError(
                    f"Node {node.name} has dynamic shape {node.shape} during compilation."
                )

            shape_tuple = tuple(s for s in node.shape if s is not None)
            return TensorMetadata(shape=shape_tuple, dtype=node.dtype)

        # Calculate total memory needed
        max_offset = 0
        for alloc in allocations.values():
            max_offset = max(max_offset, alloc.offset + alloc.size_bytes)

        input_offsets = {}
        output_offsets = {}

        for node in nodes:
            # Store metadata
            node_metadata[node.name] = get_metadata(node)

            # Inputs
            if node.op_type == "Input":
                input_offsets[node.name] = get_offset(node)
                continue

            if node.op_type == "Constant":
                # Constants are handled via load_weights usually,
                # or treated as Inputs in execution if not embedded in kernel
                continue

            # Find Kernel
            backend = recipe.assignments.get(node, node.backend)
            input_sigs = [p.signature for p in node.parents]

            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel:
                raise RuntimeError(f"Kernel not found for {node.op_type} on {backend}")

            # Create Instruction
            input_offs = [get_offset(p) for p in node.parents]
            input_names = [p.name for p in node.parents]

            # Currently TensorNode supports single output, so we wrap it in a list
            output_offs = [get_offset(node)]

            instr = OpInstruction(
                node_name=node.name,
                kernel=kernel,
                input_offsets=input_offs,
                input_node_names=input_names,
                output_offsets=output_offs,
                attrs=node.attrs,
            )
            instructions.append(instr)

        # Root is the output
        output_offsets[recipe.root.name] = get_offset(recipe.root)

        return CompiledGraph(
            instructions=instructions,
            buffer_allocations={n.name: a for n, a in allocations.items()},
            node_metadata=node_metadata,
            total_memory_bytes=max_offset,
            input_offsets=input_offsets,
            output_offsets=output_offsets,
        )
