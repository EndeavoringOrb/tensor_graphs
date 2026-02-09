from typing import Optional, Dict, Any
from ..ir.graph import topological_sort
from ..backend.registry import KernelRegistry
from .planner import ExecutionRecipe
from .compiled_graph import CompiledGraph, OpInstruction
from .propagation import GraphPropagator


class Compiler:
    def compile(
        self, recipe: ExecutionRecipe, known_values: Optional[Dict[str, Any]] = None
    ) -> CompiledGraph:
        # 1. Topo Sort
        nodes = topological_sort(recipe.root)

        # 2. Shape Inference
        if known_values:
            GraphPropagator.infer_shapes(nodes, known_values)

        # 3. Calculate Ref Counts for memory release
        ref_counts = {n.name: 0 for n in nodes}
        for node in nodes:
            for p in node.parents:
                ref_counts[p.name] = ref_counts.get(p.name, 0) + 1

        # Root is implicitly consumed by the session output
        ref_counts[recipe.root.name] += 1

        # 4. Instruction Generation
        instructions = []
        nodes_map = {}

        for node in nodes:
            nodes_map[node.name] = node

            if node.op_type in ("Input", "Constant"):
                continue

            backend = recipe.assignments.get(node, node.backend)
            input_sigs = [p.signature for p in node.parents]

            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel:
                raise RuntimeError(f"Kernel not found for {node.op_type}")

            input_names = [p.name for p in node.parents]

            instr = OpInstruction(
                node_name=node.name,
                kernel=kernel,
                input_node_names=input_names,
                attrs=node.attrs,
            )
            instructions.append(instr)

        return CompiledGraph(
            instructions=instructions,
            ref_counts=ref_counts,
            nodes_map=nodes_map,
        )
