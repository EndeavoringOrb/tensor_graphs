from typing import Optional, Dict, Any
from ..ir.graph import topological_sort
from ..backend.registry import KernelRegistry
from .planner import ExecutionRecipe
from .compiled_graph import CompiledGraph, OpInstruction
from .propagation import GraphPropagator
from ..ir.buffer import StorageType


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
            if node.name in nodes_map:
                raise ValueError(
                    f"Error compiling, {node.name} already exists in nodes_map"
                )
            nodes_map[node.name] = node

            if node.op_type in ("Input", "Constant"):
                continue

            backend = recipe.assignments.get(node, node.backend)
            input_sigs = [p.signature for p in node.parents]

            # Unpack the tuple (kernel, is_inplace)
            kernel_result = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel_result:
                raise RuntimeError(f"Kernel not found for {node.op_type}")

            kernel, is_inplace = kernel_result
            inplace_input_index = None

            # 5. Safety Analysis for In-Place Execution
            if is_inplace:
                # We assume in-place kernels modify the first input (index 0)
                input_names = [p.name for p in node.parents]
                src_node = nodes_map[input_names[0]]
                
                # Check Safety:
                # 1. Ref count must be 1 (this op is the last consumer)
                # 2. Storage must be TRANSIENT (cannot overwrite weights or inputs)
                # 3. Size must match (cannot grow/shrink the buffer in-place)
                is_safe = (
                    ref_counts[src_node.name] == 1 and
                    src_node.storage_type == StorageType.TRANSIENT and
                    node.size_bytes == src_node.size_bytes
                )

                if is_safe:
                    inplace_input_index = 0
                else:
                    # Fallback: Re-query the registry for a non-in-place version
                    fallback_result = KernelRegistry.select_best_kernel(
                        node.op_type, input_sigs, backend, 
                        target_dtype=node.dtype, 
                        allow_inplace=False
                    )
                    if fallback_result:
                        kernel, _ = fallback_result
                        inplace_input_index = None
                    else:
                        # If no out-of-place version exists, we are forced to allocate
                        # but we cannot reuse the buffer. The executor will handle
                        # the out-of-place allocation for the kernel.
                        inplace_input_index = None

            input_names = [p.name for p in node.parents]

            instr = OpInstruction(
                node_name=node.name,
                kernel=kernel,
                input_node_names=input_names,
                attrs=node.attrs,
                inplace_input_index=inplace_input_index
            )
            instructions.append(instr)

        return CompiledGraph(
            instructions=instructions,
            ref_counts=ref_counts,
            nodes_map=nodes_map,
        )