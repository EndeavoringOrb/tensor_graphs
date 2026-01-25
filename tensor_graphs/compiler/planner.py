from typing import List, Dict, Any, Optional, Iterator, Set, Tuple
from ..ir.node import TensorNode
from ..ir.dtypes import Backend
from ..ops.registry import get_reference_factory
from ..ops.atomic_types import OpType
from ..backend.registry import KernelRegistry
import copy
import json


class ExecutionRecipe:
    def __init__(self, root: TensorNode, assignments: Dict[TensorNode, Backend]):
        self.root = root
        self.assignments = assignments  # Node -> Backend

    def to_dict(self):
        # We need a way to identify nodes that is stable.
        # Since this is an execution plan for a specific graph instance,
        # we can use node names if they are unique, or just serialize the structure.
        return {
            "root_name": self.root.name,
            "assignments": {
                node.name: backend.value for node, backend in self.assignments.items()
            },
        }


class PathGenerator:
    def __init__(self, root: TensorNode):
        self.root = root

    def generate_all_strategies(self) -> Iterator[ExecutionRecipe]:
        """
        Generates different execution recipes for the graph.
        """
        # 1. Generate all possible graph topologies (monolithic vs decomposed)
        for graph_variant in self._generate_graph_variants(self.root):
            # 2. For each topology, generate backend assignments
            for assignments in self._generate_backend_assignments(graph_variant):
                # 3. Insert CopyTo nodes where needed
                final_graph, final_assignments = self._insert_transfers(
                    graph_variant, assignments
                )
                yield ExecutionRecipe(final_graph, final_assignments)

    def _generate_graph_variants(
        self, node: TensorNode, memo=None
    ) -> Iterator[TensorNode]:
        if memo is None:
            memo = {}

        # This is recursive. For each node, we can either keep it or decompose it (if composite).
        factory = get_reference_factory(node.op_type)

        # Option 1: Keep as-is (but recursively generate variants for parents)
        # For simplicity, let's just do two extremes: fully monolithic and fully decomposed.
        yield node

        if factory:
            yield self._fully_decompose(node)

    def _fully_decompose(self, node: TensorNode, memo=None) -> TensorNode:
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]

        new_parents = [self._fully_decompose(p, memo) for p in node.parents]
        factory = get_reference_factory(node.op_type)

        if factory:
            decomposed_node = factory(new_parents, node.attrs)
            memo[node] = decomposed_node
            return decomposed_node

        if new_parents == node.parents:
            memo[node] = node
            return node

        new_node = copy.copy(node)
        new_node.parents = new_parents
        memo[node] = new_node
        return new_node

    def _generate_backend_assignments(
        self, graph: TensorNode
    ) -> Iterator[Dict[TensorNode, Backend]]:
        nodes = list(self._get_all_nodes(graph))

        # Option 1: All on CPU_NUMPY
        yield {node: Backend.CPU_NUMPY for node in nodes}

        # Option 2: All on GPU_TORCH
        yield {node: Backend.GPU_TORCH for node in nodes}

        # In a real implementation, we'd do more interesting mixed assignments.

    def _insert_transfers(
        self, root: TensorNode, assignments: Dict[TensorNode, Backend]
    ) -> Tuple[TensorNode, Dict[TensorNode, Backend]]:
        """
        Recursively walks the graph and inserts COPY_TO nodes where parent backend != child backend.
        """
        new_assignments = assignments.copy()
        memo = {}

        def walk(node: TensorNode) -> TensorNode:
            if node in memo:
                return memo[node]

            node_backend = assignments[node]
            new_parents = []
            changed = False

            for p in node.parents:
                processed_p = walk(p)
                p_backend = assignments.get(
                    p, node_backend
                )  # Default to node_backend for leaf inputs if not assigned

                if p_backend != node_backend:
                    # Insert CopyTo
                    copy_node = TensorNode(
                        op_type=OpType.COPY_TO,
                        shape=p.shape,
                        dtype=p.dtype,
                        parents=[processed_p],
                        name=f"copy_{p.name}_to_{node_backend.value}",
                        attrs={"target_backend": node_backend.value},
                    )
                    new_parents.append(copy_node)
                    new_assignments[copy_node] = node_backend
                    changed = True
                else:
                    if processed_p != p:
                        changed = True
                    new_parents.append(processed_p)

            if not changed:
                memo[node] = node
                return node

            new_node = copy.copy(node)
            new_node.parents = new_parents
            memo[node] = new_node
            new_assignments[new_node] = node_backend
            return new_node

        final_root = walk(root)
        return final_root, new_assignments

    def _get_all_nodes(self, root: TensorNode, visited=None) -> Set[TensorNode]:
        if visited is None:
            visited = set()
        if root in visited:
            return visited
        visited.add(root)
        for p in root.parents:
            self._get_all_nodes(p, visited)
        return visited
