from typing import List, Dict, Any, Tuple
from ..ir.node import TensorNode
from ..ir.dtypes import DType
from ..ops.atomic_types import OpType
import math
from ..ir.graph import topological_sort


class ConstantFolding:
    """
    Evaluates constant subgraphs and replaces them with a single OpType.CONSTANT node.
    """

    # Mapping of OpType to a function that evaluates the operation
    # Inputs are expected to be constant values (scalars or arrays)
    _EVALUATORS = {
        OpType.ADD: lambda a, b: a + b,
        OpType.MUL: lambda a, b: a * b,
        OpType.NEGATE: lambda a: -a,
        OpType.DIVIDE: lambda a, b: a / b,
        OpType.SQRT: lambda a: math.sqrt(a),
        OpType.EXP: lambda a: math.exp(a),
        OpType.SIN: lambda a: math.sin(a),
        OpType.COS: lambda a: math.cos(a),
        OpType.POWER: lambda a, b: a**b,
    }

    @classmethod
    def fold(cls, root: TensorNode, known_values: Dict[str, Any]) -> TensorNode:
        """
        Traverses the graph and folds constants.
        Returns the new root node.
        """
        # We need to rebuild the graph.
        # A simple approach is to traverse topologically and build a new graph.

        # 1. Topological Sort
        nodes = topological_sort(root)

        # Map old nodes to new nodes (for parent references)
        node_map = {}

        # 2. Process nodes
        for node in nodes:
            if node.op_type == OpType.INPUT or len(node.parents) == 0:
                node_map[node] = node
                continue

            # Check if all parents are constants (or already folded constants)
            parent_values = []
            all_parents_constant = True

            for parent in node.parents:
                if parent not in node_map:
                    # Parent not processed yet (shouldn't happen in topo sort, but safety check)
                    all_parents_constant = False
                    break

                parent_val = node_map[parent]

                # If parent is a constant node, get its value
                if parent.op_type == OpType.CONSTANT:
                    val = parent.attrs.get("value")
                    if val is None:
                        all_parents_constant = False
                        break
                    parent_values.append(val)
                else:
                    # Parent is a non-constant node
                    all_parents_constant = False
                    break

            if all_parents_constant and len(parent_values) == len(node.parents):
                # Evaluate the operation
                result = cls._evaluate_op(node.op_type, parent_values, node.attrs)

                # Create a new Constant node
                new_node = TensorNode(
                    OpType.CONSTANT,
                    node.shape,  # Keep shape for compatibility
                    node.dtype,
                    [],  # No parents
                    f"folded_{node.name}",
                    attrs={"value": result},
                    backend=node.backend,
                )
                node_map[node] = new_node
            else:
                # Keep the node as is
                node_map[node] = node

        # 3. Reconstruct Graph
        # We need to find the root again from the map
        # The root is the node that is not a parent of any other node in the map
        all_nodes = set(node_map.values())
        parents = set()
        for n in all_nodes:
            for p in n.parents:
                if p in node_map:
                    parents.add(node_map[p])

        roots = all_nodes - parents
        if len(roots) != 1:
            raise ValueError("Graph folding resulted in multiple roots or no roots.")

        return roots.pop()

    @classmethod
    def _evaluate_op(
        cls, op_type: str, values: List[Any], attrs: Dict[str, Any]
    ) -> Any:
        if op_type == OpType.ADD:
            return values[0] + values[1]
        elif op_type == OpType.MUL:
            return values[0] * values[1]
        elif op_type == OpType.NEGATE:
            return -values[0]
        elif op_type == OpType.DIVIDE:
            return values[0] / values[1]
        elif op_type == OpType.SQRT:
            return math.sqrt(values[0])
        elif op_type == OpType.EXP:
            return math.exp(values[0])
        elif op_type == OpType.SIN:
            return math.sin(values[0])
        elif op_type == OpType.COS:
            return math.cos(values[0])
        elif op_type == OpType.POWER:
            return values[0] ** values[1]
        else:
            # Cannot fold this op
            raise ValueError(f"Cannot fold op type: {op_type}")
