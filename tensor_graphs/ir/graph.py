from typing import List, Set
from .node import TensorNode


def topological_sort(root: TensorNode) -> List[TensorNode]:
    """
    Returns a linear execution order for the graph ending at 'root'.
    """
    visited: Set[TensorNode] = set()
    order: List[TensorNode] = []

    def _visit(node: TensorNode):
        if node in visited:
            return
        visited.add(node)
        for parent in node.parents:
            _visit(parent)
        order.append(node)

    _visit(root)
    return order


def get_inputs(root: TensorNode) -> List[TensorNode]:
    """Returns all leaf nodes (OpType.INPUT) required for this graph."""
    topo = topological_sort(root)
    return [n for n in topo if n.op_type == "Input"]
