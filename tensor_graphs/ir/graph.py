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


def normalize_graph(root: TensorNode):
    """
    Recursively normalizes the graph in-place by sorting commutative inputs (Add, Mul)
    based on their structural hashes.
    """
    from .hashing import compute_structural_hash
    from ..ops.atomic_types import OpType

    visited = set()

    def _normalize(node: TensorNode):
        if node in visited:
            return
        visited.add(node)

        # Bottom-up normalization
        for parent in node.parents:
            _normalize(parent)

        if node.op_type in (OpType.ADD, OpType.MUL):
            # Sort parents by their structural hash
            # Note: We need to compute hashes *after* parents are normalized
            node.parents.sort(key=lambda n: compute_structural_hash(n))

    _normalize(root)


def find_subgraph(large_graph: TensorNode, subgraph: TensorNode) -> List[TensorNode]:
    """
    Finds all nodes in large_graph that are roots of a subgraph structurally
    identical to 'subgraph'.
    """
    from .hashing import compute_structural_hash
    from .graph import topological_sort

    sub_hash = compute_structural_hash(subgraph)
    matches = []

    # Important: Both graphs should be normalized for this to be reliable
    # We don't normalize here to avoid side effects, assuming user has done it
    # or expects exact structural match as is.

    for node in topological_sort(large_graph):
        if compute_structural_hash(node) == sub_hash:
            matches.append(node)

    return matches
