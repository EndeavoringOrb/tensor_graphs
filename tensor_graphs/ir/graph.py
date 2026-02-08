from typing import List, Set, Dict
from .node import TensorNode
from .dtypes import DType, Backend
import json
import numpy as np
from enum import Enum


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


# --- Serialization Helpers ---


class GraphEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating) and not isinstance(o, bool):
            return float(o)
        if isinstance(o, np.integer) and not isinstance(o, bool):
            return int(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, TensorNode):
            return str(o)
        # Shapes are now plain int/None; sympy no longer used
        return super().default(o)


def graph_to_json(root: TensorNode) -> str:
    """
    Serializes a graph to a JSON string representation.
    Format: Flat list of nodes in topological order.
    """
    nodes = topological_sort(root)
    # Map node instances to temporary IDs (0, 1, 2...)
    node_to_id = {node: i for i, node in enumerate(nodes)}

    serialized_nodes = []
    for node in nodes:
        parent_ids = [node_to_id[p] for p in node.parents]
        serialized_nodes.append(
            {
                "id": node_to_id[node],
                "op_type": node.op_type,
                "name": node.name,
                "shape": node.shape,
                "dtype": node.dtype.value,
                "backend": node.backend.value,
                "parents": parent_ids,
                "attrs": node.attrs,
            }
        )

    return json.dumps(serialized_nodes, cls=GraphEncoder)


def graph_from_json(json_str: str) -> TensorNode:
    """
    Reconstructs a graph from a JSON string. Returns the root node.
    """
    if not json_str:
        raise ValueError("Empty JSON string for graph reconstruction")

    data = json.loads(json_str)
    if not isinstance(data, list) or not data:
        raise ValueError("Invalid JSON format: expected non-empty list of nodes")

    id_to_node: Dict[int, TensorNode] = {}

    for node_data in data:
        parents = [id_to_node[pid] for pid in node_data["parents"]]

        # Reconstruct DType and Backend Enums
        dtype = DType(node_data["dtype"])
        backend = Backend(node_data["backend"])

        # Handle shape (convert list back to tuple)
        shape = tuple(node_data["shape"]) if node_data["shape"] is not None else ()

        node = TensorNode(
            op_type=node_data["op_type"],
            shape=shape,
            dtype=dtype,
            parents=parents,
            name=node_data["name"],
            attrs=node_data.get("attrs", {}),
            backend=backend,
        )
        id_to_node[node_data["id"]] = node

    # The last node in the topological list is the root
    return id_to_node[data[-1]["id"]]

class GraphBuilder:
    def __init__(self):
        self.params = {}
        self.inputs = {}
        self._count = 0

    def _next_name(self, op_name):
        self._count += 1
        return f"{op_name}_{self._count}"
    
    def input(self, name, shape, dtype=DType.FP32):
        node = TensorNode(
            OpType.INPUT, dtype, [], shape, name, storage_type=StorageType.TRANSIENT
        )
        self.inputs[name] = node
        return node

    def constant(self, value, shape, dtype, name):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        node = TensorNode(
            OpType.CONSTANT,
            dtype,
            [],
            shape,
            name,
            attrs={"value": value},
            storage_type=StorageType.PERSISTENT,
        )
        return node

    def param(self, name, shape, dtype=DType.FP32):
        node = TensorNode(
            OpType.INPUT, dtype, [], shape, name, storage_type=StorageType.PERSISTENT
        )
        self.params[name] = node
        return node