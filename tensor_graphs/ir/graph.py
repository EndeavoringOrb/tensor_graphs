from .node import TensorNode
from .dtypes import DType
from ..ops.atomic_types import OpType
from .buffer import StorageType
import numpy as np
from typing import Set, Dict, List
from .dtypes import Backend
import json
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

    # --- Core Nodes ---

    def input(self, name, shape, dtype=DType.FP32):
        node = TensorNode(
            OpType.INPUT, dtype, [], shape, name, storage_type=StorageType.TRANSIENT
        )
        self.inputs[name] = node
        return node

    def const(self, value, dtype=None):
        if dtype is None:
            if isinstance(value, float) or (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], float)
            ):
                dtype = DType.FP32
            elif isinstance(value, int) or (
                isinstance(value, list) and len(value) > 0 and isinstance(value[0], int)
            ):
                dtype = DType.INT32
            elif hasattr(value, "dtype"):
                # Handle numpy arrays and numpy scalar types
                dt = str(value.dtype)
                if "float32" in dt or "float64" in dt:
                    dtype = DType.FP32
                elif "int" in dt:
                    dtype = DType.INT32
                elif "bool" in dt:
                    dtype = DType.BOOL
                else:
                    dtype = DType.FP32
            else:
                raise ValueError(
                    f"Could not infer dtype for constant with type {type(value)}"
                )
        if not isinstance(value, np.ndarray):
            # If it's a scalar, wrap it; if it's a list, convert it
            value = np.array(
                value, dtype=np.float32 if dtype == DType.FP32 else np.int32
            )
            if value.ndim == 0:
                value = np.array([value])
        node = TensorNode(
            OpType.CONSTANT,
            dtype,
            [],
            value.shape,
            name=self._next_name("const"),
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

    # --- Math Ops ---

    def add(self, a, b):
        return TensorNode(OpType.ADD, a.dtype, [a, b], name=self._next_name("add"))

    def mul(self, a, b):
        return TensorNode(OpType.MUL, a.dtype, [a, b], name=self._next_name("mul"))

    def divide(self, a, b):
        return TensorNode(OpType.DIVIDE, a.dtype, [a, b], name=self._next_name("div"))

    def dot(self, a, b):
        return TensorNode(OpType.DOT, a.dtype, [a, b], name=self._next_name("dot"))

    def sqrt(self, a):
        return TensorNode(OpType.SQRT, a.dtype, [a], name=self._next_name("sqrt"))

    def sin(self, a):
        return TensorNode(OpType.SIN, a.dtype, [a], name=self._next_name("sin"))

    def cos(self, a):
        return TensorNode(OpType.COS, a.dtype, [a], name=self._next_name("cos"))

    def exp(self, a):
        return TensorNode(OpType.EXP, a.dtype, [a], name=self._next_name("exp"))

    def negate(self, a):
        return TensorNode(OpType.NEGATE, a.dtype, [a], name=self._next_name("neg"))

    def power(self, a, b):
        return TensorNode(OpType.POWER, a.dtype, [a, b], name=self._next_name("pow"))

    # --- Reduction Ops ---

    def sum(self, a, axis=None, keepdims=True):
        return TensorNode(
            OpType.SUM,
            a.dtype,
            [a],
            name=self._next_name("sum"),
            attrs={"axis": axis, "keepdims": keepdims},
        )

    def max(self, a, axis=None, keepdims=True):
        return TensorNode(
            OpType.MAX,
            a.dtype,
            [a],
            name=self._next_name("max"),
            attrs={"axis": axis, "keepdims": keepdims},
        )

    # --- Manipulation Ops ---

    def reshape(self, a, shape_node):
        return TensorNode(
            OpType.RESHAPE, a.dtype, [a, shape_node], name=self._next_name("reshape")
        )

    def permute(self, a, dims: List[int]):
        return TensorNode(
            OpType.PERMUTE,
            a.dtype,
            [a],
            name=self._next_name("permute"),
            attrs={"dims": dims},
        )

    def slice(self, a, starts, ends, steps=None):
        attrs = {"starts": starts, "ends": ends}
        if steps:
            attrs["steps"] = steps
        return TensorNode(
            OpType.SLICE, a.dtype, [a], name=self._next_name("slice"), attrs=attrs
        )

    def concat(self, tensors: List[TensorNode], axis: int):
        return TensorNode(
            OpType.CONCAT,
            tensors[0].dtype,
            tensors,
            name=self._next_name("concat"),
            attrs={"axis": axis},
        )

    def cast(self, a, dtype: DType):
        return TensorNode(
            OpType.CAST, dtype, [a], name=self._next_name("cast"), attrs={"to": dtype}
        )

    def repeat(self, a, repeats: int, axis: int):
        return TensorNode(
            OpType.REPEAT,
            a.dtype,
            [a],
            name=self._next_name("repeat"),
            attrs={"repeats": repeats, "axis": axis},
        )

    def arange(self, start, stop, step):
        return TensorNode(
            OpType.ARANGE,
            start.dtype,
            [start, stop, step],
            name=self._next_name("arange"),
        )

    def triu(self, a, k=0):
        return TensorNode(
            OpType.TRIU, a.dtype, [a], name=self._next_name("triu"), attrs={"k": k}
        )

    def gather(self, data, indices):
        return TensorNode(
            OpType.GATHER, data.dtype, [data, indices], name=self._next_name("gather")
        )

    def fill(self, value_node, shape_node):
        return TensorNode(
            OpType.FILL,
            value_node.dtype,
            [value_node, shape_node],
            name=self._next_name("fill"),
        )

    def where(self, condition, x, y):
        return TensorNode(
            OpType.WHERE, x.dtype, [condition, x, y], name=self._next_name("where")
        )

    # --- Fused Ops ---

    def rms_norm(self, x, scale, eps):
        return TensorNode(
            "RMSNorm", x.dtype, [x, scale, eps], name=self._next_name("rmsnorm")
        )

    def softmax(self, x, axis=-1):
        return TensorNode(
            "Softmax",
            x.dtype,
            [x],
            name=self._next_name("softmax"),
            attrs={"axis": axis},
        )

    def gelu(self, x):
        return TensorNode("GELU", x.dtype, [x], name=self._next_name("gelu"))

    def rope(self, x, cos, sin):
        return TensorNode("RoPE", x.dtype, [x, cos, sin], name=self._next_name("rope"))

    def tanh(self, x):
        return TensorNode("Tanh", x.dtype, [x], name=self._next_name("tanh"))

    def silu(self, x):
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def sigmoid(self, x):
        return TensorNode("Sigmoid", x.dtype, [x], name=self._next_name("sigmoid"))

    def conv2d(self, x, weight, bias=None, kernel_size=3, stride=1, padding=1):
        inputs = [x, weight]
        if bias:
            inputs.append(bias)
        return TensorNode(
            "Conv2D",
            x.dtype,
            inputs,
            name=self._next_name("conv2d"),
            attrs={"kernel_size": kernel_size, "stride": stride, "padding": padding},
        )

    def group_norm(self, x, weight, bias, num_groups, eps=1e-6):
        return TensorNode(
            "GroupNorm",
            x.dtype,
            [x, weight, bias],
            name=self._next_name("groupnorm"),
            attrs={"num_groups": num_groups, "eps": eps},
        )

    def upsample_nearest_2x(self, x):
        return TensorNode(
            "Upsample2x",
            x.dtype,
            [x],
            name=self._next_name("upsample"),
            attrs={"scale": 2},
        )
