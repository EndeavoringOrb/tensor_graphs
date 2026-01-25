import hashlib
import json
from typing import Dict, Any, List
from .node import TensorNode
from ..ops.atomic import OpType


class GraphHasher:
    def __init__(self, root: TensorNode):
        self.root = root
        self.input_ids: Dict[TensorNode, int] = {}
        self.next_input_id = 0
        self.hashes: Dict[TensorNode, str] = {}

    def _get_input_id(self, node: TensorNode) -> int:
        if node not in self.input_ids:
            self.input_ids[node] = self.next_input_id
            self.next_input_id += 1
        return self.input_ids[node]

    def _hash_value(self, val: Any) -> str:
        # Simple value hasher
        return hashlib.sha256(str(val).encode("utf-8")).hexdigest()

    def compute_hash(self, node: TensorNode) -> str:
        if node in self.hashes:
            return self.hashes[node]

        # Base case: Input
        if node.op_type == OpType.INPUT:
            # We treat inputs as variables. Their identity is determined by their discovery order.
            # This makes the hash invariant to variable names (x, y vs a, b)
            # but preserves structure (x+x vs x+y).
            idx = self._get_input_id(node)
            node_hash = self._hash_value(f"Input_{idx}_{node.dtype.value}_{node.shape}")
            self.hashes[node] = node_hash
            return node_hash

        # Base case: Constant
        if node.op_type == OpType.CONSTANT:
            # For constants, the value matters.
            # We round floats to avoid precision jitter if needed, but for now str() is okay.
            val = node.attrs.get("value")
            val_str = str(val)
            node_hash = self._hash_value(
                f"Const_{node.dtype.value}_{node.shape}_{val_str}"
            )
            self.hashes[node] = node_hash
            return node_hash

        # Recursive case: Ops
        parent_hashes = [self.compute_hash(p) for p in node.parents]

        # Handle Commutativity
        if node.op_type in (OpType.ADD, OpType.MUL):
            parent_hashes.sort()

        # Construct the signature string
        # "OpType|DType|Shape|AttrHash|ParentHash1,ParentHash2,..."

        # Serialize attributes (sort keys for consistency)
        attrs_str = ""
        if node.attrs:
            try:
                attrs_str = json.dumps(node.attrs, sort_keys=True, default=str)
            except TypeError:
                attrs_str = str(node.attrs)

        raw_str = (
            f"{node.op_type}|{node.dtype.value}|{node.shape}|{attrs_str}|"
            + ",".join(parent_hashes)
        )

        node_hash = self._hash_value(raw_str)
        self.hashes[node] = node_hash
        return node_hash


def compute_structural_hash(root: TensorNode) -> str:
    """
    Computes a structural hash of the graph rooted at 'root'.
    """
    hasher = GraphHasher(root)
    return hasher.compute_hash(root)
