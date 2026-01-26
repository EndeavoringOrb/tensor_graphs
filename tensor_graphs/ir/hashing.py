import hashlib
import json
from ..ops.atomic_types import OpType


def _hash_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_structural_hash(node, memo=None) -> str:
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]

    # 1. Base Cases
    if node.op_type == OpType.INPUT:
        # For planning, inputs with same signature are identical
        h = _hash_string(f"INPUT|{node.dtype.value}|{node.shape}|{node.backend.value}")
        memo[node] = h
        return h

    if node.op_type == OpType.CONSTANT:
        if node.shape and (
            len(node.shape) == 0 or (len(node.shape) == 1 and node.shape[0] == 1)
        ):
            # Include value for scalars (often params like axis)
            val = node.attrs.get("value", "?")
            h = _hash_string(f"CONST|{val}")
        else:
            h = _hash_string(f"CONST|{node.dtype.value}|{node.shape}")
        memo[node] = h
        return h

    # 2. Recurse
    parent_hashes = [get_structural_hash(p, memo) for p in node.parents]

    # 3. Canonicalize Commutative Ops
    if node.op_type in (OpType.ADD, OpType.MUL):
        parent_hashes.sort()

    # 4. Attributes
    attrs_str = json.dumps(node.attrs, sort_keys=True, default=str)

    # 5. Compute
    content = f"{node.op_type}|{node.dtype.value}|{node.shape}|{node.backend.value}|{attrs_str}|{','.join(parent_hashes)}"
    h = _hash_string(content)

    memo[node] = h
    return h


def compute_structural_hash(node) -> str:
    return get_structural_hash(node)
