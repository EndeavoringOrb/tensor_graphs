import hashlib
import json
import numpy as np
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
        h = _hash_string(
            f"INPUT|{node.name}|{node.dtype.value}|{node.shape}|{node.backend.value}"
        )
        memo[node] = h
        return h

    if node.op_type == OpType.CONSTANT:
        val = node.attrs.get("value")
        # Include a hash of the value for all constants to avoid collisions
        # for non-scalar constants (like shape tensors).
        if val is not None:
            if isinstance(val, np.ndarray):
                val_content = str(val.tolist())
            else:
                val_content = str(val)
            h = _hash_string(f"CONST|{node.dtype.value}|{node.shape}|{val_content}")
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
