"""
File: tensor_graphs/optim/fusion.py
"""

from ..ir.node import TensorNode
from ..ops.atomic import OpType
from ..ops.fused.math import FusedMulAdd


def fuse_graph(node: TensorNode, memo=None) -> TensorNode:
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]

    new_parents = [fuse_graph(p, memo) for p in node.parents]

    if node.op_type == OpType.ADD:
        lhs, rhs = new_parents
        if lhs.op_type == OpType.MUL:
            fused_node = TensorNode(
                op_type=FusedMulAdd.op_type,  # Use the fused op type definition
                shape=node.shape,
                dtype=node.dtype,
                parents=[*lhs.parents, rhs],
                name=f"fused_{node.name}",
            )
            memo[node] = fused_node
            return fused_node

    if new_parents == node.parents:
        memo[node] = node
        return node

    new_node = TensorNode(
        node.op_type, node.shape, node.dtype, new_parents, name=node.name
    )
    memo[node] = new_node
    return new_node
