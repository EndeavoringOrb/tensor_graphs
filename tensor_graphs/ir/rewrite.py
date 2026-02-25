from .node import TensorNode
from ..ops.atomic_types import OpType
from typing import List, Set
from .hashing import get_pattern_hash
import numpy as np


class RewriteRule:
    def apply(self, node: TensorNode) -> List[TensorNode]:
        return []


class CommutativeRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        if node.op_type in (OpType.ADD, OpType.MUL) and len(node.parents) == 2:
            # Return a new node with swapped parents
            return [
                TensorNode(
                    op_type=node.op_type,
                    dtype=node.dtype,
                    parents=[node.parents[1], node.parents[0]],
                    shape=node.shape,
                )
            ]
        return []


class DistributiveRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # Matches: a * (b + c) -> (a * b) + (a * c)
        if node.op_type == OpType.MUL and len(node.parents) == 2:
            a, add_node = node.parents[0], node.parents[1]
            if add_node.op_type == OpType.ADD and len(add_node.parents) == 2:
                b, c = add_node.parents
                mul1 = TensorNode(OpType.MUL, node.dtype, [a, b], shape=node.shape)
                mul2 = TensorNode(OpType.MUL, node.dtype, [a, c], shape=node.shape)
                return [
                    TensorNode(OpType.ADD, node.dtype, [mul1, mul2], shape=node.shape)
                ]
        return []


def generate_all_equivalents(
    root: TensorNode, rules: List[RewriteRule]
) -> Set[TensorNode]:
    equivalents = {root}
    worklist = [root]
    seen_hashes = {get_pattern_hash(root)}

    while worklist:
        current = worklist.pop(0)

        for rule in rules:
            new_nodes = rule.apply(current)
            for new_node in new_nodes:
                new_hash = get_pattern_hash(new_node)

                # Check hash to prevent infinite loops (e.g., from Commutativity)
                if new_hash not in seen_hashes:
                    seen_hashes.add(new_hash)
                    equivalents.add(new_node)
                    worklist.append(new_node)

    return equivalents


def match_pattern(concrete_node, pattern_node, variables, binding) -> bool:
    # 1. Variable Binding
    if pattern_node in variables:
        if pattern_node in binding:
            # Variable was already bound, ensure it's the exact same node
            return binding[pattern_node] == concrete_node
        else:
            # Bind the variable
            binding[pattern_node] = concrete_node
            return True

    # 2. OpType matching
    if concrete_node.op_type != pattern_node.op_type and not (
        pattern_node.op_type == OpType.INPUT
    ):  # pattern_node.op_type == OpType.INPUT means wildcard
        return False

    # 3. Constant value matching
    if pattern_node.op_type == OpType.CONSTANT:
        return np.isclose(
            concrete_node.attrs.get("value"), pattern_node.attrs.get("value")
        )

    # 4. Topology matching
    if len(concrete_node.parents) != len(pattern_node.parents):
        return False

    for c_p, p_p in zip(concrete_node.parents, pattern_node.parents):
        if not match_pattern(c_p, p_p, variables, binding):
            return False

    return True
