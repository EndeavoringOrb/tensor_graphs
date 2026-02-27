from .node import TensorNode
from ..ops.atomic_types import OpType
from typing import List, Set, Dict
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


class FactoringRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # Matches: (a * b) + (a * c) -> a * (b + c)
        if node.op_type == OpType.ADD and len(node.parents) == 2:
            mul1, mul2 = node.parents
            if mul1.op_type == OpType.MUL and mul2.op_type == OpType.MUL:
                if len(mul1.parents) == 2 and len(mul2.parents) == 2:
                    for i in range(2):
                        for j in range(2):
                            if mul1.parents[i] == mul2.parents[j]:
                                a = mul1.parents[i]
                                b = mul1.parents[1 - i]
                                c = mul2.parents[1 - j]
                                add_node = TensorNode(
                                    OpType.ADD, node.dtype, [b, c], shape=node.shape
                                )
                                return [
                                    TensorNode(
                                        OpType.MUL,
                                        node.dtype,
                                        [a, add_node],
                                        shape=node.shape,
                                    )
                                ]
        return []


class AssociativeRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        results = []
        if node.op_type in (OpType.ADD, OpType.MUL) and len(node.parents) == 2:
            a, b = node.parents

            # (x op y) op z -> x op (y op z)
            if a.op_type == node.op_type and len(a.parents) == 2:
                x, y = a.parents
                z = b
                new_inner = TensorNode(
                    node.op_type, node.dtype, [y, z], shape=node.shape
                )
                new_outer = TensorNode(
                    node.op_type, node.dtype, [x, new_inner], shape=node.shape
                )
                results.append(new_outer)

            # x op (y op z) -> (x op y) op z
            if b.op_type == node.op_type and len(b.parents) == 2:
                x = a
                y, z = b.parents
                new_inner = TensorNode(
                    node.op_type, node.dtype, [x, y], shape=node.shape
                )
                new_outer = TensorNode(
                    node.op_type, node.dtype, [new_inner, z], shape=node.shape
                )
                results.append(new_outer)

        return results


class DoubleNegationRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # -(-A) -> A
        if node.op_type == OpType.NEGATE and len(node.parents) == 1:
            inner = node.parents[0]
            if inner.op_type == OpType.NEGATE and len(inner.parents) == 1:
                return [inner.parents[0]]
        return []


class NegateAddRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # -(A + B) -> -A + -B
        if node.op_type == OpType.NEGATE and len(node.parents) == 1:
            inner = node.parents[0]
            if inner.op_type == OpType.ADD and len(inner.parents) == 2:
                a, b = inner.parents
                neg_a = TensorNode(OpType.NEGATE, node.dtype, [a], shape=node.shape)
                neg_b = TensorNode(OpType.NEGATE, node.dtype, [b], shape=node.shape)
                return [
                    TensorNode(OpType.ADD, node.dtype, [neg_a, neg_b], shape=node.shape)
                ]
        return []


class DivMulRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # (A / B) * B -> A
        if node.op_type == OpType.MUL and len(node.parents) == 2:
            a, b = node.parents
            if a.op_type == OpType.DIVIDE and len(a.parents) == 2 and a.parents[1] == b:
                return [a.parents[0]]
            if b.op_type == OpType.DIVIDE and len(b.parents) == 2 and b.parents[1] == a:
                return [b.parents[0]]
        return []


class DivAddRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # (A / C) + (B / C) -> (A + B) / C
        if node.op_type == OpType.ADD and len(node.parents) == 2:
            div1, div2 = node.parents
            if div1.op_type == OpType.DIVIDE and div2.op_type == OpType.DIVIDE:
                if len(div1.parents) == 2 and len(div2.parents) == 2:
                    if div1.parents[1] == div2.parents[1]:
                        a, c1 = div1.parents
                        b, c2 = div2.parents
                        add_node = TensorNode(
                            OpType.ADD, node.dtype, [a, b], shape=node.shape
                        )
                        return [
                            TensorNode(
                                OpType.DIVIDE,
                                node.dtype,
                                [add_node, c1],
                                shape=node.shape,
                            )
                        ]
        return []


class ExpAddRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # exp(A) * exp(B) -> exp(A + B)
        if node.op_type == OpType.MUL and len(node.parents) == 2:
            a, b = node.parents
            if a.op_type == OpType.EXP and b.op_type == OpType.EXP:
                if len(a.parents) == 1 and len(b.parents) == 1:
                    inner_a = a.parents[0]
                    inner_b = b.parents[0]
                    add_node = TensorNode(
                        OpType.ADD, node.dtype, [inner_a, inner_b], shape=node.shape
                    )
                    return [
                        TensorNode(OpType.EXP, node.dtype, [add_node], shape=node.shape)
                    ]
        return []


class ExpAddReverseRule(RewriteRule):
    def apply(self, node: TensorNode) -> List[TensorNode]:
        # exp(A + B) -> exp(A) * exp(B)
        if node.op_type == OpType.EXP and len(node.parents) == 1:
            inner = node.parents[0]
            if inner.op_type == OpType.ADD and len(inner.parents) == 2:
                a, b = inner.parents
                exp_a = TensorNode(OpType.EXP, node.dtype, [a], shape=node.shape)
                exp_b = TensorNode(OpType.EXP, node.dtype, [b], shape=node.shape)
                return [
                    TensorNode(OpType.MUL, node.dtype, [exp_a, exp_b], shape=node.shape)
                ]
        return []


def generate_all_equivalents(
    root: TensorNode, rules: List[RewriteRule], memo: Dict = {}
) -> Set[TensorNode]:
    equivalents = {root}
    worklist = [root]
    seen_hashes = {get_pattern_hash(root, memo)}

    while worklist:
        current = worklist.pop(0)

        for rule in rules:
            new_nodes = rule.apply(current)
            for new_node in new_nodes:
                new_hash = get_pattern_hash(new_node, memo)

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
