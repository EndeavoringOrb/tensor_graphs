"""
File: tensor_graphs/ops/fused/math.py
"""

from typing import List
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class FusedMulAdd(CompositeOp):
    op_type = "FusedMulAdd"

    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
        # Expected: A, B, C -> (A * B) + C
        if len(inputs) != 3:
            raise ValueError("FusedMulAdd requires 3 inputs")

        a, b, c = inputs
        mul_node = TensorNode(
            OpType.MUL, a.shape, a.dtype, [a, b], f"decomp_mul_{a.name}"
        )
        add_node = TensorNode(
            OpType.ADD, a.shape, a.dtype, [mul_node, c], f"decomp_fma_{a.name}"
        )
        return add_node
