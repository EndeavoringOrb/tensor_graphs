"""
File: tensor_graphs/ops/fused/norm.py
"""

from typing import List
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class RMSNorm(CompositeOp):
    op_type = "RMSNorm"

    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
        # Inputs: x, scale, eps
        x, scale, eps = inputs

        # 1. x^2
        sq = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "rmsnorm_sq")

        # 2. Mean = Sum(sq) / N
        axis_val = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "axis_last")
        sum_sq = TensorNode(OpType.SUM, (1,), x.dtype, [sq, axis_val], "rmsnorm_sum")
        n_elements = TensorNode(OpType.INPUT, (1,), x.dtype, [], "n_elements")
        mean_sq = TensorNode(
            OpType.DIVIDE, sum_sq.shape, x.dtype, [sum_sq, n_elements], "rmsnorm_mean"
        )

        # 3. Add Epsilon
        add_eps = TensorNode(
            OpType.ADD, mean_sq.shape, x.dtype, [mean_sq, eps], "add_eps"
        )

        # 4. Rsqrt
        rsqrt = TensorNode(OpType.SQRT, add_eps.shape, x.dtype, [add_eps], "sqrt")
        one = TensorNode(OpType.INPUT, (1,), x.dtype, [], "one_const")
        inv_sqrt = TensorNode(
            OpType.DIVIDE, rsqrt.shape, x.dtype, [one, rsqrt], "inv_sqrt"
        )

        # 5. Normalize
        norm = TensorNode(OpType.MUL, x.shape, x.dtype, [x, inv_sqrt], "norm_pre_scale")

        # 6. Scale (1 + scale) [Gemma 3 Specific]
        one_scale = TensorNode(
            OpType.ADD, scale.shape, scale.dtype, [one, scale], "1_plus_scale"
        )
        out = TensorNode(OpType.MUL, x.shape, x.dtype, [norm, one_scale], "rmsnorm_out")

        return out
