"""
File: tensor_graphs/ops/fused/norm.py
"""

from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode, ConstantNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class RMSNorm(CompositeOp):
    op_type = "RMSNorm"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        # Inputs: x, scale, eps
        x, scale, eps = inputs

        # 1. x^2
        sq = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "rmsnorm_sq")

        # 2. Mean = Sum(sq) / N
        # Use attributes for axis instead of input node
        axis = attrs.get("axis", -1) if attrs else -1
        sum_shape = list(x.shape)
        sum_shape[axis] = 1
        sum_sq = TensorNode(
            OpType.SUM,
            tuple(sum_shape),
            x.dtype,
            [sq],
            "rmsnorm_sum",
            attrs={"axis": axis, "keepdims": True},
        )

        n = x.shape[axis]
        if n is None:
            raise ValueError(
                f"RMSNorm requires static shape on normalization axis {axis}"
            )
        n_elements = ConstantNode(
            OpType.CONSTANT, (1,), x.dtype, [], "n_elements", value=float(n)
        )
        mean_sq = TensorNode(
            OpType.DIVIDE, sum_sq.shape, x.dtype, [sum_sq, n_elements], "rmsnorm_mean"
        )

        # 3. Add Epsilon
        add_eps = TensorNode(
            OpType.ADD, mean_sq.shape, x.dtype, [mean_sq, eps], "add_eps"
        )

        # 4. Rsqrt
        rsqrt = TensorNode(OpType.SQRT, add_eps.shape, x.dtype, [add_eps], "sqrt")
        one = ConstantNode(OpType.CONSTANT, (1,), x.dtype, [], "one_const", value=1.0)
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
