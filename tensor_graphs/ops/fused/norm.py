"""
File: tensor_graphs/ops/fused/norm.py
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ...ir.node import TensorNode
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

        # Scale mean by 1/N
        n = x.shape[axis] or 1
        n_elements = TensorNode(
            OpType.CONSTANT, (1,), x.dtype, [], "n_elements", attrs={"value": float(n)}
        )
        mean_sq = TensorNode(
            OpType.DIVIDE, sum_sq.shape, x.dtype, [sum_sq, n_elements], "rmsnorm_mean"
        )

        # inv_sqrt = 1 / sqrt(mean + eps)
        add_eps = TensorNode(
            OpType.ADD, mean_sq.shape, x.dtype, [mean_sq, eps], "add_eps"
        )
        rsqrt = TensorNode(OpType.SQRT, add_eps.shape, x.dtype, [add_eps], "sqrt")
        one = TensorNode(
            OpType.CONSTANT, (1,), x.dtype, [], "one_const", attrs={"value": 1.0}
        )
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

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        # Case 1: 2D Input, last dim matches scale
        x = np.random.randn(2, 4).astype(np.float32)
        scale = np.random.randn(4).astype(np.float32)
        eps = np.array([1e-6], dtype=np.float32)
        return [([x, scale, eps], {"axis": -1})]
