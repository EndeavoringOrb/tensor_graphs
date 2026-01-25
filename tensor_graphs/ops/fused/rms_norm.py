from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def rms_norm_ref(
    inputs: List[TensorNode], name="rmsnorm_out", attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for RMSNorm.
    Inputs: x, scale, eps
    """
    x, scale, eps = inputs

    # 1. x^2
    sq = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "rmsnorm_sq")

    # 2. Mean = Sum(sq) / N
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
    add_eps = TensorNode(OpType.ADD, mean_sq.shape, x.dtype, [mean_sq, eps], "add_eps")
    rsqrt = TensorNode(OpType.SQRT, add_eps.shape, x.dtype, [add_eps], "sqrt")
    one = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "one_const", attrs={"value": 1.0}
    )
    inv_sqrt = TensorNode(OpType.DIVIDE, rsqrt.shape, x.dtype, [one, rsqrt], "inv_sqrt")

    # 5. Normalize
    norm = TensorNode(OpType.MUL, x.shape, x.dtype, [x, inv_sqrt], "norm_pre_scale")

    # 6. Scale (1 + scale) [Gemma 3 Specific]
    one_scale = TensorNode(
        OpType.ADD, scale.shape, scale.dtype, [one, scale], "1_plus_scale"
    )
    out = TensorNode(OpType.MUL, x.shape, x.dtype, [norm, one_scale], name)

    return out
