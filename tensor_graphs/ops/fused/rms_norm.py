from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def rms_norm_decomposition(inputs, attrs=None):
    x, scale, eps = inputs
    if x.shape is None:
        raise ValueError(
            f"Decomposition of RMSNorm failed: Node '{x.name}' has no inferred shape. "
            "Ensure GraphPropagator.infer_shapes() is called before expanding fused operations."
        )
    sq = TensorNode(OpType.MUL, x.dtype, [x, x])
    axis = attrs.get("axis", -1) if attrs else -1
    sum_sq = TensorNode(
        OpType.SUM,
        x.dtype,
        [sq],
        attrs={"axis": axis, "keepdims": True},
    )
    n = x.shape[axis]
    n_const = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": float(n)})
    mean_sq = TensorNode(OpType.DIVIDE, x.dtype, [sum_sq, n_const])
    add_eps = TensorNode(OpType.ADD, x.dtype, [mean_sq, eps])
    rsqrt = TensorNode(OpType.SQRT, x.dtype, [add_eps])
    one = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 1.0})
    inv_sqrt = TensorNode(OpType.DIVIDE, x.dtype, [one, rsqrt])
    norm = TensorNode(OpType.MUL, x.dtype, [x, inv_sqrt])
    one_scale = TensorNode(OpType.ADD, scale.dtype, [one, scale])
    return TensorNode(OpType.MUL, x.dtype, [norm, one_scale])


register_reference_factory("RMSNorm", rms_norm_decomposition)
