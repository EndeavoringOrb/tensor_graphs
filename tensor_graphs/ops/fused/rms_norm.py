from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory

def rms_norm(x, scale, eps, name=None):
    return TensorNode(
        "RMSNorm",
        x.dtype,
        [x, scale, eps],
        name=name or f"{x.name}_rmsnorm"
    )

def rms_norm_decomposition(inputs, attrs=None):
    x, scale, eps = inputs
    if x.shape is None:
        raise ValueError(
            f"Decomposition of RMSNorm failed: Node '{x.name}' has no inferred shape. "
            "Ensure ShapeInference.infer() is called before expanding fused operations."
        )
    sq = TensorNode(OpType.MUL, x.dtype, [x, x], name="rmsnorm_sq")
    axis = attrs.get("axis", -1) if attrs else -1
    sum_sq = TensorNode(
        OpType.SUM,
        x.dtype,
        [sq],
        name="rmsnorm_sum",
        attrs={"axis": axis, "keepdims": True},
    )
    n = x.shape[axis]
    n_const = TensorNode(
        OpType.CONSTANT, x.dtype, [], name="n", attrs={"value": float(n)}
    )
    mean_sq = TensorNode(OpType.DIVIDE, x.dtype, [sum_sq, n_const], name="mean")
    add_eps = TensorNode(OpType.ADD, x.dtype, [mean_sq, eps], name="add_eps")
    rsqrt = TensorNode(OpType.SQRT, x.dtype, [add_eps], name="sqrt")
    one = TensorNode(OpType.CONSTANT, x.dtype, [], name="one", attrs={"value": 1.0})
    inv_sqrt = TensorNode(OpType.DIVIDE, x.dtype, [one, rsqrt], name="inv_sqrt")
    norm = TensorNode(OpType.MUL, x.dtype, [x, inv_sqrt], name="norm")
    one_scale = TensorNode(OpType.ADD, scale.dtype, [one, scale], name="1_plus_scale")
    return TensorNode(OpType.MUL, x.dtype, [norm, one_scale], name="rmsnorm_out")


register_reference_factory("RMSNorm", rms_norm_decomposition)
