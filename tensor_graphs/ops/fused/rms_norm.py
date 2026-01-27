from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def rms_norm_decomposition(inputs, attrs=None):
    # (Implementation of atomic decomposition from original file)
    x, scale, eps = inputs
    sq = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "rmsnorm_sq")
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
    n = x.shape[axis] or 1
    n_const = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "n", attrs={"value": float(n)}
    )
    mean_sq = TensorNode(
        OpType.DIVIDE, sum_sq.shape, x.dtype, [sum_sq, n_const], "mean"
    )
    add_eps = TensorNode(OpType.ADD, mean_sq.shape, x.dtype, [mean_sq, eps], "add_eps")
    rsqrt = TensorNode(OpType.SQRT, add_eps.shape, x.dtype, [add_eps], "sqrt")
    one = TensorNode(OpType.CONSTANT, (1,), x.dtype, [], "one", attrs={"value": 1.0})
    inv_sqrt = TensorNode(OpType.DIVIDE, rsqrt.shape, x.dtype, [one, rsqrt], "inv_sqrt")
    norm = TensorNode(OpType.MUL, x.shape, x.dtype, [x, inv_sqrt], "norm")
    one_scale = TensorNode(
        OpType.ADD, scale.shape, scale.dtype, [one, scale], "1_plus_scale"
    )
    return TensorNode(OpType.MUL, x.shape, x.dtype, [norm, one_scale], "rmsnorm_out")


register_reference_factory("RMSNorm", rms_norm_decomposition)


def rms_norm_ref(inputs, name="rmsnorm", attrs=None):
    return TensorNode(
        "RMSNorm",
        inputs[0].shape,
        inputs[0].dtype,
        inputs,
        name,
        attrs=attrs if attrs is not None else {},
    )
