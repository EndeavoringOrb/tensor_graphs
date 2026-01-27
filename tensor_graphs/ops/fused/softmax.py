from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def softmax_decomposition(inputs, attrs=None):
    x = inputs[0]
    axis = attrs.get("axis", -1) if attrs else -1
    reduced_shape = list(x.shape)
    reduced_shape[axis] = 1

    x_max = TensorNode(
        OpType.MAX,
        tuple(reduced_shape),
        x.dtype,
        [x],
        "max",
        attrs={"axis": axis, "keepdims": True},
    )
    neg_max = TensorNode(OpType.NEGATE, x_max.shape, x_max.dtype, [x_max], "neg_max")
    shifted = TensorNode(OpType.ADD, x.shape, x.dtype, [x, neg_max], "shifted")
    exps = TensorNode(OpType.EXP, x.shape, x.dtype, [shifted], "exps")
    sum_exps = TensorNode(
        OpType.SUM,
        tuple(reduced_shape),
        x.dtype,
        [exps],
        "sum",
        attrs={"axis": axis, "keepdims": True},
    )
    return TensorNode(OpType.DIVIDE, x.shape, x.dtype, [exps, sum_exps], "softmax_out")


register_reference_factory("Softmax", softmax_decomposition)


def softmax_ref(inputs, attrs=None):
    return TensorNode(
        "Softmax",
        inputs[0].shape,
        inputs[0].dtype,
        inputs,
        "softmax",
        attrs=attrs if attrs is not None else {},
    )
