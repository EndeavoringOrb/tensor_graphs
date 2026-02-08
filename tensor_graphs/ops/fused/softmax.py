from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory

def softmax(x, axis=-1, name=None):
    return TensorNode(
        "Softmax",
        x.dtype,
        [x],
        name=name or f"{x.name}_softmax",
        attrs={"axis": axis},
    )

def softmax_decomposition(inputs, attrs=None):
    x = inputs[0]
    axis = attrs.get("axis", -1) if attrs else -1
    reduced_shape = list(x.shape)
    reduced_shape[axis] = 1

    x_max = TensorNode(
        OpType.MAX,
        x.dtype,
        [x],
        name="max",
        attrs={"axis": axis, "keepdims": True},
    )
    neg_max = TensorNode(OpType.NEGATE, x_max.dtype, [x_max], name="neg_max")
    shifted = TensorNode(OpType.ADD, x.dtype, [x, neg_max], name="shifted")
    exps = TensorNode(OpType.EXP, x.dtype, [shifted], name="exps")
    sum_exps = TensorNode(
        OpType.SUM,
        x.dtype,
        [exps],
        name="sum",
        attrs={"axis": axis, "keepdims": True},
    )
    return TensorNode(OpType.DIVIDE, x.dtype, [exps, sum_exps], name="softmax_out")


register_reference_factory("Softmax", softmax_decomposition)
