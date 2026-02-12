from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def softmax_decomposition(inputs, attrs={"axis": -1}):
    x = inputs[0]
    axis = attrs["axis"]
    reduced_shape = list(x.shape)
    reduced_shape[axis] = 1

    x_max = TensorNode(
        OpType.MAX,
        x.dtype,
        [x],
        attrs={"axis": axis, "keepdims": True},
    )
    neg_max = TensorNode(OpType.NEGATE, x_max.dtype, [x_max])
    shifted = TensorNode(OpType.ADD, x.dtype, [x, neg_max])
    exps = TensorNode(OpType.EXP, x.dtype, [shifted])
    sum_exps = TensorNode(
        OpType.SUM,
        x.dtype,
        [exps],
        attrs={"axis": axis, "keepdims": True},
    )
    return TensorNode(OpType.DIVIDE, x.dtype, [exps, sum_exps])


register_reference_factory("Softmax", softmax_decomposition)
