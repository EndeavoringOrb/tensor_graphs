from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def tanh_decomposition(inputs, attrs=None):
    x = inputs[0]
    exp_x = TensorNode(OpType.EXP, x.dtype, [x])
    neg_x = TensorNode(OpType.NEGATE, x.dtype, [x])
    exp_neg_x = TensorNode(OpType.EXP, x.dtype, [neg_x])

    neg_exp_neg = TensorNode(OpType.NEGATE, x.dtype, [exp_neg_x])
    num = TensorNode(OpType.ADD, x.dtype, [exp_x, neg_exp_neg])
    den = TensorNode(OpType.ADD, x.dtype, [exp_x, exp_neg_x])

    return TensorNode(OpType.DIVIDE, x.dtype, [num, den])


register_reference_factory("Tanh", tanh_decomposition)
