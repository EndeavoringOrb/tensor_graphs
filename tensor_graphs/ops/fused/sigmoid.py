from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def sigmoid_decomposition(inputs, attrs=None):
    """
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    x = inputs[0]

    neg_x = TensorNode(OpType.NEGATE, x.dtype, [x])
    exp_neg = TensorNode(OpType.EXP, neg_x.dtype, [neg_x])

    one = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 1.0})
    one_plus = TensorNode(OpType.ADD, exp_neg.dtype, [one, exp_neg])

    return TensorNode(OpType.DIVIDE, one.dtype, [one, one_plus])


register_reference_factory("Sigmoid", sigmoid_decomposition)
