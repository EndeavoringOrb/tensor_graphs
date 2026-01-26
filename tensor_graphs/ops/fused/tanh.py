from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def tanh_decomposition(inputs, attrs=None):
    x = inputs[0]
    exp_x = TensorNode(OpType.EXP, x.shape, x.dtype, [x], "exp_x")
    neg_x = TensorNode(OpType.NEGATE, x.shape, x.dtype, [x], "neg_x")
    exp_neg_x = TensorNode(OpType.EXP, x.shape, x.dtype, [neg_x], "exp_neg_x")

    neg_exp_neg = TensorNode(
        OpType.NEGATE, x.shape, x.dtype, [exp_neg_x], "neg_exp_neg"
    )
    num = TensorNode(OpType.ADD, x.shape, x.dtype, [exp_x, neg_exp_neg], "num")
    den = TensorNode(OpType.ADD, x.shape, x.dtype, [exp_x, exp_neg_x], "den")

    return TensorNode(OpType.DIVIDE, x.shape, x.dtype, [num, den], "tanh_out")


register_reference_factory("Tanh", tanh_decomposition)


def tanh_ref(inputs, attrs=None):
    return TensorNode("Tanh", inputs[0].shape, inputs[0].dtype, inputs, "tanh")
