from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def silu_decomposition(inputs, attrs={}):
    """
    SiLU(x) = x * sigmoid(x)
    """
    x = inputs[0]

    # Sigmoid Node
    sig = TensorNode("Sigmoid", x.dtype, [x], attrs=attrs)

    # Mul
    return TensorNode(OpType.MUL, x.dtype, [x, sig])


register_reference_factory("SiLU", silu_decomposition)
