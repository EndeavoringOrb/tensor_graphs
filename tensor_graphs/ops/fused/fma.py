from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def fma_decomposition(inputs, attrs=None):
    a, b, c = inputs
    mul = TensorNode(OpType.MUL, a.dtype, [a, b])
    return TensorNode(OpType.ADD, a.dtype, [mul, c])


register_reference_factory("FusedMulAdd", fma_decomposition)
