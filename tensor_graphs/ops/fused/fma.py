from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def fma_decomposition(inputs, attrs=None):
    a, b, c = inputs
    mul = TensorNode(OpType.MUL, a.shape, a.dtype, [a, b], "mul")
    return TensorNode(OpType.ADD, a.shape, a.dtype, [mul, c], "fma_out")


register_reference_factory("FusedMulAdd", fma_decomposition)


def fused_mul_add_ref(inputs, attrs=None):
    return TensorNode("FusedMulAdd", inputs[0].shape, inputs[0].dtype, inputs, "fma")
