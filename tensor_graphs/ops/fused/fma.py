from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def fma(x, y, z):
    return TensorNode(
        "FusedMulAdd",
        x.dtype,
        [x, y, z],
        name=f"{x.name}_fma",
    )


def fma_decomposition(inputs, attrs=None):
    a, b, c = inputs
    mul = TensorNode(OpType.MUL, a.dtype, [a, b], name="mul")
    return TensorNode(OpType.ADD, a.dtype, [mul, c], name="fma_out")


register_reference_factory("FusedMulAdd", fma_decomposition)
