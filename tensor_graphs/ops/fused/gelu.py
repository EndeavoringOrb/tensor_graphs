from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from .tanh import tanh_decomposition


def gelu_decomposition(inputs, attrs=None):
    # (Implementation from original file)
    x = inputs[0]
    c_cube = TensorNode(
        OpType.CONSTANT, x.dtype, [], name="c1", attrs={"value": 0.044715}
    )
    c_sqrt = TensorNode(
        OpType.CONSTANT, x.dtype, [], name="c2", attrs={"value": np.sqrt(2 / np.pi)}
    )
    c_half = TensorNode(OpType.CONSTANT, x.dtype, [], name="c3", attrs={"value": 0.5})
    c_one = TensorNode(OpType.CONSTANT, x.dtype, [], name="c4", attrs={"value": 1.0})

    x2 = TensorNode(OpType.MUL, x.dtype, [x, x], name="x2")
    x3 = TensorNode(OpType.MUL, x.dtype, [x2, x], name="x3")
    term1 = TensorNode(OpType.MUL, x.dtype, [x3, c_cube], name="t1")
    term2 = TensorNode(OpType.ADD, x.dtype, [x, term1], name="t2")
    inner = TensorNode(OpType.MUL, x.dtype, [term2, c_sqrt], name="inner")

    tanh_node = tanh_decomposition([inner])

    one_plus = TensorNode(OpType.ADD, x.dtype, [c_one, tanh_node], name="one_plus")
    half_x = TensorNode(OpType.MUL, x.dtype, [x, c_half], name="half_x")
    return TensorNode(OpType.MUL, x.dtype, [half_x, one_plus], name="gelu_out")


register_reference_factory("GELU", gelu_decomposition)
