from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from .tanh import tanh_decomposition


def gelu_decomposition(inputs, attrs=None):
    x = inputs[0]
    c_cube = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 0.044715})
    c_sqrt = TensorNode(
        OpType.CONSTANT, x.dtype, [], attrs={"value": np.sqrt(2 / np.pi)}
    )
    c_half = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 0.5})
    c_one = TensorNode(OpType.CONSTANT, x.dtype, [], attrs={"value": 1.0})

    x2 = TensorNode(OpType.MUL, x.dtype, [x, x])
    x3 = TensorNode(OpType.MUL, x.dtype, [x2, x])
    term1 = TensorNode(OpType.MUL, x.dtype, [x3, c_cube])
    term2 = TensorNode(OpType.ADD, x.dtype, [x, term1])
    inner = TensorNode(OpType.MUL, x.dtype, [term2, c_sqrt])

    tanh_node = tanh_decomposition([inner])

    one_plus = TensorNode(OpType.ADD, x.dtype, [c_one, tanh_node])
    half_x = TensorNode(OpType.MUL, x.dtype, [x, c_half])
    return TensorNode(OpType.MUL, x.dtype, [half_x, one_plus])


register_reference_factory("GELU", gelu_decomposition)
