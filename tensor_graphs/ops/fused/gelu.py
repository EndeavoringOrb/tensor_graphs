from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from .tanh import tanh_ref


def gelu_decomposition(inputs, attrs=None):
    # (Implementation from original file)
    x = inputs[0]
    c_cube = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "c1", attrs={"value": 0.044715}
    )
    c_sqrt = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "c2", attrs={"value": np.sqrt(2 / np.pi)}
    )
    c_half = TensorNode(OpType.CONSTANT, (1,), x.dtype, [], "c3", attrs={"value": 0.5})
    c_one = TensorNode(OpType.CONSTANT, (1,), x.dtype, [], "c4", attrs={"value": 1.0})

    x2 = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "x2")
    x3 = TensorNode(OpType.MUL, x.shape, x.dtype, [x2, x], "x3")
    term1 = TensorNode(OpType.MUL, x.shape, x.dtype, [x3, c_cube], "t1")
    term2 = TensorNode(OpType.ADD, x.shape, x.dtype, [x, term1], "t2")
    inner = TensorNode(OpType.MUL, x.shape, x.dtype, [term2, c_sqrt], "inner")

    # Check if we should use high-level tanh or atomic
    # For robust decomposition, use atomic factory directly if needed, or rely on recursion
    # Here we invoke the decomposition logic for tanh manually to be safe, or just return Tanh node
    # Let's return a Tanh Node and let the Planner handle IT.
    tanh_node = TensorNode("Tanh", x.shape, x.dtype, [inner], "tanh_inner")

    one_plus = TensorNode(OpType.ADD, x.shape, x.dtype, [c_one, tanh_node], "one_plus")
    half_x = TensorNode(OpType.MUL, x.shape, x.dtype, [x, c_half], "half_x")
    return TensorNode(OpType.MUL, x.shape, x.dtype, [half_x, one_plus], "gelu_out")


register_reference_factory("GELU", gelu_decomposition)


def gelu_ref(inputs, attrs=None):
    return TensorNode("GELU", inputs[0].shape, inputs[0].dtype, inputs, "gelu")
