from typing import List, Dict, Any, Optional
import numpy as np
from ...ir.node import TensorNode
from ..atomic_types import OpType
from .tanh import tanh_ref


def gelu_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = inputs[0]
    c_cube = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "c_0.044", attrs={"value": 0.044715}
    )
    c_sqrt = TensorNode(
        OpType.CONSTANT,
        (1,),
        x.dtype,
        [],
        "c_sqrt_2_pi",
        attrs={"value": np.sqrt(2 / np.pi)},
    )
    c_half = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "c_0.5", attrs={"value": 0.5}
    )
    c_one = TensorNode(
        OpType.CONSTANT, (1,), x.dtype, [], "c_1.0", attrs={"value": 1.0}
    )

    # x^3
    x2 = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "x2")
    x3 = TensorNode(OpType.MUL, x.shape, x.dtype, [x2, x], "x3")

    # inner
    term1 = TensorNode(OpType.MUL, x.shape, x.dtype, [x3, c_cube], "term1")
    term2 = TensorNode(OpType.ADD, x.shape, x.dtype, [x, term1], "term2")
    inner = TensorNode(OpType.MUL, x.shape, x.dtype, [term2, c_sqrt], "inner")

    # tanh (using the factory defined above)
    tanh_node = tanh_ref([inner], attrs)

    # outer
    one_plus = TensorNode(OpType.ADD, x.shape, x.dtype, [c_one, tanh_node], "one_plus")
    half_x = TensorNode(OpType.MUL, x.shape, x.dtype, [x, c_half], "half_x")

    return TensorNode(OpType.MUL, x.shape, x.dtype, [half_x, one_plus], "gelu_out")
