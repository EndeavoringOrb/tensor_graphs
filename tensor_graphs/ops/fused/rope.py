from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


# Register atomic decomposition logic
def rope_decomposition(inputs, attrs={}):
    x, cos, sin = inputs
    D = x.shape[-1]
    half_d = D // 2

    x1_node = x[..., :half_d]
    x2_node = x[..., half_d:]
    neg_x2_node = TensorNode(OpType.NEGATE, x2_node.dtype, [x2_node])

    rotated_node = TensorNode(
        OpType.CONCAT,
        x.dtype,
        [neg_x2_node, x1_node],
        attrs={"axis": -1},
    )

    term1 = TensorNode(OpType.MUL, x.dtype, [x, cos])
    term2 = TensorNode(OpType.MUL, x.dtype, [rotated_node, sin])

    return TensorNode(OpType.ADD, x.dtype, [term1, term2])


register_reference_factory("RoPE", rope_decomposition)
