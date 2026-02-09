from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def rope(x, cos, sin, name=None):
    return TensorNode("RoPE", x.dtype, [x, cos, sin], name=name or f"{x.name}_rope")


# Register atomic decomposition logic
def rope_decomposition(inputs, attrs=None):
    x, cos, sin = inputs
    D = x.shape[-1]
    half_d = D // 2

    x1_node = x[..., :half_d]
    x2_node = x[..., half_d:]
    neg_x2_node = TensorNode(
        OpType.NEGATE, x2_node.dtype, [x2_node], name=f"{x.name}_neg_x2"
    )

    rotated_node = TensorNode(
        OpType.CONCAT,
        x.dtype,
        [neg_x2_node, x1_node],
        name=f"{x.name}_rotated",
        attrs={"axis": -1},
    )

    term1 = TensorNode(OpType.MUL, x.dtype, [x, cos], name=f"{x.name}_t1")
    term2 = TensorNode(OpType.MUL, x.dtype, [rotated_node, sin], name=f"{x.name}_t2")

    return TensorNode(OpType.ADD, x.dtype, [term1, term2], name=f"{x.name}_rope_out")


register_reference_factory("RoPE", rope_decomposition)
