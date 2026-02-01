from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic_types import OpType
from ..registry import register_reference_factory


# Register atomic decomposition logic
def rope_decomposition(inputs, attrs=None):
    x, cos, sin = inputs
    D = x.shape[-1]
    half_d = D // 2 if D is not None else None

    x1_node = x[..., :half_d]
    x2_node = x[..., half_d:]
    neg_x2_node = TensorNode(
        OpType.NEGATE, x2_node.shape, x2_node.dtype, [x2_node], f"{x.name}_neg_x2"
    )

    rotated_node = TensorNode(
        OpType.CONCAT,
        x.shape,
        x.dtype,
        [neg_x2_node, x1_node],
        f"{x.name}_rotated",
        attrs={"axis": -1},
    )

    term1 = TensorNode(OpType.MUL, x.shape, x.dtype, [x, cos], f"{x.name}_t1")
    term2 = TensorNode(
        OpType.MUL, x.shape, x.dtype, [rotated_node, sin], f"{x.name}_t2"
    )

    return TensorNode(
        OpType.ADD, x.shape, x.dtype, [term1, term2], f"{x.name}_rope_out"
    )


register_reference_factory("RoPE", rope_decomposition)


# Main entry point returns High-Level Node
def rope_ref(inputs, attrs=None):
    return TensorNode("RoPE", inputs[0].shape, inputs[0].dtype, inputs, "rope")
