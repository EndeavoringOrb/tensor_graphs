from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from ...ir.dtypes import DType


def rope_2d_consecutive_decomposition(inputs, attrs={}):
    """
    RoPE with consecutive pair rotation (x[2i], x[2i+1]).
    Used by Flux
    """
    x, cos, sin = inputs

    # Check for static shape if possible, though TensorNode handles dynamic slices gracefully
    D = x.shape[-1] if x.shape else None

    # 1. Slice consecutive pairs: even and odd indices
    # x_even = x[..., 0::2]
    # x_odd  = x[..., 1::2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 2. Slice Cos/Sin (they are duplicated for pairs, so taking evens is sufficient)
    c = cos[..., 0::2]
    s = sin[..., 0::2]

    # 3. Apply rotation
    # out_even = x_even * c - x_odd * s
    # out_odd  = x_odd * c + x_even * s

    term1 = TensorNode(OpType.MUL, x.dtype, [x_even, c])
    term2 = TensorNode(OpType.MUL, x.dtype, [x_odd, s])

    neg_term2 = TensorNode(OpType.NEGATE, x.dtype, [term2])
    out_even = TensorNode(OpType.ADD, x.dtype, [term1, neg_term2])

    term3 = TensorNode(OpType.MUL, x.dtype, [x_odd, c])
    term4 = TensorNode(OpType.MUL, x.dtype, [x_even, s])
    out_odd = TensorNode(OpType.ADD, x.dtype, [term3, term4])

    # 4. Interleave results
    # We need to reshape to [..., D/2, 1] and concat to [..., D/2, 2], then reshape to [..., D]

    # We need to construct the shape tensor for the reshape.
    # Since we are in decomposition, we assume we can build the shape node.
    if x.shape is None:
        raise ValueError(
            "RoPE2DConsecutive decomposition requires known input shape for interleaving"
        )

    base_shape = list(x.shape)
    if base_shape[-1] is None:
        raise ValueError(
            "RoPE2DConsecutive decomposition requires known last dimension"
        )

    half_d = base_shape[-1] // 2

    # Shape: [..., half_d, 1]
    interleave_shape_val = np.array(base_shape[:-1] + [half_d, 1], dtype=np.int32)
    shape_node_1 = TensorNode(
        OpType.CONSTANT, DType.INT32, [], attrs={"value": interleave_shape_val}
    )

    even_reshaped = TensorNode(OpType.RESHAPE, x.dtype, [out_even, shape_node_1])
    odd_reshaped = TensorNode(OpType.RESHAPE, x.dtype, [out_odd, shape_node_1])

    # Concat along last axis -> [..., half_d, 2]
    stacked = TensorNode(
        OpType.CONCAT, x.dtype, [even_reshaped, odd_reshaped], attrs={"axis": -1}
    )

    # Reshape back to [..., D]
    shape_node_2 = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array(base_shape, dtype=np.int32)},
    )

    return TensorNode(OpType.RESHAPE, x.dtype, [stacked, shape_node_2])


register_reference_factory("RoPE2DConsecutive", rope_2d_consecutive_decomposition)
