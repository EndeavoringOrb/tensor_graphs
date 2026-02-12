from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np


def conv2d_decomposition(inputs, attrs=None):
    """
    Decomposes Conv2D into Im2Col + Dot.

    Inputs:
        inputs[0]: Data [N, C_in, H, W]
        inputs[1]: Weight [C_out, C_in, kH, kW]
        inputs[2]: Bias [C_out] (Optional)
    Attrs:
        kernel_size, stride, padding
    """
    x = inputs[0]
    w = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None

    kernel_size = attrs["kernel_size"]
    stride = attrs["stride"]
    padding = attrs["padding"]

    N, C_in, H, W = x.shape

    # Handle both 4D (Standard Conv) and 2D (Linear/1x1) weights
    if len(w.shape) == 4:
        C_out, _, kH, kW = w.shape
    elif len(w.shape) == 2:
        C_out, _ = w.shape
        kH, kW = 1, 1
    else:
        raise ValueError(f"Conv2D expects 2D or 4D weights, got {w.shape}")

    # 1. Im2Col
    # [N, C_in, H, W] -> [N, C_in*kH*kW, H_out*W_out]
    col = TensorNode(
        OpType.IM2COL,
        x.dtype,
        [x],
        attrs={
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        },
    )

    # 2. Reshape Weight
    # [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
    # We need the shape tensor
    w_shape_node = TensorNode(
        OpType.CONSTANT,
        x.dtype,  # dummy
        [],
        attrs={"value": np.array([C_out, C_in * kH * kW], dtype=np.int32)},
    )
    w_flat = TensorNode(OpType.RESHAPE, w.dtype, [w, w_shape_node])

    # 3. Batch Matrix Multiplication
    # Output: [N, C_out, H_out, W_out]

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Flatten N and M (H_out*W_out)
    # Shape: [N*M, K]
    M = H_out * W_out
    col_flat_shape = TensorNode(
        OpType.CONSTANT,
        x.dtype,
        [],
        name="col_shape",
        attrs={"value": np.array([N * M, C_in * kH * kW], dtype=np.int32)},
    )

    col_perm = TensorNode(
        OpType.PERMUTE,
        col.dtype,
        [col],
        attrs={"dims": [0, 2, 1]},
    )  # [N, M, K]
    col_flat = TensorNode(
        OpType.RESHAPE, col.dtype, [col_perm, col_flat_shape]
    )  # [N*M, K]

    # Permute col_flat to [K, N*M] for dot product
    col_t = TensorNode(
        OpType.PERMUTE,
        col.dtype,
        [col_flat],
        attrs={"dims": [1, 0]},
    )

    # Dot(W, col_t) -> [C_out, K] @ [K, N*M] = [C_out, N*M]
    out_flat = TensorNode(OpType.DOT, x.dtype, [w_flat, col_t])

    # Reshape to [C_out, N, M]
    out_reshape1 = TensorNode(
        OpType.RESHAPE,
        out_flat.dtype,
        [
            out_flat,
            TensorNode(
                OpType.CONSTANT,
                x.dtype,
                [],
                attrs={"value": np.array([C_out, N, M], dtype=np.int32)},
            ),
        ],
    )

    # Permute to [N, C_out, M]
    out_perm = TensorNode(
        OpType.PERMUTE,
        out_reshape1.dtype,
        [out_reshape1],
        attrs={"dims": [1, 0, 2]},
    )

    # Reshape to [N, C_out, H_out, W_out]
    out_shape = TensorNode(
        OpType.CONSTANT,
        x.dtype,
        [],
        attrs={"value": np.array([N, C_out, H_out, W_out], dtype=np.int32)},
    )
    result = TensorNode(OpType.RESHAPE, out_perm.dtype, [out_perm, out_shape])

    # Bias Add
    if bias:
        # Bias is [C_out]. Need to broadcast to [N, C_out, H_out, W_out]
        # Reshape bias to [1, C_out, 1, 1]
        b_shape = TensorNode(
            OpType.CONSTANT,
            x.dtype,
            [],
            attrs={"value": np.array([1, C_out, 1, 1], dtype=np.int32)},
        )
        b_reshaped = TensorNode(OpType.RESHAPE, bias.dtype, [bias, b_shape])
        result = TensorNode(OpType.ADD, result.dtype, [result, b_reshaped])

    return result


register_reference_factory("Conv2D", conv2d_decomposition)
