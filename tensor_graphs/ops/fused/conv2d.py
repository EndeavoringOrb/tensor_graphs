from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory
import numpy as np
from ...ir.dtypes import DType


def conv2d_decomposition(inputs, attrs=None):
    x = inputs[0]
    w = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None

    kernel_size = attrs["kernel_size"]
    stride = attrs["stride"]
    padding = attrs["padding"]

    # Use weight dimensions for C_out and K (input_channels * k * k)
    if len(w.shape) == 4:
        C_out = w.shape[0]
        K = w.shape[1] * w.shape[2] * w.shape[3]
    elif len(w.shape) == 2:
        C_out = w.shape[0]
        K = w.shape[1]
    else:
        raise ValueError(f"Conv2D expects 2D or 4D weights, got {w.shape}")

    # 1. Im2Col: [N, C_in, H, W] -> [N, K, M]
    col = TensorNode(
        OpType.IM2COL,
        x.dtype,
        [x],
        attrs={"kernel_size": kernel_size, "stride": stride, "padding": padding},
    )

    # 2. Reshape Weight: [C_out, K]
    # Use -1 to allow the second dimension to adapt to the weight size
    w_shape_node = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array([int(C_out), -1], dtype=np.int32)},
    )
    w_flat = TensorNode(OpType.RESHAPE, w.dtype, [w, w_shape_node])

    # 3. Prepare Im2Col for Dot Product: [K, N*M]
    # First transpose [N, K, M] -> [K, N, M]
    col_perm = TensorNode(OpType.PERMUTE, col.dtype, [col], attrs={"dims": [1, 0, 2]})

    # Reshape to [K, N*M] (K must match weight's K)
    col_t_shape = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={"value": np.array([int(K), -1], dtype=np.int32)},
    )
    col_t = TensorNode(OpType.RESHAPE, col.dtype, [col_perm, col_t_shape])

    # 4. Dot Product: [C_out, K] @ [K, N*M] -> [C_out, N*M]
    out_flat = TensorNode(OpType.DOT, x.dtype, [w_flat, col_t])

    # 5. Final Reshaping back to NCHW
    # Need N, H_out, W_out. Since we don't have a Shape op yet,
    # we derive them from x.shape but use -1 for the spatial dim M.
    N = x.shape[0] if x.shape else 1

    # Reshape [C_out, N*M] -> [C_out, N, M]
    out_reshape1 = TensorNode(
        OpType.RESHAPE,
        out_flat.dtype,
        [
            out_flat,
            TensorNode(
                OpType.CONSTANT,
                DType.INT32,
                [],
                attrs={"value": np.array([int(C_out), int(N), -1], dtype=np.int32)},
            ),
        ],
    )

    # Permute [C_out, N, M] -> [N, C_out, M]
    out_perm = TensorNode(
        OpType.PERMUTE, out_reshape1.dtype, [out_reshape1], attrs={"dims": [1, 0, 2]}
    )

    # Reshape [N, C_out, M] -> [N, C_out, H_out, W_out]
    # We calculate H_out/W_out for the final 4D representation
    H, W = x.shape[2], x.shape[3]
    if H is not None and W is not None:
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
    else:
        H_out, W_out = -1, -1  # Fallback for dynamic

    final_shape = TensorNode(
        OpType.CONSTANT,
        DType.INT32,
        [],
        attrs={
            "value": np.array(
                [int(N), int(C_out), int(H_out), int(W_out)], dtype=np.int32
            )
        },
    )
    result = TensorNode(OpType.RESHAPE, out_perm.dtype, [out_perm, final_shape])

    if bias:
        b_shape = TensorNode(
            OpType.CONSTANT,
            DType.INT32,
            [],
            attrs={"value": np.array([1, int(C_out), 1, 1], dtype=np.int32)},
        )
        result = TensorNode(
            OpType.ADD,
            result.dtype,
            [result, TensorNode(OpType.RESHAPE, bias.dtype, [bias, b_shape])],
        )

    return result


register_reference_factory("Conv2D", conv2d_decomposition)
