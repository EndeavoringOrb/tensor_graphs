import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.im2col import im2col_ref


def _im2col_indices(x_shape, kernel_size, stride, padding):
    # N, C, H, W
    N, C, H, W = x_shape
    kH, kW = kernel_size, kernel_size

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Create grid for top-left corner of patches
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(H_out), W_out)

    j0 = np.tile(np.arange(kW), kH * C)
    j1 = stride * np.tile(np.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return i, j


@KernelRegistry.register(
    OpType.IM2COL,
    [
        TensorSignature(
            DType.FP32, shape=(None, None, None, None), backend=Backend.CPU_NUMPY
        )
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=im2col_ref,
)
def im2col_np(inputs, outputs, attrs):
    x = inputs[0]
    kernel_size = attrs["kernel_size"]
    stride = attrs["stride"]
    padding = attrs["padding"]

    N, C, H, W = x.shape
    kH, kW = kernel_size, kernel_size

    # Pad input
    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        x_padded = x

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # (N, C, H, W) -> (N, C*kH*kW, H_out*W_out)
    # We use advanced indexing

    i, j = _im2col_indices((N, C, H, W), kernel_size, stride, padding)

    # k: indices for channel/patch dim (C*kH*kW)
    # i, j: indices for spatial output dim

    # We iterate over N
    cols = []
    for n in range(N):
        col = np.zeros((C * kH * kW, H_out * W_out), dtype=x.dtype)

        for c in range(C):
            # Channel offset in output
            c_offset = c * kH * kW
            # Indices for this channel
            c_i = i[c_offset : c_offset + kH * kW, :]
            c_j = j[c_offset : c_offset + kH * kW, :]

            # Gather from x_padded[n, c, :, :]
            col[c_offset : c_offset + kH * kW, :] = x_padded[n, c, c_i, c_j]

        cols.append(col)

    result = np.stack(cols, axis=0)  # (N, C*kH*kW, H_out*W_out)
    outputs[0][:] = result
