import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.rope_2d_consecutive import rope_2d_consecutive_decomposition


@KernelRegistry.register(
    "RoPE2DConsecutive",
    [
        TensorSignature(DType.FP32, shape=None),  # x
        TensorSignature(DType.FP32, shape=None),  # cos
        TensorSignature(DType.FP32, shape=None),  # sin
    ],
    reference_factory=rope_2d_consecutive_decomposition,
)
def rope_2d_consecutive_kernel(inputs, outputs, attrs):
    x, cos, sin = inputs

    # Allocate output if needed
    if outputs[0] is None:
        outputs[0] = np.empty_like(x)
    x_out = outputs[0]

    # Consecutive pairs logic: (i, i+1)
    # x_even: indices 0, 2, 4...
    # x_odd:  indices 1, 3, 5...

    # Slicing creates views, math ops create temps
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Cos/Sin input is assumed to be duplicated [c0, c0, c1, c1...]
    # So we take stride 2 to get values for the pairs
    c = cos[..., 0::2]
    s = sin[..., 0::2]

    # out_even = x_even * c - x_odd * s
    # out_odd  = x_odd * c + x_even * s

    # Write directly to output view using slice assignment
    x_out[..., 0::2] = x_even * c - x_odd * s
    x_out[..., 1::2] = x_odd * c + x_even * s
