import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.rope import rope_decomposition


# --- RoPE ---
@KernelRegistry.register(
    "RoPE",
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, shape=None),
    ],
    reference_factory=rope_decomposition,
)
def rope_kernel(inputs, attrs=None, outputs=None):
    x, cos, sin = inputs
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = np.concatenate((-x2, x1), axis=-1)
    result = (x * cos) + (rotated * sin)
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
