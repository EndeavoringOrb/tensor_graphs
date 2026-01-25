import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.rms_norm import rms_norm_ref


# --- RMSNorm ---
@KernelRegistry.register(
    "RMSNorm",
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (1,)),
    ],
    reference_factory=rms_norm_ref,
)
def rms_norm_kernel(inputs, attrs=None):
    x, scale, eps = inputs
    eps_val = eps[0]
    var = np.mean(x**2, axis=-1, keepdims=True)
    x_norm = x * (1.0 / np.sqrt(var + eps_val))
    return x_norm * (1.0 + scale)
