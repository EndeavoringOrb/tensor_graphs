import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.rms_norm import rms_norm_decomposition


# --- RMSNorm ---
@KernelRegistry.register(
    "RMSNorm",
    [
        TensorSignature(DType.FP32, shape=None),
        TensorSignature(DType.FP32, (None,)),
        TensorSignature(DType.FP32, (1,)),
    ],
    reference_factory=rms_norm_decomposition,
)
def rms_norm_kernel(inputs, outputs, attrs):
    x, scale, eps = inputs
    eps_val = eps[0]
    var = np.mean(x**2, axis=-1, keepdims=True)
    x_norm = x * (1.0 / np.sqrt(var + eps_val))
    result = x_norm * scale
    outputs[0][:] = result
