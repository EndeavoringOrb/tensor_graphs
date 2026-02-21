# tensor_graphs/backend/kernels/cpu_numpy/rms_norm.py
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.fused.rms_norm import rms_norm_decomposition


@KernelRegistry.register(
    "RMSNorm",
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=(None,), backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=(1,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=rms_norm_decomposition,
)
def rms_norm_kernel(inputs, outputs, attrs):
    x, scale, eps = inputs
    eps_val = eps[0]
    var = np.mean(x**2, axis=-1, keepdims=True)
    x_norm = x * (1.0 / np.sqrt(var + eps_val))
    result = x_norm * scale
    outputs[0][:] = result


@KernelRegistry.register(
    "RMSNorm",
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=(None,), backend=Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, shape=(1,), backend=Backend.CPU_NUMPY),
    ],
    backend=Backend.CPU_NUMPY,
    reference_factory=rms_norm_decomposition,
    inplace=True,
)
def rms_norm_inplace(inputs, outputs, attrs):
    x, scale, eps = inputs
    out = outputs[0]
    eps_val = eps[0]

    # In-place RMS norm approximation
    var = np.mean(x**2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps_val)
    np.multiply(x, inv_std, out=out)
    np.multiply(out, scale, out=out)
