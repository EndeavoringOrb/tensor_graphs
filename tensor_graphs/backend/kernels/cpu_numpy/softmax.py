# tensor_graphs/backend/kernels/cpu_numpy/softmax.py
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.fused.softmax import softmax_decomposition


@KernelRegistry.register(
    "Softmax",
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=softmax_decomposition,
)
def softmax_kernel(inputs, outputs, attrs):
    x = inputs[0]
    axis = attrs.get("axis", -1)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    outputs[0][:] = result


@KernelRegistry.register(
    "Softmax",
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=softmax_decomposition,
    inplace=True,
)
def softmax_inplace(inputs, outputs, attrs):
    x = inputs[0]
    out = outputs[0]
    axis = attrs.get("axis", -1)

    # In-place softmax leveraging intermediate array state
    x_max = np.max(x, axis=axis, keepdims=True)
    np.subtract(x, x_max, out=out)
    np.exp(out, out=out)
    sum_exp = np.sum(out, axis=axis, keepdims=True)
    np.divide(out, sum_exp, out=out)
