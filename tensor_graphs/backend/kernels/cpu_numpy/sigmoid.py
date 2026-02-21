# tensor_graphs/backend/kernels/cpu_numpy/sigmoid.py
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.fused.sigmoid import sigmoid_decomposition


@KernelRegistry.register(
    "Sigmoid",
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=sigmoid_decomposition,
)
def sigmoid_kernel(inputs, outputs, attrs):
    x = inputs[0]
    result = np.exp(-np.logaddexp(0, -x))
    outputs[0][:] = result


@KernelRegistry.register(
    "Sigmoid",
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    backend=Backend.CPU_NUMPY,
    reference_factory=sigmoid_decomposition,
    inplace=True,
)
def sigmoid_inplace(inputs, outputs, attrs):
    x = inputs[0]
    out = outputs[0]
    # In-place sequence: 1 / (1 + exp(-x))
    np.negative(x, out=out)
    np.exp(out, out=out)
    np.add(out, 1.0, out=out)
    np.divide(1.0, out, out=out)
