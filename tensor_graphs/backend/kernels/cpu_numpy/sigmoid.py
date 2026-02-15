import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.sigmoid import sigmoid_decomposition

@KernelRegistry.register(
    "Sigmoid",
    [TensorSignature(DType.FP32, shape=None)],
    reference_factory=sigmoid_decomposition,
)
def sigmoid_kernel(inputs, outputs, attrs):
    x = inputs[0]
    result = np.exp(-np.logaddexp(0, -x))
    outputs[0][:] = result