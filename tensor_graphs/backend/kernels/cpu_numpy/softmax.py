import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.softmax import softmax_decomposition


# --- Softmax ---
@KernelRegistry.register(
    "Softmax",
    [TensorSignature(DType.FP32, shape=None)],
    reference_factory=softmax_decomposition,
)
def softmax_kernel(inputs, attrs=None, outputs=None):
    x = inputs[0]
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    result = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    if outputs is not None:
        outputs[0][:] = result
        return outputs[0]
    return result
