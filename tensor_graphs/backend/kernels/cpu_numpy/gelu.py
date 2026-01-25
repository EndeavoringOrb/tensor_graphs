import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.gelu import gelu_ref


# --- GELU ---
@KernelRegistry.register(
    "GELU", [TensorSignature(DType.FP32, shape=None)], reference_factory=gelu_ref
)
def gelu_kernel(inputs, attrs=None):
    x = inputs[0]
    c1 = np.sqrt(2.0 / np.pi)
    c2 = 0.044715
    inner = c1 * (x + c2 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))
