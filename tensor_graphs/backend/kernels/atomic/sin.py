import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType

@KernelRegistry.register(OpType.SIN, [
    TensorSignature(DType.FP32, shape=None)  # Matches any rank/shape
])
def sin_generic(inputs):
    """
    Generic Sine Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.sin(inputs[0])