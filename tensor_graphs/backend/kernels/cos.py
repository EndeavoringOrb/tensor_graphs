import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

@KernelRegistry.register(OpType.COS, [
    TensorSignature(DType.FP32, shape=None)  # Matches any rank/shape
])
def cos_generic(inputs):
    """
    Generic Cosine Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.cos(inputs[0])