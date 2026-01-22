import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.SQRT, [TensorSignature(DType.FP32, shape=None)]  # Matches any rank/shape
)
def sqrt_generic(inputs, attrs=None):
    """
    Generic Square Root Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.sqrt(inputs[0])
