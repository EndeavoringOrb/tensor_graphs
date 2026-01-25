import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.EXP, [TensorSignature(DType.FP32, shape=None)]  # Matches any rank/shape
)
def exp_generic(inputs, attrs=None):
    """
    Generic Exponential Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.exp(inputs[0])
