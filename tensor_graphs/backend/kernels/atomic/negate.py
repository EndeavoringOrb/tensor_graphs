import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType

@KernelRegistry.register(OpType.NEGATE, [
    TensorSignature(DType.FP32, shape=None)  # Matches any rank/shape
])
def negate_generic(inputs):
    """
    Generic Negate Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return -inputs[0]