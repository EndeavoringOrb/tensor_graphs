import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.negate import negate_ref


@KernelRegistry.register(
    OpType.NEGATE, [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)]
)
def negate_generic(inputs, attrs=None):
    """
    Generic Negate Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return -inputs[0]
