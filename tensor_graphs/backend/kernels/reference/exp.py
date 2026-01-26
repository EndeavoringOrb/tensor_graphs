import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, Backend, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.exp import exp_ref


@KernelRegistry.register(
    OpType.EXP,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    reference_factory=exp_ref,
)
def exp_generic(inputs, attrs=None):
    """
    Generic Exponential Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.exp(inputs[0])
