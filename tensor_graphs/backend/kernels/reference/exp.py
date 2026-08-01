import numpy as np

from ....backend.registry import KernelRegistry
from ....ir.dtypes import Backend, DType, TensorSignature
from ....ops.atomic.exp import exp_ref
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.EXP,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    reference_factory=exp_ref,
)
def exp_generic(inputs, outputs, attrs):
    """
    Generic Exponential Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    np.exp(inputs[0], out=outputs[0])
