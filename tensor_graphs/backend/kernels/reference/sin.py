import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.SIN,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)
    ],  # Matches any rank/shape
)
def sin_generic(inputs, outputs, attrs):
    """
    Generic Sine Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    np.sin(inputs[0], out=outputs[0])
