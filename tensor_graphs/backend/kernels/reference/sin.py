import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.sin import sin_ref


@KernelRegistry.register(
    OpType.SIN,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)
    ],  # Matches any rank/shape
)
def sin_generic(inputs, attrs=None, outputs=None):
    """
    Generic Sine Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    if outputs is None:
        return np.sin(inputs[0])
    np.sin(inputs[0], out=outputs[0])
    return outputs[0]
