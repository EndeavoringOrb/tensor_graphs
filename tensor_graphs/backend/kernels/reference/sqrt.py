import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.sqrt import sqrt_ref


@KernelRegistry.register(
    OpType.SQRT,
    [
        TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)
    ],  # Matches any rank/shape
)
def sqrt_generic(inputs, outputs, attrs):
    """
    Generic Square Root Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    np.sqrt(inputs[0], out=outputs[0])
