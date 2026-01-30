import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.negate import negate_ref


@KernelRegistry.register(
    OpType.NEGATE, [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)]
)
def negate_generic(inputs, attrs=None, outputs=None):
    """
    Generic Negate Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    if outputs is None:
        return -inputs[0]
    np.negative(inputs[0], out=outputs[0])
    return outputs[0]
