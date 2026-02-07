import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType


@KernelRegistry.register(
    OpType.NEGATE, [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)]
)
def negate_generic(inputs, outputs, attrs):
    """
    Generic Negate Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    np.negative(inputs[0], out=outputs[0])
