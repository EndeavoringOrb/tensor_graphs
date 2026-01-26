import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, Backend, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.cos import cos_ref


@KernelRegistry.register(
    OpType.COS,
    [TensorSignature(DType.FP32, shape=None, backend=Backend.CPU_NUMPY)],
    reference_factory=cos_ref,
)
def cos_generic(inputs, attrs=None):
    """
    Generic Cosine Implementation.
    inputs[0]: Data tensor (Any Rank)
    """
    return np.cos(inputs[0])
