import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature, Backend
from ....ops.atomic_types import OpType
from ....ops.atomic.permute import permute_ref


@KernelRegistry.register(
    OpType.PERMUTE,
    [
        TensorSignature(
            DType.FP32, shape=None, backend=Backend.CPU_NUMPY
        ),  # Matches any rank/shape
        TensorSignature(
            DType.INT32, shape=(None,), backend=Backend.CPU_NUMPY
        ),  # Permutation order vector
    ],
    backend=Backend.CPU_NUMPY,
    target_dtype=DType.FP32,
    reference_factory=permute_ref,
)
def permute_generic(inputs, attrs=None):
    """
    Generic Permute/Transpose Implementation.
    inputs[0]: Data tensor (Any Rank)
    inputs[1]: Permutation axes (1D tensor of integers)
    """
    data = inputs[0]
    perm = inputs[1]

    # Convert numpy array of dims to tuple of ints
    perm_tuple = tuple(perm.astype(int))
    return np.transpose(data, axes=perm_tuple)
