import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

@KernelRegistry.register(OpType.PERMUTE, [
    TensorSignature(DType.FP32, shape=None),     # Matches any rank/shape
    TensorSignature(DType.INT32, shape=(None,))  # Permutation order vector
])
def permute_generic(inputs):
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