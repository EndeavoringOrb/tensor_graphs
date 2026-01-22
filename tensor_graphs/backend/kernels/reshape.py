import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

@KernelRegistry.register(OpType.RESHAPE, [
    TensorSignature(DType.FP32, shape=None),     # Matches any rank/shape
    TensorSignature(DType.INT32, shape=(None,))  # Shape tensor is always 1D
])
def reshape_generic(inputs):
    """
    Generic Reshape Implementation.
    inputs[0]: Data tensor (Any Rank)
    inputs[1]: Target shape (1D tensor of integers)
    """
    data = inputs[0]
    target_shape = inputs[1]
    
    # Convert numpy array of shape dims to tuple of ints
    shape_tuple = tuple(target_shape.astype(int))
    return np.reshape(data, shape_tuple)