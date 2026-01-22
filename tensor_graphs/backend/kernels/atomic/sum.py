"""
File: tensor_graphs/backend/kernels/atomic/sum.py
"""
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType

@KernelRegistry.register(OpType.SUM, [
    TensorSignature(DType.FP32, shape=None),
    TensorSignature(DType.INT32, shape=(1,))
])
def sum_generic(inputs):
    data = inputs[0]
    axis = int(inputs[1][0])
    # Keepdims is essential for correct broadcasting in norms, 
    # but our atomic signature doesn't pass flags yet. 
    # We default to keepdims=True for consistency with "Reduction" behavior in graphs often.
    return np.sum(data, axis=axis, keepdims=True)