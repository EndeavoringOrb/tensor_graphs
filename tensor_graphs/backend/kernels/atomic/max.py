"""
File: tensor_graphs/backend/kernels/atomic/max.py
"""
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType

@KernelRegistry.register(OpType.MAX, [
    TensorSignature(DType.FP32, shape=None),
    TensorSignature(DType.INT32, shape=(1,))
])
def max_generic(inputs):
    data = inputs[0]
    axis = int(inputs[1][0])
    return np.max(data, axis=axis, keepdims=True)