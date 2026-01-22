"""
File: tensor_graphs/backend/kernels/atomic/tanh.py
"""
import numpy as np
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType

@KernelRegistry.register(OpType.TANH, [
    TensorSignature(DType.FP32, shape=None)
])
def tanh_generic(inputs):
    return np.tanh(inputs[0])