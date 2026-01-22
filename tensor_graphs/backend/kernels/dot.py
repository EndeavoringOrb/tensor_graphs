import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

# Generic Matrix Multiplication
@KernelRegistry.register(OpType.DOT, [
    TensorSignature(DType.FP32, (None, None)), 
    TensorSignature(DType.FP32, (None, None))
])
def dot_generic(inputs):
    return np.matmul(inputs[0], inputs[1])

# Tiny Matrix Optimization
@KernelRegistry.register(OpType.DOT, [
    TensorSignature(DType.FP32, (2, 2)), 
    TensorSignature(DType.FP32, (2, 2))
])
def dot_2x2_optimized(inputs):
    return np.matmul(inputs[0], inputs[1])