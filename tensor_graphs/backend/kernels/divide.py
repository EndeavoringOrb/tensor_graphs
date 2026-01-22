import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

# --- 1. Generic Vector ---
@KernelRegistry.register(OpType.DIVIDE, [
    TensorSignature(DType.FP32, (None,)), 
    TensorSignature(DType.FP32, (None,))
])
def div_generic_vector(inputs):
    return inputs[0] / inputs[1]

# --- 2. Generic Matrix ---
@KernelRegistry.register(OpType.DIVIDE, [
    TensorSignature(DType.FP32, (None, None)), 
    TensorSignature(DType.FP32, (None, None))
])
def div_generic_matrix(inputs):
    return inputs[0] / inputs[1]

# --- 3. Scalar Broadcast (Scalar / Matrix) ---
@KernelRegistry.register(OpType.DIVIDE, [
    TensorSignature(DType.FP32, (1,)), 
    TensorSignature(DType.FP32, (None, None))
])
def div_scalar_broadcast(inputs):
    # Scalar / Matrix
    return inputs[0] / inputs[1]

# --- 4. Matrix / Scalar ---
@KernelRegistry.register(OpType.DIVIDE, [
    TensorSignature(DType.FP32, (None, None)), 
    TensorSignature(DType.FP32, (1,))
])
def div_matrix_scalar(inputs):
    # Matrix / Scalar
    return inputs[0] / inputs[1]