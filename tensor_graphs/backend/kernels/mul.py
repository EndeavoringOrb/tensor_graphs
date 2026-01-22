import numpy as np
from ...backend.registry import KernelRegistry
from ...ir.dtypes import DType, TensorSignature
from ...ops.atomic import OpType

@KernelRegistry.register(OpType.MUL, [
    TensorSignature(DType.FP32, (None,)), 
    TensorSignature(DType.FP32, (None,))
])
def mul_generic_vector(inputs):
    return inputs[0] * inputs[1]

@KernelRegistry.register(OpType.MUL, [
    TensorSignature(DType.FP32, (1,)), 
    TensorSignature(DType.FP32, (1,))
])
def mul_scalar(inputs):
    return inputs[0] * inputs[1]

@KernelRegistry.register(OpType.MUL, [
    TensorSignature(DType.FP32, (32,)), 
    TensorSignature(DType.FP32, (32,))
])
def mul_vec32(inputs):
    return inputs[0] * inputs[1]