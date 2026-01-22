"""
File: tensor_graphs/backend/kernels/fused/math.py
"""
from ...registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.fused.math import FusedMulAdd

# Register using the CompositeOp name
@KernelRegistry.register(FusedMulAdd.op_type, [
    TensorSignature(DType.FP32, (None,)), 
    TensorSignature(DType.FP32, (None,)),
    TensorSignature(DType.FP32, (None,))
])
def fma_generic(inputs):
    return inputs[0] * inputs[1] + inputs[2]