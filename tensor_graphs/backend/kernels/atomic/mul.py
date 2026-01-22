import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, TensorSignature
from ....ops.atomic import OpType


@KernelRegistry.register(
    OpType.MUL,
    [TensorSignature(DType.FP32, (None,)), TensorSignature(DType.FP32, (None,))],
)
def mul_generic_vector(inputs, attrs=None):
    return inputs[0] * inputs[1]


@KernelRegistry.register(
    OpType.MUL,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=(1,))],
)
def mul_tensor_generic_scalar(inputs, attrs=None):
    return inputs[0] * inputs[1]


# Generic Tensor (Any Rank)
@KernelRegistry.register(
    OpType.MUL,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
)
def mul_generic_tensor(inputs, attrs=None):
    return inputs[0] * inputs[1]
